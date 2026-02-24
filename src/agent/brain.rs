//! Brain trait for pluggable agent orchestration strategies.
//!
//! The [`Brain`] trait abstracts the core agent loop — the cycle of receiving
//! user input, building prompts, calling the LLM, parsing tool calls, executing
//! tools, and producing a final response. Different implementations can provide
//! different orchestration strategies (e.g., agentic multi-turn, single-shot Q&A,
//! autonomous long-running agents).
//!
//! # Extension
//!
//! To add a new brain strategy, implement [`Brain`] and register it in
//! [`create_brain`]. The [`BrainContext`] struct carries all shared dependencies
//! (provider, memory, tools, observer, etc.) so implementations can focus on
//! orchestration logic rather than dependency wiring.

use crate::memory::Memory;
use crate::observability::Observer;
use crate::providers::{ChatMessage, ConversationMessage, Provider};
use crate::tools::ToolSpec;
use async_trait::async_trait;
use std::sync::Arc;

use super::dispatcher::ToolDispatcher;
use super::memory_loader::MemoryLoader;
use super::prompt::SystemPromptBuilder;
use crate::tools::registry::ToolRegistry;

/// Input to a brain turn.
#[derive(Debug, Clone)]
pub struct BrainInput {
    /// The user's message text.
    pub message: String,
    /// Conversation history accumulated so far.
    pub history: Vec<ConversationMessage>,
    /// Optional override for the system prompt.
    pub system_prompt: Option<String>,
}

/// Output from a brain turn.
#[derive(Debug, Clone)]
pub struct BrainOutput {
    /// The final text response from the brain.
    pub response: String,
    /// Number of tool calls that were executed during this turn.
    pub tool_calls_executed: usize,
    /// Updated conversation history (including this turn).
    pub history: Vec<ConversationMessage>,
    /// The reason the brain stopped producing output.
    pub stop_reason: StopReason,
}

/// Why the brain stopped producing output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// Natural end of response (no more tool calls).
    EndTurn,
    /// Exceeded the maximum number of tool iterations.
    MaxIterations,
    /// An error occurred during processing.
    Error(String),
}

/// Capabilities declared by a brain implementation.
#[derive(Debug, Clone, Default)]
pub struct BrainCapabilities {
    /// Whether this brain supports multi-turn tool-calling loops.
    pub supports_tool_calling: bool,
    /// Whether this brain supports streaming responses.
    pub supports_streaming: bool,
    /// Whether this brain supports multi-turn conversations.
    pub supports_multi_turn: bool,
    /// Whether this brain supports extended reasoning / chain-of-thought.
    pub supports_reasoning: bool,
}

/// Shared dependencies for brain execution.
///
/// Carries references to all subsystems a brain implementation might need.
/// Passed by reference to each brain turn so implementations don't need to
/// manage their own dependency wiring.
pub struct BrainContext {
    pub provider: Box<dyn Provider>,
    pub memory: Arc<dyn Memory>,
    pub tool_registry: Box<dyn ToolRegistry>,
    pub tool_specs: Vec<ToolSpec>,
    pub observer: Arc<dyn Observer>,
    pub dispatcher: Box<dyn ToolDispatcher>,
    pub memory_loader: Box<dyn MemoryLoader>,
    pub prompt_builder: SystemPromptBuilder,
    pub config: crate::config::AgentConfig,
    pub model_name: String,
    pub temperature: f64,
    pub workspace_dir: std::path::PathBuf,
    pub identity_config: crate::config::IdentityConfig,
    pub skills: Vec<crate::skills::Skill>,
    pub skills_prompt_mode: crate::config::SkillsPromptInjectionMode,
    pub auto_save: bool,
    pub classification_config: crate::config::QueryClassificationConfig,
    pub available_hints: Vec<String>,
}

/// Core brain trait — implement for any agent orchestration strategy.
///
/// A brain encapsulates the logic of how an agent processes a user message:
/// prompt construction, LLM invocation, tool execution, and response assembly.
/// Different implementations can provide different strategies without changing
/// the surrounding infrastructure.
#[async_trait]
pub trait Brain: Send + Sync {
    /// Human-readable name of this brain strategy.
    fn name(&self) -> &str;

    /// Capabilities of this brain implementation.
    fn capabilities(&self) -> BrainCapabilities;

    /// Run a single turn: user message in, agent response out.
    ///
    /// The brain receives the user message and a mutable context containing
    /// all dependencies. It returns a [`BrainOutput`] with the response text,
    /// tool call count, updated history, and stop reason.
    async fn run_turn(
        &mut self,
        input: BrainInput,
        ctx: &mut BrainContext,
    ) -> anyhow::Result<BrainOutput>;
}

/// The standard agentic brain — multi-turn tool-calling loop.
///
/// This is the default brain that wraps the existing `Agent::turn` logic.
/// It builds a system prompt, enriches user messages with memory context,
/// calls the LLM in a loop, parses and executes tool calls, and returns
/// the final response.
pub struct AgenticBrain;

impl AgenticBrain {
    pub fn new() -> Self {
        Self
    }

    fn classify_model(ctx: &BrainContext, user_message: &str) -> String {
        if let Some(hint) =
            super::classifier::classify(&ctx.classification_config, user_message)
        {
            if ctx.available_hints.contains(&hint) {
                tracing::info!(hint = hint.as_str(), "Auto-classified query");
                return format!("hint:{hint}");
            }
        }
        ctx.model_name.clone()
    }
}

#[async_trait]
impl Brain for AgenticBrain {
    fn name(&self) -> &str {
        "agentic"
    }

    fn capabilities(&self) -> BrainCapabilities {
        BrainCapabilities {
            supports_tool_calling: true,
            supports_streaming: false,
            supports_multi_turn: true,
            supports_reasoning: false,
        }
    }

    async fn run_turn(
        &mut self,
        input: BrainInput,
        ctx: &mut BrainContext,
    ) -> anyhow::Result<BrainOutput> {
        use crate::memory::MemoryCategory;
        use crate::observability::ObserverEvent;
        use crate::providers::ChatRequest;
        use std::io::Write as IoWrite;
        use std::time::Instant;

        let mut history = input.history;

        // Build system prompt if history is empty
        if history.is_empty() {
            let instructions = ctx.dispatcher.prompt_instructions(ctx.tool_registry.list());
            let prompt_ctx = super::prompt::PromptContext {
                workspace_dir: &ctx.workspace_dir,
                model_name: &ctx.model_name,
                tools: ctx.tool_registry.list(),
                skills: &ctx.skills,
                skills_prompt_mode: ctx.skills_prompt_mode,
                identity_config: Some(&ctx.identity_config),
                dispatcher_instructions: &instructions,
            };
            let system_prompt = ctx.prompt_builder.build(&prompt_ctx)?;
            history.push(ConversationMessage::Chat(ChatMessage::system(
                system_prompt,
            )));
        }

        // Auto-save user message to memory
        if ctx.auto_save {
            let _ = ctx
                .memory
                .store("user_msg", &input.message, MemoryCategory::Conversation, None)
                .await;
        }

        // Load memory context
        let context = ctx
            .memory_loader
            .load_context(ctx.memory.as_ref(), &input.message)
            .await
            .unwrap_or_default();

        let enriched = if context.is_empty() {
            input.message.clone()
        } else {
            format!("{context}{}", input.message)
        };

        history.push(ConversationMessage::Chat(ChatMessage::user(enriched)));

        let effective_model = Self::classify_model(ctx, &input.message);
        let mut tool_calls_executed: usize = 0;

        for _ in 0..ctx.config.max_tool_iterations {
            let messages = ctx.dispatcher.to_provider_messages(&history);
            let response = ctx
                .provider
                .chat(
                    ChatRequest {
                        messages: &messages,
                        tools: if ctx.dispatcher.should_send_tool_specs() {
                            Some(&ctx.tool_specs)
                        } else {
                            None
                        },
                    },
                    &effective_model,
                    ctx.temperature,
                )
                .await?;

            let (text, calls) = ctx.dispatcher.parse_response(&response);

            if calls.is_empty() {
                let final_text = if text.is_empty() {
                    response.text.unwrap_or_default()
                } else {
                    text
                };

                history.push(ConversationMessage::Chat(ChatMessage::assistant(
                    final_text.clone(),
                )));
                trim_history(&mut history, ctx.config.max_history_messages);

                return Ok(BrainOutput {
                    response: final_text,
                    tool_calls_executed,
                    history,
                    stop_reason: StopReason::EndTurn,
                });
            }

            if !text.is_empty() {
                history.push(ConversationMessage::Chat(ChatMessage::assistant(
                    text.clone(),
                )));
                print!("{text}");
                let _ = std::io::stdout().flush();
            }

            history.push(ConversationMessage::AssistantToolCalls {
                text: response.text.clone(),
                tool_calls: response.tool_calls.clone(),
                reasoning_content: response.reasoning_content.clone(),
            });

            // Execute tool calls
            let mut results = Vec::with_capacity(calls.len());
            for call in &calls {
                let start = Instant::now();
                let (output, succeeded) =
                    if let Some(tool) = ctx.tool_registry.get(&call.name) {
                        match tool.execute(call.arguments.clone()).await {
                            Ok(r) => {
                                ctx.observer.record_event(&ObserverEvent::ToolCall {
                                    tool: call.name.clone(),
                                    duration: start.elapsed(),
                                    success: r.success,
                                });
                                if r.success {
                                    (r.output, true)
                                } else {
                                    (format!("Error: {}", r.error.unwrap_or(r.output)), false)
                                }
                            }
                            Err(e) => {
                                ctx.observer.record_event(&ObserverEvent::ToolCall {
                                    tool: call.name.clone(),
                                    duration: start.elapsed(),
                                    success: false,
                                });
                                (format!("Error executing {}: {e}", call.name), false)
                            }
                        }
                    } else {
                        (format!("Unknown tool: {}", call.name), false)
                    };

                results.push(super::dispatcher::ToolExecutionResult {
                    name: call.name.clone(),
                    output,
                    success: succeeded,
                    tool_call_id: call.tool_call_id.clone(),
                });
                tool_calls_executed += 1;
            }

            let formatted = ctx.dispatcher.format_results(&results);
            history.push(formatted);
            trim_history(&mut history, ctx.config.max_history_messages);
        }

        Ok(BrainOutput {
            response: format!(
                "Agent exceeded maximum tool iterations ({})",
                ctx.config.max_tool_iterations
            ),
            tool_calls_executed,
            history,
            stop_reason: StopReason::MaxIterations,
        })
    }
}

/// A simple single-shot brain — no tool calling, just LLM Q&A.
pub struct SimpleBrain;

impl SimpleBrain {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Brain for SimpleBrain {
    fn name(&self) -> &str {
        "simple"
    }

    fn capabilities(&self) -> BrainCapabilities {
        BrainCapabilities {
            supports_tool_calling: false,
            supports_streaming: false,
            supports_multi_turn: false,
            supports_reasoning: false,
        }
    }

    async fn run_turn(
        &mut self,
        input: BrainInput,
        ctx: &mut BrainContext,
    ) -> anyhow::Result<BrainOutput> {
        let response = ctx
            .provider
            .simple_chat(&input.message, &ctx.model_name, ctx.temperature)
            .await?;

        let mut history = input.history;
        history.push(ConversationMessage::Chat(ChatMessage::user(
            input.message,
        )));
        history.push(ConversationMessage::Chat(ChatMessage::assistant(
            response.clone(),
        )));

        Ok(BrainOutput {
            response,
            tool_calls_executed: 0,
            history,
            stop_reason: StopReason::EndTurn,
        })
    }
}

/// Factory: create a brain from a mode string.
pub fn create_brain(mode: &str) -> Box<dyn Brain> {
    match mode {
        "simple" => Box::new(SimpleBrain::new()),
        "agentic" | _ => Box::new(AgenticBrain::new()),
    }
}

/// Trim conversation history, preserving system messages.
fn trim_history(history: &mut Vec<ConversationMessage>, max: usize) {
    if history.len() <= max {
        return;
    }

    let mut system_messages = Vec::new();
    let mut other_messages = Vec::new();

    for msg in history.drain(..) {
        match &msg {
            ConversationMessage::Chat(chat) if chat.role == "system" => {
                system_messages.push(msg);
            }
            _ => other_messages.push(msg),
        }
    }

    if other_messages.len() > max {
        let drop_count = other_messages.len() - max;
        other_messages.drain(0..drop_count);
    }

    *history = system_messages;
    history.extend(other_messages);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agentic_brain_name() {
        let brain = AgenticBrain::new();
        assert_eq!(brain.name(), "agentic");
    }

    #[test]
    fn simple_brain_name() {
        let brain = SimpleBrain::new();
        assert_eq!(brain.name(), "simple");
    }

    #[test]
    fn agentic_brain_capabilities() {
        let brain = AgenticBrain::new();
        let caps = brain.capabilities();
        assert!(caps.supports_tool_calling);
        assert!(caps.supports_multi_turn);
    }

    #[test]
    fn simple_brain_capabilities() {
        let brain = SimpleBrain::new();
        let caps = brain.capabilities();
        assert!(!caps.supports_tool_calling);
        assert!(!caps.supports_multi_turn);
    }

    #[test]
    fn create_brain_agentic() {
        let brain = create_brain("agentic");
        assert_eq!(brain.name(), "agentic");
    }

    #[test]
    fn create_brain_simple() {
        let brain = create_brain("simple");
        assert_eq!(brain.name(), "simple");
    }

    #[test]
    fn create_brain_default_is_agentic() {
        let brain = create_brain("unknown");
        assert_eq!(brain.name(), "agentic");
    }

    #[test]
    fn stop_reason_equality() {
        assert_eq!(StopReason::EndTurn, StopReason::EndTurn);
        assert_eq!(StopReason::MaxIterations, StopReason::MaxIterations);
        assert_ne!(StopReason::EndTurn, StopReason::MaxIterations);
    }

    #[test]
    fn trim_history_preserves_system() {
        let mut history = vec![
            ConversationMessage::Chat(ChatMessage::system("sys")),
            ConversationMessage::Chat(ChatMessage::user("u1")),
            ConversationMessage::Chat(ChatMessage::assistant("a1")),
            ConversationMessage::Chat(ChatMessage::user("u2")),
            ConversationMessage::Chat(ChatMessage::assistant("a2")),
        ];
        trim_history(&mut history, 2);
        // System message preserved + last 2 non-system messages
        assert_eq!(history.len(), 3);
        match &history[0] {
            ConversationMessage::Chat(msg) => assert_eq!(msg.role, "system"),
            _ => panic!("expected system message"),
        }
    }
}
