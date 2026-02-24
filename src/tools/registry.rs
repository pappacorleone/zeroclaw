//! Tool registry for dynamic tool management.
//!
//! The [`ToolRegistry`] trait abstracts tool collection management, enabling
//! runtime registration, lookup, and removal of tools. The default
//! implementation [`DefaultToolRegistry`] wraps a `Vec<Box<dyn Tool>>` and
//! provides O(n) lookup by name.
//!
//! # Extension
//!
//! To provide a custom tool registry (e.g., one that loads tools from a database
//! or plugin system), implement the [`ToolRegistry`] trait.

use super::traits::{Tool, ToolSpec};

/// Trait for managing a collection of tools.
///
/// Provides dynamic registration, lookup, listing, and removal of tools.
/// Implementations are expected to be used during agent setup and within
/// the brain's tool-calling loop.
pub trait ToolRegistry: Send + Sync {
    /// Register a new tool. If a tool with the same name already exists,
    /// it is replaced.
    fn register(&mut self, tool: Box<dyn Tool>);

    /// Look up a tool by name.
    fn get(&self, name: &str) -> Option<&dyn Tool>;

    /// Return all registered tools as a slice.
    fn list(&self) -> &[Box<dyn Tool>];

    /// Return tool specifications for all registered tools.
    fn specs(&self) -> Vec<ToolSpec> {
        self.list().iter().map(|t| t.spec()).collect()
    }

    /// Remove a tool by name. Returns true if a tool was removed.
    fn remove(&mut self, name: &str) -> bool;

    /// Number of registered tools.
    fn count(&self) -> usize {
        self.list().len()
    }

    /// Check if a tool with the given name is registered.
    fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }
}

/// Default tool registry backed by a Vec.
pub struct DefaultToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl DefaultToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    /// Create a registry pre-populated with tools.
    pub fn with_tools(tools: Vec<Box<dyn Tool>>) -> Self {
        Self { tools }
    }
}

impl ToolRegistry for DefaultToolRegistry {
    fn register(&mut self, tool: Box<dyn Tool>) {
        let name = tool.name().to_string();
        self.tools.retain(|t| t.name() != name);
        self.tools.push(tool);
    }

    fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|t| t.name() == name)
            .map(|t| t.as_ref())
    }

    fn list(&self) -> &[Box<dyn Tool>] {
        &self.tools
    }

    fn remove(&mut self, name: &str) -> bool {
        let before = self.tools.len();
        self.tools.retain(|t| t.name() != name);
        self.tools.len() < before
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct FakeTool {
        tool_name: String,
    }

    impl FakeTool {
        fn new(name: &str) -> Self {
            Self {
                tool_name: name.to_string(),
            }
        }
    }

    #[async_trait]
    impl Tool for FakeTool {
        fn name(&self) -> &str {
            &self.tool_name
        }

        fn description(&self) -> &str {
            "fake"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
        ) -> anyhow::Result<crate::tools::ToolResult> {
            Ok(crate::tools::ToolResult {
                success: true,
                output: "ok".into(),
                error: None,
            })
        }
    }

    #[test]
    fn empty_registry() {
        let registry = DefaultToolRegistry::new();
        assert_eq!(registry.count(), 0);
        assert!(registry.list().is_empty());
        assert!(registry.get("anything").is_none());
    }

    #[test]
    fn register_and_get() {
        let mut registry = DefaultToolRegistry::new();
        registry.register(Box::new(FakeTool::new("shell")));
        assert_eq!(registry.count(), 1);
        assert!(registry.contains("shell"));
        assert!(!registry.contains("unknown"));

        let tool = registry.get("shell").unwrap();
        assert_eq!(tool.name(), "shell");
    }

    #[test]
    fn register_replaces_existing() {
        let mut registry = DefaultToolRegistry::new();
        registry.register(Box::new(FakeTool::new("shell")));
        registry.register(Box::new(FakeTool::new("shell")));
        assert_eq!(registry.count(), 1);
    }

    #[test]
    fn with_tools_constructor() {
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(FakeTool::new("a")),
            Box::new(FakeTool::new("b")),
        ];
        let registry = DefaultToolRegistry::with_tools(tools);
        assert_eq!(registry.count(), 2);
        assert!(registry.contains("a"));
        assert!(registry.contains("b"));
    }

    #[test]
    fn remove_tool() {
        let mut registry = DefaultToolRegistry::new();
        registry.register(Box::new(FakeTool::new("shell")));
        assert!(registry.remove("shell"));
        assert_eq!(registry.count(), 0);
        assert!(!registry.remove("shell"));
    }

    #[test]
    fn specs_returns_all() {
        let mut registry = DefaultToolRegistry::new();
        registry.register(Box::new(FakeTool::new("a")));
        registry.register(Box::new(FakeTool::new("b")));
        let specs = registry.specs();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].name, "a");
        assert_eq!(specs[1].name, "b");
    }
}
