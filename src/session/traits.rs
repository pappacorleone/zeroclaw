//! Session trait for conversation history management.
//!
//! The [`Session`] trait provides a backend-agnostic interface for storing,
//! retrieving, compacting, and clearing conversation transcripts. Each session
//! is identified by a unique string ID and supports concurrent access via
//! `Send + Sync`.

use crate::providers::ConversationMessage;
use async_trait::async_trait;

/// Result of a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// Number of messages that were removed.
    pub messages_removed: usize,
    /// Number of messages remaining after compaction.
    pub messages_remaining: usize,
}

/// Metadata about a session.
#[derive(Debug, Clone)]
pub struct SessionMetadata {
    /// Number of messages in the session.
    pub message_count: usize,
    /// Backend type name (e.g., "memory", "jsonl", "sqlite").
    pub backend: String,
}

/// Core session trait — implement for any conversation persistence backend.
#[async_trait]
pub trait Session: Send + Sync {
    /// Unique identifier for this session.
    fn id(&self) -> &str;

    /// Append a message to the session transcript.
    async fn append(&self, message: ConversationMessage) -> anyhow::Result<()>;

    /// Retrieve the full conversation history.
    async fn history(&self) -> anyhow::Result<Vec<ConversationMessage>>;

    /// Compact the session by removing old messages, keeping `keep_recent`
    /// non-system messages plus all system messages.
    async fn compact(&self, keep_recent: usize) -> anyhow::Result<CompactionResult>;

    /// Clear all messages from the session.
    async fn clear(&self) -> anyhow::Result<()>;

    /// Return metadata about this session.
    async fn metadata(&self) -> anyhow::Result<SessionMetadata>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compaction_result_debug() {
        let result = CompactionResult {
            messages_removed: 5,
            messages_remaining: 10,
        };
        let debug = format!("{result:?}");
        assert!(debug.contains("messages_removed: 5"));
        assert!(debug.contains("messages_remaining: 10"));
    }

    #[test]
    fn session_metadata_debug() {
        let meta = SessionMetadata {
            message_count: 42,
            backend: "memory".to_string(),
        };
        let debug = format!("{meta:?}");
        assert!(debug.contains("message_count: 42"));
        assert!(debug.contains("memory"));
    }
}
