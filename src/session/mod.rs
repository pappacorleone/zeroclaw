//! Session management for conversation persistence.
//!
//! The [`Session`] trait abstracts conversation history storage and retrieval.
//! Implementations can persist transcripts in-memory, on disk (JSONL), or in
//! a database. The agent loop uses sessions to maintain conversation state
//! across turns.
//!
//! # Implementations
//!
//! - [`InMemorySession`] — Ephemeral in-memory storage (default).
//! - [`JsonlSession`] — JSONL file-backed persistence (inspired by OpenClaw).
//!
//! # Extension
//!
//! To add a new session backend, implement [`Session`] and register it in
//! [`create_session`].

pub mod traits;

pub use traits::{CompactionResult, Session, SessionMetadata};

use crate::providers::ConversationMessage;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::sync::RwLock;

/// Create a session from a mode string.
///
/// Supported modes:
/// - `"memory"` or `"in-memory"` — ephemeral in-memory session
/// - `"jsonl"` — JSONL file-backed session (requires `path`)
/// - anything else defaults to in-memory
pub fn create_session(mode: &str, id: &str, path: Option<&Path>) -> Box<dyn Session> {
    match mode {
        "jsonl" => {
            let dir = path.unwrap_or_else(|| Path::new("/tmp"));
            Box::new(JsonlSession::new(id, dir))
        }
        "memory" | "in-memory" | _ => Box::new(InMemorySession::new(id)),
    }
}

/// Ephemeral in-memory session.
pub struct InMemorySession {
    id: String,
    history: RwLock<Vec<ConversationMessage>>,
}

impl InMemorySession {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            history: RwLock::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Session for InMemorySession {
    fn id(&self) -> &str {
        &self.id
    }

    async fn append(&self, message: ConversationMessage) -> anyhow::Result<()> {
        self.history.write().await.push(message);
        Ok(())
    }

    async fn history(&self) -> anyhow::Result<Vec<ConversationMessage>> {
        Ok(self.history.read().await.clone())
    }

    async fn compact(&self, keep_recent: usize) -> anyhow::Result<CompactionResult> {
        let mut guard = self.history.write().await;
        let original_len = guard.len();
        if original_len <= keep_recent {
            return Ok(CompactionResult {
                messages_removed: 0,
                messages_remaining: original_len,
            });
        }

        // Preserve system messages and last `keep_recent` non-system messages
        let mut system_msgs = Vec::new();
        let mut other_msgs = Vec::new();
        for msg in guard.drain(..) {
            match &msg {
                ConversationMessage::Chat(chat) if chat.role == "system" => {
                    system_msgs.push(msg);
                }
                _ => other_msgs.push(msg),
            }
        }

        let drop_count = other_msgs.len().saturating_sub(keep_recent);
        if drop_count > 0 {
            other_msgs.drain(0..drop_count);
        }

        let remaining = system_msgs.len() + other_msgs.len();
        *guard = system_msgs;
        guard.extend(other_msgs);

        Ok(CompactionResult {
            messages_removed: original_len - remaining,
            messages_remaining: remaining,
        })
    }

    async fn clear(&self) -> anyhow::Result<()> {
        self.history.write().await.clear();
        Ok(())
    }

    async fn metadata(&self) -> anyhow::Result<SessionMetadata> {
        let len = self.history.read().await.len();
        Ok(SessionMetadata {
            message_count: len,
            backend: "memory".to_string(),
        })
    }
}

/// A single JSONL line entry for session persistence.
#[derive(Debug, Serialize, Deserialize)]
struct JsonlEntry {
    /// ISO 8601 timestamp
    ts: String,
    /// The conversation message
    msg: ConversationMessage,
}

/// JSONL file-backed session (inspired by OpenClaw's session persistence).
///
/// Each message is appended as a single JSON line to a `.jsonl` file.
/// On load, the file is read line-by-line to reconstruct history.
pub struct JsonlSession {
    id: String,
    path: PathBuf,
    /// In-memory cache of history (kept in sync with file)
    history: RwLock<Vec<ConversationMessage>>,
}

impl JsonlSession {
    /// Create a new JSONL session. The file is stored at `dir/{id}.jsonl`.
    pub fn new(id: &str, dir: &Path) -> Self {
        let path = dir.join(format!("{id}.jsonl"));
        let history = Self::load_from_file(&path).unwrap_or_default();
        Self {
            id: id.to_string(),
            path,
            history: RwLock::new(history),
        }
    }

    /// Load history from an existing JSONL file.
    fn load_from_file(path: &Path) -> anyhow::Result<Vec<ConversationMessage>> {
        if !path.exists() {
            return Ok(Vec::new());
        }

        let content = std::fs::read_to_string(path)?;
        let mut messages = Vec::new();
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<JsonlEntry>(trimmed) {
                Ok(entry) => messages.push(entry.msg),
                Err(e) => {
                    tracing::warn!("Skipping malformed JSONL line: {e}");
                }
            }
        }
        Ok(messages)
    }

    /// Append a single entry to the JSONL file.
    fn append_to_file(&self, message: &ConversationMessage) -> anyhow::Result<()> {
        use std::io::Write;

        let entry = JsonlEntry {
            ts: chrono::Utc::now().to_rfc3339(),
            msg: message.clone(),
        };
        let line = serde_json::to_string(&entry)?;

        // Ensure parent directory exists
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(file, "{line}")?;
        Ok(())
    }

    /// Rewrite the entire JSONL file from the in-memory history.
    fn rewrite_file(&self, messages: &[ConversationMessage]) -> anyhow::Result<()> {
        use std::io::Write;

        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = std::fs::File::create(&self.path)?;
        for msg in messages {
            let entry = JsonlEntry {
                ts: chrono::Utc::now().to_rfc3339(),
                msg: msg.clone(),
            };
            let line = serde_json::to_string(&entry)?;
            writeln!(file, "{line}")?;
        }
        Ok(())
    }
}

#[async_trait]
impl Session for JsonlSession {
    fn id(&self) -> &str {
        &self.id
    }

    async fn append(&self, message: ConversationMessage) -> anyhow::Result<()> {
        // Append to file first (durable), then update cache
        self.append_to_file(&message)?;
        self.history.write().await.push(message);
        Ok(())
    }

    async fn history(&self) -> anyhow::Result<Vec<ConversationMessage>> {
        Ok(self.history.read().await.clone())
    }

    async fn compact(&self, keep_recent: usize) -> anyhow::Result<CompactionResult> {
        let mut guard = self.history.write().await;
        let original_len = guard.len();
        if original_len <= keep_recent {
            return Ok(CompactionResult {
                messages_removed: 0,
                messages_remaining: original_len,
            });
        }

        let mut system_msgs = Vec::new();
        let mut other_msgs = Vec::new();
        for msg in guard.drain(..) {
            match &msg {
                ConversationMessage::Chat(chat) if chat.role == "system" => {
                    system_msgs.push(msg);
                }
                _ => other_msgs.push(msg),
            }
        }

        let drop_count = other_msgs.len().saturating_sub(keep_recent);
        if drop_count > 0 {
            other_msgs.drain(0..drop_count);
        }

        let remaining = system_msgs.len() + other_msgs.len();
        *guard = system_msgs;
        guard.extend(other_msgs);

        // Rewrite the file with compacted history
        self.rewrite_file(&guard)?;

        Ok(CompactionResult {
            messages_removed: original_len - remaining,
            messages_remaining: remaining,
        })
    }

    async fn clear(&self) -> anyhow::Result<()> {
        self.history.write().await.clear();
        // Truncate the file
        if self.path.exists() {
            std::fs::write(&self.path, "")?;
        }
        Ok(())
    }

    async fn metadata(&self) -> anyhow::Result<SessionMetadata> {
        let len = self.history.read().await.len();
        Ok(SessionMetadata {
            message_count: len,
            backend: "jsonl".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::ChatMessage;

    #[tokio::test]
    async fn in_memory_session_append_and_history() {
        let session = InMemorySession::new("test-session");
        assert_eq!(session.id(), "test-session");

        session
            .append(ConversationMessage::Chat(ChatMessage::user("hello")))
            .await
            .unwrap();
        session
            .append(ConversationMessage::Chat(ChatMessage::assistant("hi")))
            .await
            .unwrap();

        let history = session.history().await.unwrap();
        assert_eq!(history.len(), 2);
    }

    #[tokio::test]
    async fn in_memory_session_clear() {
        let session = InMemorySession::new("test");
        session
            .append(ConversationMessage::Chat(ChatMessage::user("hello")))
            .await
            .unwrap();
        session.clear().await.unwrap();
        assert!(session.history().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn in_memory_session_compact() {
        let session = InMemorySession::new("test");
        session
            .append(ConversationMessage::Chat(ChatMessage::system("sys")))
            .await
            .unwrap();
        for i in 0..5 {
            session
                .append(ConversationMessage::Chat(ChatMessage::user(format!(
                    "msg {i}"
                ))))
                .await
                .unwrap();
        }

        let result = session.compact(2).await.unwrap();
        assert_eq!(result.messages_removed, 3);
        assert_eq!(result.messages_remaining, 3); // 1 system + 2 recent

        let history = session.history().await.unwrap();
        assert_eq!(history.len(), 3);
        match &history[0] {
            ConversationMessage::Chat(msg) => assert_eq!(msg.role, "system"),
            _ => panic!("expected system message"),
        }
    }

    #[tokio::test]
    async fn in_memory_session_metadata() {
        let session = InMemorySession::new("test");
        session
            .append(ConversationMessage::Chat(ChatMessage::user("hi")))
            .await
            .unwrap();
        let meta = session.metadata().await.unwrap();
        assert_eq!(meta.message_count, 1);
        assert_eq!(meta.backend, "memory");
    }

    #[tokio::test]
    async fn jsonl_session_persist_and_reload() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();

        // Create session and add messages
        {
            let session = JsonlSession::new("s1", dir);
            session
                .append(ConversationMessage::Chat(ChatMessage::user("hello")))
                .await
                .unwrap();
            session
                .append(ConversationMessage::Chat(ChatMessage::assistant(
                    "world",
                )))
                .await
                .unwrap();

            let history = session.history().await.unwrap();
            assert_eq!(history.len(), 2);
        }

        // Reload from file
        {
            let session = JsonlSession::new("s1", dir);
            let history = session.history().await.unwrap();
            assert_eq!(history.len(), 2);
        }
    }

    #[tokio::test]
    async fn jsonl_session_clear() {
        let tmp = tempfile::tempdir().unwrap();
        let session = JsonlSession::new("s1", tmp.path());
        session
            .append(ConversationMessage::Chat(ChatMessage::user("hi")))
            .await
            .unwrap();
        session.clear().await.unwrap();
        assert!(session.history().await.unwrap().is_empty());

        // Reload should also be empty
        let reloaded = JsonlSession::new("s1", tmp.path());
        assert!(reloaded.history().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn jsonl_session_compact() {
        let tmp = tempfile::tempdir().unwrap();
        let session = JsonlSession::new("s1", tmp.path());
        session
            .append(ConversationMessage::Chat(ChatMessage::system("sys")))
            .await
            .unwrap();
        for i in 0..5 {
            session
                .append(ConversationMessage::Chat(ChatMessage::user(format!(
                    "msg {i}"
                ))))
                .await
                .unwrap();
        }

        let result = session.compact(2).await.unwrap();
        assert_eq!(result.messages_removed, 3);
        assert_eq!(result.messages_remaining, 3);

        // After compaction, reload from file should have the compacted history
        let reloaded = JsonlSession::new("s1", tmp.path());
        let history = reloaded.history().await.unwrap();
        assert_eq!(history.len(), 3);
    }

    #[tokio::test]
    async fn jsonl_session_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let session = JsonlSession::new("s1", tmp.path());
        session
            .append(ConversationMessage::Chat(ChatMessage::user("hi")))
            .await
            .unwrap();
        let meta = session.metadata().await.unwrap();
        assert_eq!(meta.message_count, 1);
        assert_eq!(meta.backend, "jsonl");
    }

    #[test]
    fn create_session_memory() {
        let session = create_session("memory", "test", None);
        assert_eq!(session.id(), "test");
    }

    #[test]
    fn create_session_jsonl() {
        let tmp = tempfile::tempdir().unwrap();
        let session = create_session("jsonl", "test", Some(tmp.path()));
        assert_eq!(session.id(), "test");
    }

    #[test]
    fn create_session_default() {
        let session = create_session("unknown", "test", None);
        assert_eq!(session.id(), "test");
    }
}
