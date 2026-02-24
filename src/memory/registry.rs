//! Memory registry for dynamic memory backend management.
//!
//! The [`MemoryRegistry`] manages a collection of [`MemoryFactory`]
//! implementations, enabling runtime discovery and instantiation of memory
//! backends by name. This extends the existing `create_memory` factory
//! with a registry pattern for plug-and-play memory backends.
//!
//! # Extension
//!
//! To add a new memory backend, implement [`MemoryFactory`] and register
//! it with [`MemoryRegistry::register`].

use super::traits::Memory;
use std::path::Path;

/// Information about a registered memory backend.
#[derive(Debug, Clone)]
pub struct MemoryBackendInfo {
    /// Backend name (e.g., "sqlite", "markdown", "postgres").
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Whether this backend requires external services (e.g., database).
    pub requires_external: bool,
}

/// Factory trait for creating memory backend instances.
///
/// Each memory backend implements this trait to describe itself and
/// create instances from a workspace directory and optional configuration.
pub trait MemoryFactory: Send + Sync {
    /// Canonical name of this memory backend.
    fn name(&self) -> &str;

    /// Human-readable description.
    fn description(&self) -> &str {
        ""
    }

    /// Whether this backend requires external services.
    fn requires_external(&self) -> bool {
        false
    }

    /// Create a memory instance for the given workspace.
    fn create(
        &self,
        workspace_dir: &Path,
        api_key: Option<&str>,
    ) -> anyhow::Result<Box<dyn Memory>>;
}

/// Registry of memory backend factories.
///
/// Manages a collection of [`MemoryFactory`] implementations and provides
/// lookup by name.
pub struct MemoryRegistry {
    factories: Vec<Box<dyn MemoryFactory>>,
}

impl MemoryRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            factories: Vec::new(),
        }
    }

    /// Register a new memory backend factory.
    pub fn register(&mut self, factory: Box<dyn MemoryFactory>) {
        self.factories.push(factory);
    }

    /// Look up a factory by name (case-insensitive).
    pub fn get_factory(&self, name: &str) -> Option<&dyn MemoryFactory> {
        let lower = name.to_ascii_lowercase();
        self.factories
            .iter()
            .find(|f| f.name().to_ascii_lowercase() == lower)
            .map(|f| f.as_ref())
    }

    /// Create a memory backend by name.
    pub fn create_memory(
        &self,
        name: &str,
        workspace_dir: &Path,
        api_key: Option<&str>,
    ) -> anyhow::Result<Box<dyn Memory>> {
        let factory = self
            .get_factory(name)
            .ok_or_else(|| anyhow::anyhow!("Unknown memory backend: {name}"))?;
        factory.create(workspace_dir, api_key)
    }

    /// List all registered memory backends.
    pub fn list_backends(&self) -> Vec<MemoryBackendInfo> {
        self.factories
            .iter()
            .map(|f| MemoryBackendInfo {
                name: f.name().to_string(),
                description: f.description().to_string(),
                requires_external: f.requires_external(),
            })
            .collect()
    }

    /// Number of registered factories.
    pub fn count(&self) -> usize {
        self.factories.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::traits::{Memory, MemoryCategory, MemoryEntry};
    use async_trait::async_trait;

    struct MockMemory;

    #[async_trait]
    impl Memory for MockMemory {
        fn name(&self) -> &str {
            "mock"
        }

        async fn store(
            &self,
            _key: &str,
            _content: &str,
            _category: MemoryCategory,
            _session_id: Option<&str>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        async fn recall(
            &self,
            _query: &str,
            _limit: usize,
            _session_id: Option<&str>,
        ) -> anyhow::Result<Vec<MemoryEntry>> {
            Ok(vec![])
        }

        async fn get(&self, _key: &str) -> anyhow::Result<Option<MemoryEntry>> {
            Ok(None)
        }

        async fn list(
            &self,
            _category: Option<&MemoryCategory>,
            _session_id: Option<&str>,
        ) -> anyhow::Result<Vec<MemoryEntry>> {
            Ok(vec![])
        }

        async fn forget(&self, _key: &str) -> anyhow::Result<bool> {
            Ok(true)
        }

        async fn count(&self) -> anyhow::Result<usize> {
            Ok(0)
        }

        async fn health_check(&self) -> bool {
            true
        }
    }

    struct MockMemoryFactory;

    impl MemoryFactory for MockMemoryFactory {
        fn name(&self) -> &str {
            "mock"
        }

        fn description(&self) -> &str {
            "A mock memory backend"
        }

        fn create(
            &self,
            _workspace_dir: &Path,
            _api_key: Option<&str>,
        ) -> anyhow::Result<Box<dyn Memory>> {
            Ok(Box::new(MockMemory))
        }
    }

    #[test]
    fn empty_registry() {
        let registry = MemoryRegistry::new();
        assert_eq!(registry.count(), 0);
        assert!(registry.list_backends().is_empty());
        assert!(registry.get_factory("anything").is_none());
    }

    #[test]
    fn register_and_lookup() {
        let mut registry = MemoryRegistry::new();
        registry.register(Box::new(MockMemoryFactory));
        assert_eq!(registry.count(), 1);
        assert!(registry.get_factory("mock").is_some());
    }

    #[test]
    fn lookup_case_insensitive() {
        let mut registry = MemoryRegistry::new();
        registry.register(Box::new(MockMemoryFactory));
        assert!(registry.get_factory("MOCK").is_some());
        assert!(registry.get_factory("Mock").is_some());
    }

    #[test]
    fn create_memory_success() {
        let mut registry = MemoryRegistry::new();
        registry.register(Box::new(MockMemoryFactory));
        let mem = registry.create_memory("mock", Path::new("/tmp"), None);
        assert!(mem.is_ok());
    }

    #[test]
    fn create_memory_unknown_fails() {
        let registry = MemoryRegistry::new();
        let result = registry.create_memory("nonexistent", Path::new("/tmp"), None);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unknown memory backend"));
    }

    #[test]
    fn list_backends_returns_info() {
        let mut registry = MemoryRegistry::new();
        registry.register(Box::new(MockMemoryFactory));
        let backends = registry.list_backends();
        assert_eq!(backends.len(), 1);
        assert_eq!(backends[0].name, "mock");
        assert_eq!(backends[0].description, "A mock memory backend");
        assert!(!backends[0].requires_external);
    }
}
