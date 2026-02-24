//! Provider registry for dynamic provider management.
//!
//! The [`ProviderRegistry`] manages a collection of [`ProviderFactory`]
//! implementations, enabling runtime discovery and instantiation of LLM
//! providers by name or alias. This replaces hard-coded match statements
//! in factory functions with a registry pattern.
//!
//! # Extension
//!
//! To add a new provider backend, implement [`ProviderFactory`] and register
//! it with [`ProviderRegistry::register`].

use super::traits::Provider;

/// Information about a registered provider.
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    /// Canonical provider name (e.g., "openai", "anthropic").
    pub name: String,
    /// Alternative names that resolve to this provider.
    pub aliases: Vec<String>,
    /// Human-readable description.
    pub description: String,
}

/// Factory trait for creating provider instances.
///
/// Each provider backend implements this trait to describe itself and
/// create instances from configuration. The registry uses factories to
/// resolve provider names to concrete implementations.
pub trait ProviderFactory: Send + Sync {
    /// Canonical name of this provider (e.g., "openai").
    fn name(&self) -> &str;

    /// Alternative names that resolve to this provider.
    fn aliases(&self) -> Vec<String> {
        Vec::new()
    }

    /// Human-readable description.
    fn description(&self) -> &str {
        ""
    }

    /// Create a provider instance with the given API key and optional base URL.
    fn create(
        &self,
        api_key: &str,
        base_url: Option<&str>,
    ) -> anyhow::Result<Box<dyn Provider>>;
}

/// Registry of provider factories.
///
/// Manages a collection of [`ProviderFactory`] implementations and provides
/// lookup by name or alias.
pub struct ProviderRegistry {
    factories: Vec<Box<dyn ProviderFactory>>,
}

impl ProviderRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            factories: Vec::new(),
        }
    }

    /// Register a new provider factory.
    pub fn register(&mut self, factory: Box<dyn ProviderFactory>) {
        self.factories.push(factory);
    }

    /// Look up a factory by name or alias (case-insensitive).
    pub fn get_factory(&self, name: &str) -> Option<&dyn ProviderFactory> {
        let lower = name.to_ascii_lowercase();
        self.factories
            .iter()
            .find(|f| {
                f.name().to_ascii_lowercase() == lower
                    || f.aliases()
                        .iter()
                        .any(|a| a.to_ascii_lowercase() == lower)
            })
            .map(|f| f.as_ref())
    }

    /// Create a provider by name/alias with the given API key and optional base URL.
    pub fn create_provider(
        &self,
        name: &str,
        api_key: &str,
        base_url: Option<&str>,
    ) -> anyhow::Result<Box<dyn Provider>> {
        let factory = self
            .get_factory(name)
            .ok_or_else(|| anyhow::anyhow!("Unknown provider: {name}"))?;
        factory.create(api_key, base_url)
    }

    /// List all registered providers.
    pub fn list_providers(&self) -> Vec<ProviderInfo> {
        self.factories
            .iter()
            .map(|f| ProviderInfo {
                name: f.name().to_string(),
                aliases: f.aliases(),
                description: f.description().to_string(),
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
    use crate::providers::traits::{ChatRequest, Provider};
    use async_trait::async_trait;

    struct MockProvider;

    #[async_trait]
    impl Provider for MockProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok("mock".into())
        }
    }

    struct MockFactory;

    impl ProviderFactory for MockFactory {
        fn name(&self) -> &str {
            "mock"
        }

        fn aliases(&self) -> Vec<String> {
            vec!["test-mock".to_string(), "fake".to_string()]
        }

        fn description(&self) -> &str {
            "A mock provider for testing"
        }

        fn create(
            &self,
            _api_key: &str,
            _base_url: Option<&str>,
        ) -> anyhow::Result<Box<dyn Provider>> {
            Ok(Box::new(MockProvider))
        }
    }

    #[test]
    fn empty_registry() {
        let registry = ProviderRegistry::new();
        assert_eq!(registry.count(), 0);
        assert!(registry.list_providers().is_empty());
        assert!(registry.get_factory("anything").is_none());
    }

    #[test]
    fn register_and_lookup_by_name() {
        let mut registry = ProviderRegistry::new();
        registry.register(Box::new(MockFactory));
        assert_eq!(registry.count(), 1);
        assert!(registry.get_factory("mock").is_some());
    }

    #[test]
    fn lookup_by_alias() {
        let mut registry = ProviderRegistry::new();
        registry.register(Box::new(MockFactory));
        assert!(registry.get_factory("test-mock").is_some());
        assert!(registry.get_factory("fake").is_some());
    }

    #[test]
    fn lookup_case_insensitive() {
        let mut registry = ProviderRegistry::new();
        registry.register(Box::new(MockFactory));
        assert!(registry.get_factory("MOCK").is_some());
        assert!(registry.get_factory("Test-Mock").is_some());
    }

    #[test]
    fn unknown_provider_returns_none() {
        let mut registry = ProviderRegistry::new();
        registry.register(Box::new(MockFactory));
        assert!(registry.get_factory("nonexistent").is_none());
    }

    #[test]
    fn create_provider_success() {
        let mut registry = ProviderRegistry::new();
        registry.register(Box::new(MockFactory));
        let provider = registry.create_provider("mock", "key", None);
        assert!(provider.is_ok());
    }

    #[test]
    fn create_provider_unknown_fails() {
        let registry = ProviderRegistry::new();
        let result = registry.create_provider("nonexistent", "key", None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown provider"));
    }

    #[test]
    fn list_providers_returns_info() {
        let mut registry = ProviderRegistry::new();
        registry.register(Box::new(MockFactory));
        let providers = registry.list_providers();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].name, "mock");
        assert_eq!(providers[0].aliases.len(), 2);
        assert_eq!(providers[0].description, "A mock provider for testing");
    }
}
