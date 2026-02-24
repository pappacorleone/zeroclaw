//! Plugin system for extensible agent functionality.
//!
//! The plugin system provides a compile-time extensibility mechanism inspired
//! by OpenClaw's `OpenClawPluginApi`. Plugins can register:
//!
//! - **Tools** — Additional tool implementations
//! - **Hooks** — Lifecycle event handlers
//! - **Providers** — LLM provider factories
//!
//! Channels, CLI commands, HTTP routes, and services are deferred to a future
//! phase.
//!
//! # Architecture
//!
//! Plugins implement the [`Plugin`] trait and register their extensions via
//! the [`PluginApi`] trait during the `register` phase. The [`PluginRegistry`]
//! manages plugin lifecycle: discovery → registration → activation.
//!
//! Plugin registration uses compile-time feature flags. Each plugin is gated
//! behind a Cargo feature flag and registered in [`PluginRegistry::with_defaults`].
//!
//! # Example
//!
//! ```rust,ignore
//! struct MyPlugin;
//!
//! impl Plugin for MyPlugin {
//!     fn id(&self) -> &str { "my-plugin" }
//!     fn name(&self) -> &str { "My Plugin" }
//!     fn version(&self) -> &str { "0.1.0" }
//!     fn description(&self) -> &str { "Adds custom tools" }
//!
//!     fn register(&self, api: &mut dyn PluginApi) -> anyhow::Result<()> {
//!         api.register_tool(Box::new(MyCustomTool));
//!         Ok(())
//!     }
//! }
//! ```

pub mod traits;

pub use traits::{Plugin, PluginApi, PluginInfo};

use crate::hooks::HookHandler;
use crate::providers::registry::ProviderFactory;
use crate::tools::Tool;

/// Concrete implementation of [`PluginApi`] that collects registrations.
///
/// This struct is passed to each plugin during the registration phase.
/// After all plugins have registered, the collected tools, hooks, and
/// provider factories can be extracted and wired into the agent.
pub struct DefaultPluginApi {
    tools: Vec<Box<dyn Tool>>,
    hooks: Vec<Box<dyn HookHandler>>,
    provider_factories: Vec<Box<dyn ProviderFactory>>,
}

impl DefaultPluginApi {
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            hooks: Vec::new(),
            provider_factories: Vec::new(),
        }
    }

    /// Extract all registered tools.
    pub fn take_tools(&mut self) -> Vec<Box<dyn Tool>> {
        std::mem::take(&mut self.tools)
    }

    /// Extract all registered hooks.
    pub fn take_hooks(&mut self) -> Vec<Box<dyn HookHandler>> {
        std::mem::take(&mut self.hooks)
    }

    /// Extract all registered provider factories.
    pub fn take_provider_factories(&mut self) -> Vec<Box<dyn ProviderFactory>> {
        std::mem::take(&mut self.provider_factories)
    }
}

impl PluginApi for DefaultPluginApi {
    fn register_tool(&mut self, tool: Box<dyn Tool>) {
        tracing::debug!(tool = tool.name(), "Plugin registered tool");
        self.tools.push(tool);
    }

    fn register_hook(&mut self, handler: Box<dyn HookHandler>) {
        tracing::debug!(hook = handler.name(), "Plugin registered hook");
        self.hooks.push(handler);
    }

    fn register_provider_factory(&mut self, factory: Box<dyn ProviderFactory>) {
        tracing::debug!(provider = factory.name(), "Plugin registered provider factory");
        self.provider_factories.push(factory);
    }
}

/// Plugin registry — manages plugin lifecycle.
///
/// The registry discovers, registers, and activates plugins. It uses
/// compile-time feature flags for plugin discovery: each plugin is gated
/// behind a Cargo feature and registered in [`PluginRegistry::with_defaults`].
pub struct PluginRegistry {
    plugins: Vec<Box<dyn Plugin>>,
}

impl PluginRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// Create a registry with all compile-time enabled plugins.
    ///
    /// Plugins gated behind feature flags are registered here. This is the
    /// main extension point for compile-time plugin registration.
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();

        // ── Feature-gated plugin registration ──────────────────────
        //
        // Add compile-time plugins here, gated behind feature flags:
        //
        // #[cfg(feature = "plugin-my-plugin")]
        // registry.register(Box::new(my_plugin::MyPlugin));
        //
        // For now, no built-in plugins are registered. This is the
        // extension point for future plugin development.

        let _ = &mut registry; // suppress unused warning
        registry
    }

    /// Register a plugin.
    pub fn register(&mut self, plugin: Box<dyn Plugin>) {
        tracing::info!(
            plugin_id = plugin.id(),
            plugin_name = plugin.name(),
            plugin_version = plugin.version(),
            "Registering plugin"
        );
        self.plugins.push(plugin);
    }

    /// Run the registration phase for all plugins.
    ///
    /// Each plugin's `register` method is called with a fresh [`PluginApi`].
    /// Returns the collected registrations (tools, hooks, provider factories).
    pub fn run_registration(&self) -> anyhow::Result<DefaultPluginApi> {
        let mut api = DefaultPluginApi::new();

        for plugin in &self.plugins {
            tracing::info!(
                plugin_id = plugin.id(),
                "Running plugin registration"
            );
            plugin.register(&mut api)?;
        }

        Ok(api)
    }

    /// Run the activation phase for all plugins.
    ///
    /// Called after all plugins have registered and the agent is wired up.
    pub fn activate_all(&self) -> anyhow::Result<()> {
        for plugin in &self.plugins {
            tracing::info!(
                plugin_id = plugin.id(),
                "Activating plugin"
            );
            plugin.activate()?;
        }
        Ok(())
    }

    /// List all registered plugins.
    pub fn list_plugins(&self) -> Vec<PluginInfo> {
        self.plugins
            .iter()
            .map(|p| PluginInfo {
                id: p.id().to_string(),
                name: p.name().to_string(),
                version: p.version().to_string(),
                description: p.description().to_string(),
            })
            .collect()
    }

    /// Number of registered plugins.
    pub fn count(&self) -> usize {
        self.plugins.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPlugin;

    impl Plugin for TestPlugin {
        fn id(&self) -> &str {
            "test-plugin"
        }
        fn name(&self) -> &str {
            "Test Plugin"
        }
        fn version(&self) -> &str {
            "1.0.0"
        }
        fn description(&self) -> &str {
            "A test plugin"
        }

        fn register(&self, api: &mut dyn PluginApi) -> anyhow::Result<()> {
            api.register_hook(Box::new(TestHook));
            Ok(())
        }
    }

    struct TestHook;

    #[async_trait::async_trait]
    impl crate::hooks::HookHandler for TestHook {
        fn name(&self) -> &str {
            "test-hook"
        }
    }

    #[test]
    fn empty_registry() {
        let registry = PluginRegistry::new();
        assert_eq!(registry.count(), 0);
        assert!(registry.list_plugins().is_empty());
    }

    #[test]
    fn register_and_list() {
        let mut registry = PluginRegistry::new();
        registry.register(Box::new(TestPlugin));
        assert_eq!(registry.count(), 1);
        let plugins = registry.list_plugins();
        assert_eq!(plugins[0].id, "test-plugin");
        assert_eq!(plugins[0].name, "Test Plugin");
        assert_eq!(plugins[0].version, "1.0.0");
    }

    #[test]
    fn run_registration_collects_hooks() {
        let mut registry = PluginRegistry::new();
        registry.register(Box::new(TestPlugin));
        let mut api = registry.run_registration().unwrap();
        let hooks = api.take_hooks();
        assert_eq!(hooks.len(), 1);
        assert_eq!(hooks[0].name(), "test-hook");
    }

    #[test]
    fn activate_all_succeeds() {
        let mut registry = PluginRegistry::new();
        registry.register(Box::new(TestPlugin));
        assert!(registry.activate_all().is_ok());
    }

    #[test]
    fn with_defaults_creates_empty() {
        let registry = PluginRegistry::with_defaults();
        // No built-in plugins registered yet
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn default_plugin_api_empty() {
        let mut api = DefaultPluginApi::new();
        assert!(api.take_tools().is_empty());
        assert!(api.take_hooks().is_empty());
        assert!(api.take_provider_factories().is_empty());
    }
}
