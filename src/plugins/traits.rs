//! Plugin trait definitions.
//!
//! The [`Plugin`] trait is the core interface that all plugins implement.
//! The [`PluginApi`] trait provides the registration surface that plugins
//! use to contribute tools, hooks, and provider factories.

use crate::hooks::HookHandler;
use crate::providers::registry::ProviderFactory;
use crate::tools::Tool;

/// Information about a registered plugin.
#[derive(Debug, Clone)]
pub struct PluginInfo {
    /// Unique plugin identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Semantic version string.
    pub version: String,
    /// Short description.
    pub description: String,
}

/// Core plugin trait — implement for any extensibility plugin.
///
/// A plugin goes through two lifecycle phases:
/// 1. **Registration** — `register()` is called with a [`PluginApi`] to
///    register tools, hooks, and provider factories.
/// 2. **Activation** — `activate()` is called after all plugins have
///    registered and the agent is fully wired.
pub trait Plugin: Send + Sync {
    /// Unique identifier for this plugin (e.g., "my-company/my-plugin").
    fn id(&self) -> &str;

    /// Human-readable plugin name.
    fn name(&self) -> &str;

    /// Semantic version of this plugin.
    fn version(&self) -> &str;

    /// Short description of what this plugin provides.
    fn description(&self) -> &str;

    /// Register all extensions this plugin provides.
    ///
    /// Called during the registration phase. Use the [`PluginApi`] to
    /// register tools, hooks, and provider factories.
    fn register(&self, api: &mut dyn PluginApi) -> anyhow::Result<()>;

    /// Activate the plugin after all registrations are complete.
    ///
    /// Called during the activation phase. Override for setup that depends
    /// on other plugins being registered. Default implementation is a no-op.
    fn activate(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// API surface available to plugins during registration.
///
/// Plugins use this trait to register their contributions (tools, hooks,
/// provider factories) with the agent runtime. The concrete implementation
/// ([`super::DefaultPluginApi`]) collects all registrations for later wiring.
pub trait PluginApi: Send + Sync {
    /// Register a tool implementation.
    fn register_tool(&mut self, tool: Box<dyn Tool>);

    /// Register a hook handler for lifecycle events.
    fn register_hook(&mut self, handler: Box<dyn HookHandler>);

    /// Register a provider factory for LLM provider discovery.
    fn register_provider_factory(&mut self, factory: Box<dyn ProviderFactory>);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_info_debug() {
        let info = PluginInfo {
            id: "test".to_string(),
            name: "Test".to_string(),
            version: "1.0.0".to_string(),
            description: "A test plugin".to_string(),
        };
        let debug = format!("{info:?}");
        assert!(debug.contains("test"));
        assert!(debug.contains("1.0.0"));
    }
}
