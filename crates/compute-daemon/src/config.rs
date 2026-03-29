use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub node: NodeConfig,
    pub wallet: WalletConfig,
    pub network: NetworkConfig,
    pub docker: DockerConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub name: String,
    pub max_gpu_usage: u8,
    pub max_cpu_usage: u8,
    pub idle_threshold_minutes: u32,
    pub pause_on_battery: bool,
    pub pause_on_fullscreen: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletConfig {
    pub public_address: String,
    #[serde(default)]
    pub node_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub orchestrator_url: String,
    pub region: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerConfig {
    pub socket: String,
    pub image_cache_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub file: String,
    pub max_size_mb: u32,
}

impl Default for Config {
    fn default() -> Self {
        let compute_dir = config_dir().unwrap_or_else(|| PathBuf::from("~/.compute"));
        Self {
            node: NodeConfig {
                name: hostname(),
                max_gpu_usage: 90,
                max_cpu_usage: 50,
                idle_threshold_minutes: 5,
                pause_on_battery: true,
                pause_on_fullscreen: true,
            },
            wallet: WalletConfig { public_address: String::new(), node_id: String::new() },
            network: NetworkConfig {
                orchestrator_url: "https://api.computenetwork.sh"
                    .into(),
                region: "auto".into(),
            },
            docker: DockerConfig {
                socket: default_docker_socket(),
                image_cache_dir: compute_dir.join("images").to_string_lossy().into_owned(),
            },
            logging: LoggingConfig {
                level: "info".into(),
                file: compute_dir.join("logs").join("compute.log").to_string_lossy().into_owned(),
                max_size_mb: 100,
            },
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let path = config_file_path()?;
        if !path.exists() {
            return Ok(Config::default());
        }
        let contents = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        let config: Config =
            toml::from_str(&contents).with_context(|| "Failed to parse config.toml")?;
        Ok(config)
    }

    pub fn save(&self) -> Result<()> {
        let path = config_file_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create config directory: {}", parent.display())
            })?;
        }
        let contents =
            toml::to_string_pretty(self).with_context(|| "Failed to serialize config")?;
        std::fs::write(&path, contents)
            .with_context(|| format!("Failed to write config file: {}", path.display()))?;
        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<String> {
        match key {
            "node.name" => Some(self.node.name.clone()),
            "node.max_gpu_usage" => Some(self.node.max_gpu_usage.to_string()),
            "node.max_cpu_usage" => Some(self.node.max_cpu_usage.to_string()),
            "node.idle_threshold_minutes" => Some(self.node.idle_threshold_minutes.to_string()),
            "node.pause_on_battery" => Some(self.node.pause_on_battery.to_string()),
            "node.pause_on_fullscreen" => Some(self.node.pause_on_fullscreen.to_string()),
            "wallet.public_address" => Some(self.wallet.public_address.clone()),
            "wallet.node_id" => Some(self.wallet.node_id.clone()),
            "network.orchestrator_url" => Some(self.network.orchestrator_url.clone()),
            "network.region" => Some(self.network.region.clone()),
            "logging.level" => Some(self.logging.level.clone()),
            _ => None,
        }
    }

    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "node.name" => self.node.name = value.to_string(),
            "node.max_gpu_usage" => self.node.max_gpu_usage = value.parse()?,
            "node.max_cpu_usage" => self.node.max_cpu_usage = value.parse()?,
            "node.idle_threshold_minutes" => self.node.idle_threshold_minutes = value.parse()?,
            "node.pause_on_battery" => self.node.pause_on_battery = value.parse()?,
            "node.pause_on_fullscreen" => self.node.pause_on_fullscreen = value.parse()?,
            "wallet.public_address" => self.wallet.public_address = value.to_string(),
            "wallet.node_id" => self.wallet.node_id = value.to_string(),
            "network.orchestrator_url" => self.network.orchestrator_url = value.to_string(),
            "network.region" => self.network.region = value.to_string(),
            "logging.level" => self.logging.level = value.to_string(),
            _ => anyhow::bail!("Unknown config key: {key}"),
        }
        Ok(())
    }
}

/// Returns the compute config directory path, platform-aware.
pub fn config_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        dirs::data_dir().map(|d| d.join("compute"))
    }
    #[cfg(not(target_os = "windows"))]
    {
        dirs::home_dir().map(|d| d.join(".compute"))
    }
}

/// Returns the path to config.toml.
pub fn config_file_path() -> Result<PathBuf> {
    config_dir()
        .map(|d| d.join("config.toml"))
        .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))
}

/// Returns the path to the PID file.
pub fn pid_file_path() -> Result<PathBuf> {
    config_dir()
        .map(|d| d.join("compute.pid"))
        .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))
}

/// Returns the logs directory.
pub fn logs_dir() -> Result<PathBuf> {
    config_dir()
        .map(|d| d.join("logs"))
        .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))
}

fn default_docker_socket() -> String {
    #[cfg(target_os = "windows")]
    {
        "//./pipe/docker_engine".into()
    }
    #[cfg(not(target_os = "windows"))]
    {
        "/var/run/docker.sock".into()
    }
}

fn hostname() -> String {
    sysinfo::System::host_name().unwrap_or_else(|| "compute-node".into())
}

/// Check if a config file exists.
pub fn config_exists() -> bool {
    config_file_path().map(|p| p.exists()).unwrap_or(false)
}

/// Ensure the compute directory structure exists.
pub fn ensure_dirs() -> Result<()> {
    if let Some(dir) = config_dir() {
        std::fs::create_dir_all(dir.join("logs"))?;
        std::fs::create_dir_all(dir.join("images"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.node.max_gpu_usage, 90);
        assert_eq!(config.node.max_cpu_usage, 50);
        assert_eq!(config.node.idle_threshold_minutes, 5);
        assert!(config.node.pause_on_battery);
        assert!(config.node.pause_on_fullscreen);
        assert!(config.wallet.public_address.is_empty());
        assert_eq!(
            config.network.orchestrator_url,
            "https://api.computenetwork.sh"
        );
        assert_eq!(config.network.region, "auto");
        assert_eq!(config.logging.level, "info");
    }

    #[test]
    fn test_config_roundtrip() {
        let config = Config::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: Config = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.node.max_gpu_usage, config.node.max_gpu_usage);
        assert_eq!(parsed.wallet.public_address, config.wallet.public_address);
        assert_eq!(parsed.network.orchestrator_url, config.network.orchestrator_url);
    }

    #[test]
    fn test_config_get_set() {
        let mut config = Config::default();

        // Get
        assert_eq!(config.get("node.name").unwrap(), config.node.name);
        assert_eq!(config.get("node.max_gpu_usage").unwrap(), "90");
        assert_eq!(config.get("wallet.public_address").unwrap(), "");
        assert!(config.get("nonexistent.key").is_none());

        // Set
        config.set("node.name", "test-node").unwrap();
        assert_eq!(config.node.name, "test-node");

        config.set("node.max_gpu_usage", "75").unwrap();
        assert_eq!(config.node.max_gpu_usage, 75);

        config.set("wallet.public_address", "So1anaAddress123").unwrap();
        assert_eq!(config.wallet.public_address, "So1anaAddress123");

        // Invalid key
        assert!(config.set("invalid.key", "value").is_err());
    }

    #[test]
    fn test_config_save_and_load() {
        let tmpdir = tempfile::tempdir().unwrap();
        let config_path = tmpdir.path().join("config.toml");

        let mut config = Config::default();
        config.node.name = "test-node".into();
        config.wallet.public_address = "TestAddress123".into();

        // Save
        let toml_str = toml::to_string_pretty(&config).unwrap();
        std::fs::write(&config_path, &toml_str).unwrap();

        // Load
        let contents = std::fs::read_to_string(&config_path).unwrap();
        let loaded: Config = toml::from_str(&contents).unwrap();

        assert_eq!(loaded.node.name, "test-node");
        assert_eq!(loaded.wallet.public_address, "TestAddress123");
    }

    #[test]
    fn test_config_dir_exists() {
        assert!(config_dir().is_some());
    }

    #[test]
    fn test_config_file_path() {
        let path = config_file_path().unwrap();
        assert!(path.to_string_lossy().contains("config.toml"));
    }
}
