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
    #[serde(default)]
    pub appearance: AppearanceConfig,
    #[serde(default)]
    pub models: ModelsConfig,
    #[serde(default)]
    pub experimental: ExperimentalConfig,
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
    #[serde(default)]
    pub node_token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub orchestrator_url: String,
    pub region: String,
    #[serde(default)]
    pub advertise_host: String,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppearanceConfig {
    #[serde(default = "default_theme")]
    pub theme: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    #[serde(default = "default_models_cache_dir")]
    pub cache_dir: String,
    #[serde(default = "default_active_model")]
    pub active_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentalConfig {
    #[serde(default)]
    pub stage_mode_enabled: bool,
    #[serde(default = "default_stage_backend")]
    pub stage_backend: String,
    #[serde(default)]
    pub stage_gateway_addr: String,
    #[serde(default)]
    pub stage_gateway_autostart: bool,
    #[serde(default)]
    pub stage_gateway_model_path: String,
    #[serde(default)]
    pub stage_gateway_stage_node_bin: String,
    #[serde(default)]
    pub stage_gateway_gateway_bin: String,
    #[serde(default = "default_stage_gateway_connect_timeout_ms")]
    pub stage_gateway_connect_timeout_ms: u64,
    #[serde(default = "default_stage_gateway_retry_window_ms")]
    pub stage_gateway_retry_window_ms: u64,
    #[serde(default = "default_stage_gateway_retry_interval_ms")]
    pub stage_gateway_retry_interval_ms: u64,
    #[serde(default = "default_stage_gateway_startup_grace_ms")]
    pub stage_gateway_startup_grace_ms: u64,
    #[serde(default = "default_stage_acceleration")]
    pub stage_acceleration: String,
    #[serde(default = "default_stage_acceleration_provider")]
    pub stage_acceleration_provider: String,
}

fn default_models_cache_dir() -> String {
    config_dir()
        .or_else(|| {
            // Fallback: expand ~ properly instead of using literal tilde
            std::env::var("HOME")
                .ok()
                .or_else(|| std::env::var("USERPROFILE").ok())
                .map(|h| PathBuf::from(h).join(".compute"))
        })
        .unwrap_or_else(|| PathBuf::from("/tmp/.compute"))
        .join("models")
        .to_string_lossy()
        .into_owned()
}

fn default_active_model() -> String {
    "auto".into()
}

fn default_theme() -> String {
    "system".into()
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self { cache_dir: default_models_cache_dir(), active_model: default_active_model() }
    }
}

impl Default for AppearanceConfig {
    fn default() -> Self {
        Self { theme: default_theme() }
    }
}

impl Default for ExperimentalConfig {
    fn default() -> Self {
        Self {
            stage_mode_enabled: true,
            stage_backend: default_stage_backend(),
            stage_gateway_addr: String::new(),
            stage_gateway_autostart: true,
            stage_gateway_model_path: String::new(),
            stage_gateway_stage_node_bin: String::new(),
            stage_gateway_gateway_bin: String::new(),
            stage_gateway_connect_timeout_ms: default_stage_gateway_connect_timeout_ms(),
            stage_gateway_retry_window_ms: default_stage_gateway_retry_window_ms(),
            stage_gateway_retry_interval_ms: default_stage_gateway_retry_interval_ms(),
            stage_gateway_startup_grace_ms: default_stage_gateway_startup_grace_ms(),
            stage_acceleration: default_stage_acceleration(),
            stage_acceleration_provider: default_stage_acceleration_provider(),
        }
    }
}

fn default_stage_backend() -> String {
    "llama-stage-gateway".into()
}

fn default_stage_gateway_connect_timeout_ms() -> u64 {
    2_000
}

fn default_stage_gateway_retry_window_ms() -> u64 {
    30_000
}

fn default_stage_gateway_retry_interval_ms() -> u64 {
    250
}

fn default_stage_gateway_startup_grace_ms() -> u64 {
    0
}

fn default_stage_acceleration() -> String {
    "auto".into()
}

fn default_stage_acceleration_provider() -> String {
    "auto".into()
}

impl Default for Config {
    fn default() -> Self {
        let compute_dir = config_dir()
            .or_else(|| {
                std::env::var("HOME")
                    .ok()
                    .or_else(|| std::env::var("USERPROFILE").ok())
                    .map(|h| PathBuf::from(h).join(".compute"))
            })
            .unwrap_or_else(|| PathBuf::from("/tmp/.compute"));
        Self {
            node: NodeConfig {
                name: hostname(),
                max_gpu_usage: 90,
                max_cpu_usage: 50,
                idle_threshold_minutes: 5,
                pause_on_battery: true,
                pause_on_fullscreen: true,
            },
            wallet: WalletConfig {
                public_address: String::new(),
                node_id: String::new(),
                node_token: String::new(),
            },
            network: NetworkConfig {
                orchestrator_url: "https://api.computenetwork.sh".into(),
                region: "auto".into(),
                advertise_host: String::new(),
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
            appearance: AppearanceConfig::default(),
            models: ModelsConfig {
                cache_dir: compute_dir.join("models").to_string_lossy().into_owned(),
                active_model: "auto".into(),
            },
            experimental: ExperimentalConfig::default(),
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

    /// Save config atomically: write to .tmp then rename.
    /// Prevents corruption if the process crashes mid-write.
    pub fn save(&self) -> Result<()> {
        let path = config_file_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create config directory: {}", parent.display())
            })?;
        }
        let contents =
            toml::to_string_pretty(self).with_context(|| "Failed to serialize config")?;

        // Atomic write: write to temp file, then rename (rename is atomic on all platforms)
        let tmp_path = path.with_extension("toml.tmp");
        std::fs::write(&tmp_path, &contents)
            .with_context(|| format!("Failed to write temp config: {}", tmp_path.display()))?;
        std::fs::rename(&tmp_path, &path).with_context(|| {
            format!("Failed to rename config: {} -> {}", tmp_path.display(), path.display())
        })?;
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
            "wallet.node_token" => Some(self.wallet.node_token.clone()),
            "network.orchestrator_url" => Some(self.network.orchestrator_url.clone()),
            "network.region" => Some(self.network.region.clone()),
            "network.advertise_host" => Some(self.network.advertise_host.clone()),
            "logging.level" => Some(self.logging.level.clone()),
            "appearance.theme" => Some(self.appearance.theme.clone()),
            "models.active_model" => Some(self.models.active_model.clone()),
            "experimental.stage_mode_enabled" => {
                Some(self.experimental.stage_mode_enabled.to_string())
            }
            "experimental.stage_backend" => Some(self.experimental.stage_backend.clone()),
            "experimental.stage_gateway_addr" => Some(self.experimental.stage_gateway_addr.clone()),
            "experimental.stage_gateway_autostart" => {
                Some(self.experimental.stage_gateway_autostart.to_string())
            }
            "experimental.stage_gateway_model_path" => {
                Some(self.experimental.stage_gateway_model_path.clone())
            }
            "experimental.stage_gateway_stage_node_bin" => {
                Some(self.experimental.stage_gateway_stage_node_bin.clone())
            }
            "experimental.stage_gateway_gateway_bin" => {
                Some(self.experimental.stage_gateway_gateway_bin.clone())
            }
            "experimental.stage_gateway_connect_timeout_ms" => {
                Some(self.experimental.stage_gateway_connect_timeout_ms.to_string())
            }
            "experimental.stage_gateway_retry_window_ms" => {
                Some(self.experimental.stage_gateway_retry_window_ms.to_string())
            }
            "experimental.stage_gateway_retry_interval_ms" => {
                Some(self.experimental.stage_gateway_retry_interval_ms.to_string())
            }
            "experimental.stage_gateway_startup_grace_ms" => {
                Some(self.experimental.stage_gateway_startup_grace_ms.to_string())
            }
            "experimental.stage_acceleration" => Some(self.experimental.stage_acceleration.clone()),
            "experimental.stage_acceleration_provider" => {
                Some(self.experimental.stage_acceleration_provider.clone())
            }
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
            "wallet.node_token" => self.wallet.node_token = value.to_string(),
            "network.orchestrator_url" => self.network.orchestrator_url = value.to_string(),
            "network.region" => self.network.region = value.to_string(),
            "network.advertise_host" => self.network.advertise_host = value.to_string(),
            "logging.level" => self.logging.level = value.to_string(),
            "appearance.theme" => self.appearance.theme = value.to_string(),
            "models.active_model" => self.models.active_model = value.to_string(),
            "experimental.stage_mode_enabled" => {
                self.experimental.stage_mode_enabled = value.parse()?
            }
            "experimental.stage_backend" => self.experimental.stage_backend = value.to_string(),
            "experimental.stage_gateway_addr" => {
                self.experimental.stage_gateway_addr = value.to_string()
            }
            "experimental.stage_gateway_autostart" => {
                self.experimental.stage_gateway_autostart = value.parse()?
            }
            "experimental.stage_gateway_model_path" => {
                self.experimental.stage_gateway_model_path = value.to_string()
            }
            "experimental.stage_gateway_stage_node_bin" => {
                self.experimental.stage_gateway_stage_node_bin = value.to_string()
            }
            "experimental.stage_gateway_gateway_bin" => {
                self.experimental.stage_gateway_gateway_bin = value.to_string()
            }
            "experimental.stage_gateway_connect_timeout_ms" => {
                self.experimental.stage_gateway_connect_timeout_ms = value.parse()?
            }
            "experimental.stage_gateway_retry_window_ms" => {
                self.experimental.stage_gateway_retry_window_ms = value.parse()?
            }
            "experimental.stage_gateway_retry_interval_ms" => {
                self.experimental.stage_gateway_retry_interval_ms = value.parse()?
            }
            "experimental.stage_gateway_startup_grace_ms" => {
                self.experimental.stage_gateway_startup_grace_ms = value.parse()?
            }
            "experimental.stage_acceleration" => {
                self.experimental.stage_acceleration = value.to_string()
            }
            "experimental.stage_acceleration_provider" => {
                self.experimental.stage_acceleration_provider = value.to_string()
            }
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
        std::fs::create_dir_all(dir.join("models"))?;
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
        assert_eq!(config.network.orchestrator_url, "https://api.computenetwork.sh");
        assert_eq!(config.network.region, "auto");
        assert_eq!(config.logging.level, "info");
        assert_eq!(config.experimental.stage_acceleration, "auto");
        assert_eq!(config.experimental.stage_acceleration_provider, "auto");
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
        assert_eq!(config.get("experimental.stage_acceleration").unwrap(), "auto");
        assert_eq!(config.get("experimental.stage_acceleration_provider").unwrap(), "auto");
        assert_eq!(config.get("experimental.stage_gateway_autostart").unwrap(), "false");
        assert_eq!(config.get("experimental.stage_gateway_stage_node_bin").unwrap(), "");
        assert_eq!(config.get("experimental.stage_gateway_connect_timeout_ms").unwrap(), "2000");
        assert_eq!(config.get("experimental.stage_gateway_retry_window_ms").unwrap(), "30000");
        assert_eq!(config.get("experimental.stage_gateway_retry_interval_ms").unwrap(), "250");
        assert_eq!(config.get("experimental.stage_gateway_startup_grace_ms").unwrap(), "0");
        assert!(config.get("nonexistent.key").is_none());

        // Set
        config.set("node.name", "test-node").unwrap();
        assert_eq!(config.node.name, "test-node");

        config.set("node.max_gpu_usage", "75").unwrap();
        assert_eq!(config.node.max_gpu_usage, 75);

        config.set("wallet.public_address", "So1anaAddress123").unwrap();
        assert_eq!(config.wallet.public_address, "So1anaAddress123");

        config.set("experimental.stage_acceleration", "metal").unwrap();
        assert_eq!(config.experimental.stage_acceleration, "metal");

        config.set("experimental.stage_acceleration_provider", "ggml").unwrap();
        assert_eq!(config.experimental.stage_acceleration_provider, "ggml");

        config.set("experimental.stage_gateway_autostart", "true").unwrap();
        assert!(config.experimental.stage_gateway_autostart);

        config.set("experimental.stage_gateway_model_path", "/tmp/model.gguf").unwrap();
        assert_eq!(config.experimental.stage_gateway_model_path, "/tmp/model.gguf");

        config.set("experimental.stage_gateway_stage_node_bin", "/tmp/stage-node").unwrap();
        assert_eq!(config.experimental.stage_gateway_stage_node_bin, "/tmp/stage-node");

        config.set("experimental.stage_gateway_gateway_bin", "/tmp/gateway-node").unwrap();
        assert_eq!(config.experimental.stage_gateway_gateway_bin, "/tmp/gateway-node");

        config.set("experimental.stage_gateway_connect_timeout_ms", "1500").unwrap();
        assert_eq!(config.experimental.stage_gateway_connect_timeout_ms, 1500);

        config.set("experimental.stage_gateway_retry_window_ms", "45000").unwrap();
        assert_eq!(config.experimental.stage_gateway_retry_window_ms, 45000);

        config.set("experimental.stage_gateway_retry_interval_ms", "500").unwrap();
        assert_eq!(config.experimental.stage_gateway_retry_interval_ms, 500);

        config.set("experimental.stage_gateway_startup_grace_ms", "4000").unwrap();
        assert_eq!(config.experimental.stage_gateway_startup_grace_ms, 4000);

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
