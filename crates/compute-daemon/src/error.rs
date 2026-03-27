use thiserror::Error;

/// Errors that can occur in the compute daemon.
#[derive(Debug, Error)]
pub enum ComputeError {
    // Config errors
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Config file not found. Run `compute init` first.")]
    ConfigNotFound,

    // Daemon errors
    #[error("Daemon is already running (PID: {pid})")]
    DaemonAlreadyRunning { pid: u32 },

    #[error("Daemon is not running")]
    DaemonNotRunning,

    #[error("Stale PID file (process {pid} not found)")]
    StalePidFile { pid: u32 },

    // Hardware errors
    #[error("No GPU detected. CPU-only mode available.")]
    NoGpuDetected,

    #[error("Insufficient VRAM: {available_mb}MB available, {required_mb}MB required")]
    InsufficientVram { available_mb: u64, required_mb: u64 },

    #[error("Docker not available. Install Docker: https://docker.com/get-started")]
    DockerNotAvailable,

    // Network errors
    #[error("Cannot reach orchestrator at {url}: {reason}")]
    OrchestratorUnreachable { url: String, reason: String },

    #[error("Node not registered. Run `compute start` first.")]
    NotRegistered,

    #[error("Pipeline error: {0}")]
    Pipeline(String),

    // Wallet errors
    #[error("Invalid Solana address format: {0}")]
    InvalidWalletAddress(String),

    #[error("No wallet address configured. Set with `compute wallet set <address>`")]
    NoWalletConfigured,

    // Container errors
    #[error("Container error: {0}")]
    Container(String),

    #[error("Container image not found: {0}")]
    ImageNotFound(String),

    // IO/System errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

impl From<&str> for ComputeError {
    fn from(s: &str) -> Self {
        ComputeError::Other(s.to_string())
    }
}

impl From<String> for ComputeError {
    fn from(s: String) -> Self {
        ComputeError::Other(s)
    }
}
