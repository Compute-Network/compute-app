use clap::{Parser, Subcommand};

const VERSION_INFO: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("COMPUTE_GIT_HASH"),
    " ",
    env!("COMPUTE_BUILD_DATE"),
    ")"
);

#[derive(Parser)]
#[command(
    name = "compute",
    about = "Compute — Decentralized GPU Infrastructure",
    long_about = "Compute runs local and distributed inference across contributed machines. The stable live path today is single-node serving through the Compute orchestrator, with distributed pipeline work still present in the repo for future expansion.",
    version = VERSION_INFO,
    author = "Compute Network <dev@computenetwork.sh>"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the compute daemon in the foreground
    Start,

    /// Stop the daemon gracefully
    Stop,

    /// Quick one-line status (running/stopped, earnings, GPU)
    Status,

    /// Open the full TUI dashboard (live stats, animated)
    Dashboard,

    /// Tail daemon logs
    Logs {
        /// Number of lines to show
        #[arg(short, long, default_value = "50")]
        lines: usize,

        /// Follow log output
        #[arg(short, long)]
        follow: bool,
    },

    /// First-time setup wizard
    Init,

    /// Configuration management
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Show wallet address and balances
    Wallet {
        #[command(subcommand)]
        action: Option<WalletAction>,
    },

    /// Show earnings summary
    Earnings,

    /// Run hardware benchmark, optionally including llama-server sweep
    Benchmark {
        /// Run llama-server performance sweep
        #[arg(long)]
        llama: bool,

        /// Model id to benchmark (default: gemma-4-e4b-q4)
        #[arg(long)]
        model: Option<String>,

        /// Compute API key for orchestrator comparison (falls back to OPENAI_API_KEY)
        #[arg(long)]
        api_key: Option<String>,
    },

    /// Show detected hardware info
    Hardware,

    /// Show current pipeline status and peers
    Pipeline,

    /// List online nodes in the network
    Nodes {
        /// Show all nodes (default: only online)
        #[arg(short, long)]
        all: bool,

        /// Limit number of results
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },

    /// Self-update to latest version
    Update,

    /// Clean uninstall
    Uninstall,

    /// Diagnose common issues
    Doctor,

    /// Install/uninstall as a system service (auto-start on login)
    Service {
        #[command(subcommand)]
        action: ServiceAction,
    },
}

#[derive(Subcommand)]
pub enum ServiceAction {
    /// Install as a system service (launchd on macOS, systemd on Linux)
    Install,
    /// Uninstall the system service
    Uninstall,
    /// Check if the service is installed
    Status,
}

#[derive(Subcommand)]
pub enum ConfigAction {
    /// Set a config value
    Set {
        /// Config key (e.g., node.name)
        key: String,
        /// Config value
        value: String,
    },
    /// Get a config value
    Get {
        /// Config key (e.g., node.name)
        key: String,
    },
    /// Show all config
    Show,
}

#[derive(Subcommand)]
pub enum WalletAction {
    /// Authenticate this node via wallet in the browser
    Login,

    /// Deprecated: wallet-only auth now uses browser login
    Set {
        /// Solana public address
        address: String,
    },
}
