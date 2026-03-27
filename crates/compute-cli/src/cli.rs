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
    long_about = "Compute aggregates idle GPU/CPU resources and daisy-chains them via pipeline parallelism to run large AI models. Revenue flows back to contributors via the $COMPUTE token on Solana.",
    version = VERSION_INFO,
    author = "Compute Network <dev@computenetwork.sh>"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the compute daemon (shows splash, then detaches)
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

    /// Run GPU/CPU benchmark
    Benchmark,

    /// Show detected hardware info
    Hardware,

    /// Show current pipeline status and peers
    Pipeline,

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
    /// Set/update Solana public address
    Set {
        /// Solana public address
        address: String,
    },
}
