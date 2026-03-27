use anyhow::Result;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

use crate::config;

/// Initialize the tracing/logging system.
///
/// Logs go to both stderr (for interactive use) and a file (for daemon mode).
pub fn init(to_file: bool) -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    if to_file {
        let logs_dir = config::logs_dir()?;
        std::fs::create_dir_all(&logs_dir)?;

        let log_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(logs_dir.join("compute.log"))?;

        let file_layer = fmt::layer().with_writer(log_file).with_ansi(false).with_target(false);

        tracing_subscriber::registry().with(filter).with(file_layer).init();
    } else {
        tracing_subscriber::registry().with(filter).with(fmt::layer().with_target(false)).init();
    }

    Ok(())
}
