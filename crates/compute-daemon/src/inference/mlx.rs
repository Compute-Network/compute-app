//! oMLX backend manager (v0.4.2).
//!
//! Wraps [`omlx serve`](https://github.com/jundot/omlx) as a child process
//! so the daemon can route chat completions for MLX-format models (e.g.
//! `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit`) through Apple's MLX framework on
//! Apple Silicon. Parallel to `InferenceManager` which owns llama-server;
//! both can run simultaneously per the v0.4.1 keep-warm pattern.
//!
//! oMLX is Python + FastAPI under the hood. Install via
//! `brew install jundot/omlx/omlx` on macOS (`install.sh` handles this
//! automatically in v0.4.2+). Non-Mac hosts never instantiate this manager.
//!
//! The server discovers LLMs from subdirectories under `--model-dir`, so
//! we point it at `~/.compute/models/mlx/` and each MLX snapshot lives in
//! `~/.compute/models/mlx/<folder>/`. Request-side routing matches the
//! requested model id to the `MlxVariant::folder` from the catalog.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Status of the oMLX child process.
#[derive(Debug, Clone, PartialEq)]
pub enum OmlxStatus {
    Idle,
    Starting,
    Running { port: u16, model_dir: PathBuf },
    Error(String),
}

impl std::fmt::Display for OmlxStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OmlxStatus::Idle => write!(f, "idle"),
            OmlxStatus::Starting => write!(f, "starting"),
            OmlxStatus::Running { port, .. } => write!(f, "running :{port}"),
            OmlxStatus::Error(e) => write!(f, "error: {e}"),
        }
    }
}

pub struct OmlxManager {
    status: OmlxStatus,
    child: Option<Child>,
    port: u16,
    model_dir: PathBuf,
    bin_path: Option<PathBuf>,
}

impl OmlxManager {
    /// Default port. Chosen to not clash with llama-server (8090); picked
    /// high enough to avoid common reserved ranges but not too high that
    /// firewalls mangle it.
    pub const DEFAULT_PORT: u16 = 8091;

    /// Whether the current build target is eligible to run oMLX. MLX is
    /// Apple Silicon only — Linux / Windows / Intel Macs always return
    /// false and never touch the binary.
    #[inline]
    pub fn platform_supported() -> bool {
        cfg!(all(target_os = "macos", target_arch = "aarch64"))
    }

    /// Probe the system for a usable `omlx` binary. Returns `None` when not
    /// found, in which case the daemon falls back to llama-server for MLX
    /// model requests. A quiet no-op on non-Mac platforms.
    pub fn detect_binary() -> Option<PathBuf> {
        if !Self::platform_supported() {
            return None;
        }
        let candidates = [
            // Homebrew on Apple Silicon (`brew install jundot/omlx/omlx`).
            PathBuf::from("/opt/homebrew/bin/omlx"),
            // Homebrew on Intel-arch (should never match on aarch64 but harmless).
            PathBuf::from("/usr/local/bin/omlx"),
            // pip install --user on macOS 14+ Python (e.g. 3.11).
            PathBuf::from(
                std::env::var("HOME").unwrap_or_default() + "/Library/Python/3.11/bin/omlx",
            ),
            PathBuf::from(
                std::env::var("HOME").unwrap_or_default() + "/Library/Python/3.12/bin/omlx",
            ),
        ];
        for c in candidates.iter() {
            if c.exists() {
                return Some(c.clone());
            }
        }
        // PATH fallback — `which omlx` equivalent.
        if let Ok(path) = std::env::var("PATH") {
            for dir in path.split(':') {
                let candidate = PathBuf::from(dir).join("omlx");
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }
        None
    }

    /// Construct a manager pointed at the MLX cache directory
    /// (`<models.cache_dir>/mlx`). Does not spawn — call `start()`.
    pub fn new(model_dir: PathBuf) -> Self {
        Self {
            status: OmlxStatus::Idle,
            child: None,
            port: Self::DEFAULT_PORT,
            model_dir,
            bin_path: Self::detect_binary(),
        }
    }

    pub fn status(&self) -> &OmlxStatus {
        &self.status
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn is_installed(&self) -> bool {
        self.bin_path.is_some()
    }

    /// True if the child is alive AND the HTTP probe succeeded at some
    /// point. Doesn't re-probe on every call; callers that need liveness
    /// guarantees should call `check_alive()` first.
    pub fn is_running(&self) -> bool {
        matches!(self.status, OmlxStatus::Running { .. })
    }

    /// Spawn `omlx serve --model-dir <dir> --port <port>`. No-op if already
    /// running. Returns early with an error if the binary isn't installed;
    /// the daemon is expected to fall back to llama-server in that case.
    pub fn start(&mut self) -> anyhow::Result<()> {
        if self.is_running() {
            return Ok(());
        }
        let Some(bin) = self.bin_path.clone() else {
            anyhow::bail!(
                "omlx binary not found — install via `brew install jundot/omlx/omlx` on macOS (install.sh does this for you in v0.4.2+)"
            );
        };

        if !self.model_dir.exists() {
            std::fs::create_dir_all(&self.model_dir).map_err(|e| {
                anyhow::anyhow!("creating omlx model dir {}: {e}", self.model_dir.display())
            })?;
        }

        info!(
            "[omlx] Starting {} serve --model-dir {} --port {}",
            bin.display(),
            self.model_dir.display(),
            self.port
        );

        self.status = OmlxStatus::Starting;
        let child = Command::new(&bin)
            .arg("serve")
            .arg("--model-dir")
            .arg(&self.model_dir)
            .arg("--port")
            .arg(self.port.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| {
                self.status = OmlxStatus::Error(format!("spawn: {e}"));
                anyhow::anyhow!("spawning omlx: {e}")
            })?;

        self.child = Some(child);
        self.status = OmlxStatus::Running { port: self.port, model_dir: self.model_dir.clone() };
        Ok(())
    }

    /// SIGTERM → wait up to 5s → SIGKILL. Safe to call when idle.
    pub fn stop(&mut self) {
        let Some(mut child) = self.child.take() else {
            self.status = OmlxStatus::Idle;
            return;
        };
        info!("[omlx] stopping (pid={})", child.id());
        let _ = child.kill();
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            match child.try_wait() {
                Ok(Some(_)) => break,
                _ => std::thread::sleep(Duration::from_millis(100)),
            }
        }
        let _ = child.try_wait();
        self.status = OmlxStatus::Idle;
    }

    /// Quick reap: returns `true` if the child is still alive. If the child
    /// has exited, transitions status to `Error(...)` so callers know to
    /// fall back to llama-server.
    pub fn check_alive(&mut self) -> bool {
        let Some(child) = self.child.as_mut() else {
            return false;
        };
        match child.try_wait() {
            Ok(None) => true,
            Ok(Some(status)) => {
                warn!("[omlx] child exited (status={status})");
                self.child = None;
                self.status = OmlxStatus::Error(format!("exited: {status}"));
                false
            }
            Err(e) => {
                warn!("[omlx] try_wait failed: {e}");
                self.status = OmlxStatus::Error(format!("try_wait: {e}"));
                false
            }
        }
    }

    /// Async readiness probe. Polls `http://127.0.0.1:<port>/v1/models`
    /// every 500 ms up to `timeout`; returns `true` once it gets a 200.
    /// Typical first-start latency is a few seconds (Python + FastAPI
    /// bootstrap).
    pub async fn wait_until_ready(&self, timeout: Duration) -> bool {
        let url = format!("http://127.0.0.1:{}/v1/models", self.port);
        let client = match reqwest::Client::builder().timeout(Duration::from_millis(500)).build() {
            Ok(c) => c,
            Err(_) => return false,
        };
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Ok(resp) = client.get(&url).send().await {
                if resp.status().is_success() {
                    return true;
                }
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        false
    }

    /// URL for proxying chat completions. Callers bolt their request JSON
    /// onto this exactly as they would for llama-server.
    pub fn completions_url(&self) -> String {
        format!("http://127.0.0.1:{}/v1/chat/completions", self.port)
    }
}

impl Drop for OmlxManager {
    fn drop(&mut self) {
        self.stop();
    }
}
