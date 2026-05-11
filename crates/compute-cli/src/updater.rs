use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const REPO: &str = "Compute-Network/compute-app";

#[derive(Debug, Clone)]
pub struct LatestRelease {
    pub tag: String,
    pub version: String,
    pub download_url: String,
    pub html_url: String,
}

#[derive(Debug, Clone)]
pub enum UpdateCheck {
    UpToDate { current: String },
    Available(LatestRelease),
    NoRelease,
    NoCompatibleAsset { tag: String, html_url: String },
}

#[derive(Debug, Clone)]
pub enum AutoUpdateOutcome {
    UpToDate { current: String },
    Updated { version: String },
    Skipped { version: String, reason: String },
    NoRelease,
    NoCompatibleAsset { tag: String, html_url: String },
}

pub async fn check_for_update() -> Result<UpdateCheck> {
    let client = github_client(Duration::from_secs(10))?;
    let resp =
        client.get(format!("https://api.github.com/repos/{REPO}/releases/latest")).send().await?;

    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(UpdateCheck::NoRelease);
    }

    if !resp.status().is_success() {
        bail!("GitHub API error: {}", resp.status());
    }

    let release: serde_json::Value = resp.json().await?;
    let tag = release["tag_name"].as_str().context("No tag_name in release")?.to_string();
    let version = tag.trim_start_matches('v').to_string();
    let html_url = release["html_url"]
        .as_str()
        .unwrap_or("https://github.com/Compute-Network/compute-app/releases")
        .to_string();

    if version == VERSION {
        return Ok(UpdateCheck::UpToDate { current: VERSION.to_string() });
    }

    let target = current_target();
    let asset_name = format!("compute-{target}.tar.gz");
    let assets = release["assets"].as_array().context("No assets in release")?;

    for asset in assets {
        let name = asset["name"].as_str().unwrap_or("");
        if name == asset_name || name.contains(&target) && name.ends_with(".tar.gz") {
            let download_url = asset["browser_download_url"]
                .as_str()
                .context("release asset missing browser_download_url")?
                .to_string();
            return Ok(UpdateCheck::Available(LatestRelease {
                tag,
                version,
                download_url,
                html_url,
            }));
        }
    }

    Ok(UpdateCheck::NoCompatibleAsset { tag, html_url })
}

pub fn auto_update_blocking() -> Result<AutoUpdateOutcome> {
    if auto_update_disabled() {
        return Ok(AutoUpdateOutcome::Skipped {
            version: VERSION.to_string(),
            reason: "disabled by COMPUTE_AUTO_UPDATE".to_string(),
        });
    }

    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    rt.block_on(async {
        match check_for_update().await? {
            UpdateCheck::UpToDate { current } => Ok(AutoUpdateOutcome::UpToDate { current }),
            UpdateCheck::NoRelease => Ok(AutoUpdateOutcome::NoRelease),
            UpdateCheck::NoCompatibleAsset { tag, html_url } => {
                Ok(AutoUpdateOutcome::NoCompatibleAsset { tag, html_url })
            }
            UpdateCheck::Available(release) => {
                if let Some(reason) = auto_update_skip_reason()? {
                    return Ok(AutoUpdateOutcome::Skipped { version: release.version, reason });
                }
                install_release_bundle(&release).await?;
                Ok(AutoUpdateOutcome::Updated { version: release.version })
            }
        }
    })
}

pub async fn install_release_bundle(release: &LatestRelease) -> Result<()> {
    let install_dir = install_dir()?;
    std::fs::create_dir_all(&install_dir)
        .with_context(|| format!("create install dir {}", install_dir.display()))?;

    stop_existing_daemon_and_sidecars();

    let temp_dir = make_temp_dir()?;
    let archive = temp_dir.join("compute-update.tar.gz");

    let client = github_client(Duration::from_secs(120))?;
    let resp = client.get(&release.download_url).send().await?;
    if !resp.status().is_success() {
        bail!("Download failed: {}", resp.status());
    }
    let bytes = resp.bytes().await?;
    std::fs::write(&archive, &bytes)
        .with_context(|| format!("write update archive {}", archive.display()))?;

    extract_archive(&archive, &temp_dir)?;

    let compute_bin = temp_dir.join("compute");
    if !compute_bin.exists() {
        bail!("Release archive did not contain compute binary");
    }

    install_file_atomic(&compute_bin, &install_dir.join("compute"), true)?;

    for bin in ["llama_stage_tcp_node", "llama_stage_gateway_tcp_node"] {
        let src = temp_dir.join(bin);
        if src.exists() {
            install_file_atomic(&src, &install_dir.join(bin), true)?;
        }
    }

    remove_stale_shared_libraries(&install_dir);
    for entry in std::fs::read_dir(&temp_dir)? {
        let entry = entry?;
        let src = entry.path();
        if is_shared_library(&src) {
            let dest = install_dir.join(entry.file_name());
            install_file_atomic(&src, &dest, false)?;
        }
    }

    let _ = std::fs::remove_dir_all(&temp_dir);
    Ok(())
}

fn github_client(timeout: Duration) -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .timeout(timeout)
        .user_agent(format!("compute-cli/{VERSION}"))
        .build()?)
}

fn auto_update_disabled() -> bool {
    matches!(
        std::env::var("COMPUTE_AUTO_UPDATE").ok().as_deref(),
        Some("0") | Some("false") | Some("False") | Some("FALSE") | Some("off") | Some("OFF")
    )
}

fn auto_update_skip_reason() -> Result<Option<String>> {
    if matches!(std::env::var("COMPUTE_ALLOW_DEV_AUTO_UPDATE").ok().as_deref(), Some("1")) {
        return Ok(None);
    }

    let current_exe = std::env::current_exe()?;
    let expected_dir = install_dir()?;
    let current_dir = current_exe.parent().unwrap_or_else(|| Path::new(""));

    if paths_equal(current_dir, &expected_dir) {
        Ok(None)
    } else {
        Ok(Some(format!("current binary is not in {}", expected_dir.display())))
    }
}

fn install_dir() -> Result<PathBuf> {
    if let Ok(raw) = std::env::var("COMPUTE_INSTALL_DIR") {
        if !raw.trim().is_empty() {
            return Ok(PathBuf::from(raw));
        }
    }

    let home = dirs::home_dir().context("home directory not found")?;
    Ok(home.join(".compute").join("bin"))
}

fn make_temp_dir() -> Result<PathBuf> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();
    let dir = std::env::temp_dir().join(format!("compute-update-{}-{now}", std::process::id()));
    std::fs::create_dir_all(&dir).with_context(|| format!("create temp dir {}", dir.display()))?;
    Ok(dir)
}

fn extract_archive(archive: &Path, dest: &Path) -> Result<()> {
    let status = std::process::Command::new("tar")
        .arg("xzf")
        .arg(archive)
        .arg("-C")
        .arg(dest)
        .status()
        .context("run tar to extract update archive")?;
    if !status.success() {
        bail!("tar exited with {status}");
    }
    Ok(())
}

fn install_file_atomic(src: &Path, dest: &Path, executable: bool) -> Result<()> {
    let parent = dest.parent().context("destination has no parent")?;
    std::fs::create_dir_all(parent)?;

    let file_name = dest.file_name().and_then(|s| s.to_str()).unwrap_or("file");
    let tmp = parent.join(format!(".{file_name}.update-{}", std::process::id()));
    if tmp.exists() {
        let _ = std::fs::remove_file(&tmp);
    }

    std::fs::copy(src, &tmp)
        .with_context(|| format!("copy {} to {}", src.display(), tmp.display()))?;
    if executable {
        set_executable(&tmp)?;
    }
    std::fs::rename(&tmp, dest)
        .with_context(|| format!("replace {} with update", dest.display()))?;
    Ok(())
}

#[cfg(unix)]
fn set_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_executable(_path: &Path) -> Result<()> {
    Ok(())
}

fn remove_stale_shared_libraries(install_dir: &Path) {
    let Ok(entries) = std::fs::read_dir(install_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if is_shared_library(&path) {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn is_shared_library(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
        return false;
    };
    name.ends_with(".dylib") || name.ends_with(".so") || name.contains(".so.")
}

fn stop_existing_daemon_and_sidecars() {
    let _ = compute_daemon::daemon::stop_daemon();
    for proc in ["llama_stage_gateway_tcp_node", "llama_stage_tcp_node"] {
        let _ = std::process::Command::new("pkill").args(["-TERM", "-x", proc]).status();
    }
    std::thread::sleep(Duration::from_millis(500));
    for proc in ["llama_stage_gateway_tcp_node", "llama_stage_tcp_node"] {
        let _ = std::process::Command::new("pkill").args(["-KILL", "-x", proc]).status();
    }
}

fn paths_equal(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

fn current_target() -> String {
    let os = if cfg!(target_os = "macos") {
        "apple-darwin"
    } else if cfg!(target_os = "linux") {
        "unknown-linux-gnu"
    } else if cfg!(target_os = "windows") {
        "pc-windows"
    } else {
        "unknown"
    };

    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "unknown"
    };

    format!("{arch}-{os}")
}
