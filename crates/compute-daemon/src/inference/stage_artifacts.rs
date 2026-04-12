use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tracing::{info, warn, debug};

pub fn stages_cache_dir() -> PathBuf {
    crate::config::config_dir()
        .unwrap_or_else(|| PathBuf::from("~/.compute"))
        .join("stages")
}

pub fn packed_stage_dir(model_id: &str, start_layer: u32, end_layer: u32) -> PathBuf {
    stages_cache_dir()
        .join(model_id)
        .join(format!("packed-stage-{}-{}", start_layer, end_layer))
}

pub fn is_stage_cached(model_id: &str, start_layer: u32, end_layer: u32) -> bool {
    let dir = packed_stage_dir(model_id, start_layer, end_layer);
    if !dir.is_dir() {
        return false;
    }
    has_index_file(&dir)
}

fn has_index_file(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries.flatten().any(|e| {
                e.file_name().to_string_lossy().ends_with(".index.json")
            })
        })
        .unwrap_or(false)
}

pub async fn ensure_stage_artifacts(
    model_id: &str,
    start_layer: u32,
    end_layer: u32,
    artifact_url: &str,
    expected_sha256: Option<&str>,
    expected_size: Option<u64>,
) -> Result<PathBuf> {
    let dest_dir = packed_stage_dir(model_id, start_layer, end_layer);

    if is_stage_cached(model_id, start_layer, end_layer) {
        info!(
            "Stage artifacts for {} layers {}-{} already cached at {}",
            model_id, start_layer, end_layer, dest_dir.display()
        );
        return Ok(dest_dir);
    }

    info!(
        "Downloading stage artifacts for {} layers {}-{} from {}",
        model_id, start_layer, end_layer, artifact_url
    );

    std::fs::create_dir_all(&dest_dir)
        .with_context(|| format!("Failed to create stage dir {}", dest_dir.display()))?;

    let archive_path = dest_dir.join("stage-archive.tar");
    download_file(artifact_url, &archive_path).await?;

    if let Some(sha) = expected_sha256 {
        if !sha.is_empty() {
            info!("Verifying artifact checksum...");
            let actual = sha256_file(&archive_path)?;
            if actual != sha {
                let _ = std::fs::remove_file(&archive_path);
                anyhow::bail!(
                    "Artifact checksum mismatch: expected {}, got {}",
                    sha, actual
                );
            }
            debug!("Checksum OK");
        }
    }

    if let Some(expected) = expected_size {
        if expected > 0 {
            let actual = std::fs::metadata(&archive_path)
                .map(|m| m.len())
                .unwrap_or(0);
            if actual != expected {
                warn!(
                    "Artifact size mismatch: expected {} bytes, got {}",
                    expected, actual
                );
            }
        }
    }

    if archive_path.exists() && is_tar_file(&archive_path) {
        info!("Extracting stage archive...");
        extract_tar(&archive_path, &dest_dir)?;
        let _ = std::fs::remove_file(&archive_path);
    }

    if !has_index_file(&dest_dir) {
        anyhow::bail!(
            "Downloaded artifact does not contain an .index.json file in {}",
            dest_dir.display()
        );
    }

    info!(
        "Stage artifacts ready at {}",
        dest_dir.display()
    );
    Ok(dest_dir)
}

fn is_tar_file(path: &Path) -> bool {
    std::fs::File::open(path)
        .and_then(|mut f| {
            use std::io::Read;
            let mut magic = [0u8; 4];
            f.read_exact(&mut magic)?;
            Ok(magic)
        })
        .map(|_| {
            path.extension()
                .map(|e| e == "tar" || e == "gz" || e == "tgz")
                .unwrap_or(false)
        })
        .unwrap_or(false)
}

fn extract_tar(archive: &Path, dest: &Path) -> Result<()> {
    let status = std::process::Command::new("tar")
        .args(["xf", &archive.to_string_lossy(), "-C", &dest.to_string_lossy()])
        .status()
        .context("Failed to run tar")?;
    if !status.success() {
        anyhow::bail!("tar extraction failed with status {}", status);
    }
    Ok(())
}

async fn download_file(url: &str, dest: &Path) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(7200))
        .build()?;

    let resp = client.get(url).send().await.context("Download request failed")?;

    if !resp.status().is_success() {
        anyhow::bail!("Download failed: HTTP {}", resp.status());
    }

    let total_size = resp.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;

    let tmp_path = dest.with_extension("tmp");
    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .context("Failed to create temp file")?;

    use tokio::io::AsyncWriteExt;
    use futures_util::StreamExt;
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Error reading download stream")?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;

        if total_size > 0 && downloaded % (100 * 1024 * 1024) < chunk.len() as u64 {
            let pct = (downloaded as f64 / total_size as f64) * 100.0;
            info!(
                "Stage download: {:.0}% ({:.1} GB / {:.1} GB)",
                pct,
                downloaded as f64 / 1_073_741_824.0,
                total_size as f64 / 1_073_741_824.0
            );
        }
    }

    file.flush().await?;
    drop(file);

    tokio::fs::rename(&tmp_path, dest)
        .await
        .context("Failed to move downloaded artifact")?;

    info!(
        "Downloaded {:.2} GB to {}",
        downloaded as f64 / 1_073_741_824.0,
        dest.display()
    );
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let output = std::process::Command::new("shasum")
        .args(["-a", "256", &path.to_string_lossy()])
        .output()
        .context("shasum not found")?;
    if !output.status.success() {
        anyhow::bail!("shasum failed");
    }
    let hash = String::from_utf8_lossy(&output.stdout);
    hash.split_whitespace()
        .next()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("Failed to parse shasum output"))
}
