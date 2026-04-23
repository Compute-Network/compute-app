use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

pub fn stages_cache_dir() -> PathBuf {
    crate::config::config_dir().unwrap_or_else(|| PathBuf::from("~/.compute")).join("stages")
}

pub fn packed_stage_dir(model_id: &str, start_layer: u32, end_layer: u32) -> PathBuf {
    stages_cache_dir().join(model_id).join(format!("packed-stage-{}-{}", start_layer, end_layer))
}

/// Path to a per-stage GGUF shard cached by [`ensure_gguf_shard`].
pub fn gguf_shard_path(model_id: &str, role: &str, start_layer: u32, end_layer: u32) -> PathBuf {
    stages_cache_dir().join(model_id).join(format!("{}-{}-{}.gguf", role, start_layer, end_layer))
}

/// Download a per-stage GGUF shard whose layers are renumbered 0..(N-1).
/// Returns the local file path, reusing the cached file if it already exists
/// and passes the GGUF magic check.
pub async fn ensure_gguf_shard(
    model_id: &str,
    role: &str,
    start_layer: u32,
    end_layer: u32,
    shard_url: &str,
) -> Result<PathBuf> {
    let dest = gguf_shard_path(model_id, role, start_layer, end_layer);

    if dest.exists() && gguf_magic_ok(&dest).unwrap_or(false) {
        info!(
            "GGUF shard for {} {} layers {}-{} already cached at {}",
            model_id,
            role,
            start_layer,
            end_layer,
            dest.display()
        );
        return Ok(dest);
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create shard dir {}", parent.display()))?;
    }

    info!(
        "Downloading GGUF shard for {} {} layers {}-{} from {}",
        model_id, role, start_layer, end_layer, shard_url
    );
    download_file(shard_url, &dest).await?;

    if !gguf_magic_ok(&dest).unwrap_or(false) {
        let _ = std::fs::remove_file(&dest);
        anyhow::bail!("Downloaded shard is not a valid GGUF file: {}", dest.display());
    }

    info!("GGUF shard ready at {}", dest.display());
    Ok(dest)
}

/// Default root for MLX snapshots under a models cache dir — matches
/// `OmlxManager::model_dir()`, which is passed as `omlx serve --model-dir`.
pub fn mlx_cache_root(models_cache_dir: &Path) -> PathBuf {
    models_cache_dir.join("mlx")
}

/// Local folder for a specific MLX model under the cache root.
pub fn mlx_model_path(models_cache_dir: &Path, folder: &str) -> PathBuf {
    mlx_cache_root(models_cache_dir).join(folder)
}

/// Is an MLX model folder already snapshotted and usable by oMLX?
/// A usable snapshot needs `config.json` plus actual weights, not just the
/// metadata files Hugging Face writes first. For indexed checkpoints we read
/// `model.safetensors.index.json` and require every referenced shard to exist.
pub fn mlx_model_cached(models_cache_dir: &Path, folder: &str) -> bool {
    mlx_snapshot_complete(&mlx_model_path(models_cache_dir, folder))
}

pub fn mlx_snapshot_complete(path: &Path) -> bool {
    if !path.join("config.json").exists() {
        return false;
    }

    let index_path = path.join("model.safetensors.index.json");
    if index_path.exists() {
        let Ok(index_bytes) = std::fs::read(&index_path) else {
            return false;
        };
        let Ok(index_json) = serde_json::from_slice::<serde_json::Value>(&index_bytes) else {
            return false;
        };
        let Some(weight_map) = index_json.get("weight_map").and_then(|value| value.as_object())
        else {
            return false;
        };

        let shards: BTreeSet<String> =
            weight_map.values().filter_map(|value| value.as_str()).map(ToOwned::to_owned).collect();
        if shards.is_empty() {
            return false;
        }
        return shards.iter().all(|name| mlx_weight_file_ready(&path.join(name)));
    }

    let Ok(entries) = std::fs::read_dir(path) else {
        return false;
    };
    let mut saw_weights = false;
    for entry in entries.flatten() {
        let weight_path = entry.path();
        if weight_path.extension().and_then(|ext| ext.to_str()) == Some("safetensors") {
            saw_weights = true;
            if !mlx_weight_file_ready(&weight_path) {
                return false;
            }
        }
    }
    saw_weights
}

fn mlx_weight_file_ready(path: &Path) -> bool {
    std::fs::metadata(path).map(|meta| meta.is_file() && meta.len() > 1024 * 1024).unwrap_or(false)
}

/// Download a HuggingFace MLX snapshot via the `hf` CLI (bundled with the
/// huggingface-hub Python package). On macOS, `brew install jundot/omlx/omlx`
/// pulls it in transitively; otherwise `pip install huggingface-hub`.
///
/// Idempotent — returns the existing path if already cached. The CLI itself
/// resumes partial downloads, so a crashed prior run doesn't re-fetch.
///
/// Blocking async via `tokio::process::Command` so we don't stall the
/// runtime during a multi-GB pull.
pub async fn ensure_mlx_snapshot(
    models_cache_dir: &Path,
    repo_id: &str,
    folder: &str,
) -> Result<PathBuf> {
    let dest = mlx_model_path(models_cache_dir, folder);
    if mlx_model_cached(models_cache_dir, folder) {
        info!("MLX snapshot {} already cached at {}", repo_id, dest.display());
        return Ok(dest);
    }

    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("creating mlx cache dir {}", parent.display()))?;
    }

    let hf_bin = locate_hf_cli()
        .ok_or_else(|| anyhow::anyhow!(
            "huggingface CLI (`hf` or `huggingface-cli`) not found — install via `pip install huggingface-hub` or rely on the oMLX brew formula which pulls it in"
        ))?;

    info!("Downloading MLX snapshot {} -> {}", repo_id, dest.display());
    let output = tokio::process::Command::new(&hf_bin)
        .arg("download")
        .arg(repo_id)
        .arg("--local-dir")
        .arg(&dest)
        .output()
        .await
        .with_context(|| format!("invoking {}", hf_bin.display()))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("hf download {repo_id} failed: {}", stderr.trim());
    }

    if !mlx_model_cached(models_cache_dir, folder) {
        anyhow::bail!(
            "hf download {repo_id} returned success but config.json is missing at {}",
            dest.display()
        );
    }

    info!("MLX snapshot ready at {}", dest.display());
    Ok(dest)
}

fn locate_hf_cli() -> Option<PathBuf> {
    // `hf` is the new-style CLI (huggingface-hub ≥ 0.x); `huggingface-cli`
    // is the legacy alias, still provided by the package.
    let candidates: [PathBuf; 4] = [
        PathBuf::from("/opt/homebrew/bin/hf"),
        PathBuf::from("/usr/local/bin/hf"),
        PathBuf::from("/opt/homebrew/bin/huggingface-cli"),
        PathBuf::from("/usr/local/bin/huggingface-cli"),
    ];
    for c in candidates.iter() {
        if c.exists() {
            return Some(c.clone());
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        for py in ["3.9", "3.10", "3.11", "3.12", "3.13"] {
            let p = PathBuf::from(&home).join(format!("Library/Python/{py}/bin/hf"));
            if p.exists() {
                return Some(p);
            }
        }
    }
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(':') {
            for name in ["hf", "huggingface-cli"] {
                let p = PathBuf::from(dir).join(name);
                if p.exists() {
                    return Some(p);
                }
            }
        }
    }
    None
}

fn gguf_magic_ok(path: &Path) -> Result<bool> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    Ok(&magic == b"GGUF")
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
            entries.flatten().any(|e| e.file_name().to_string_lossy().ends_with(".index.json"))
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
            model_id,
            start_layer,
            end_layer,
            dest_dir.display()
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
                anyhow::bail!("Artifact checksum mismatch: expected {}, got {}", sha, actual);
            }
            debug!("Checksum OK");
        }
    }

    if let Some(expected) = expected_size {
        if expected > 0 {
            let actual = std::fs::metadata(&archive_path).map(|m| m.len()).unwrap_or(0);
            if actual != expected {
                warn!("Artifact size mismatch: expected {} bytes, got {}", expected, actual);
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

    info!("Stage artifacts ready at {}", dest_dir.display());
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
        .map(|_| path.extension().map(|e| e == "tar" || e == "gz" || e == "tgz").unwrap_or(false))
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

pub async fn download_file(url: &str, dest: &Path) -> Result<()> {
    let client =
        reqwest::Client::builder().timeout(std::time::Duration::from_secs(7200)).build()?;

    let resp = client.get(url).send().await.context("Download request failed")?;

    if !resp.status().is_success() {
        anyhow::bail!("Download failed: HTTP {}", resp.status());
    }

    let total_size = resp.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;

    let tmp_path = dest.with_extension("tmp");
    let mut file =
        tokio::fs::File::create(&tmp_path).await.context("Failed to create temp file")?;

    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;
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

    tokio::fs::rename(&tmp_path, dest).await.context("Failed to move downloaded artifact")?;

    info!("Downloaded {:.2} GB to {}", downloaded as f64 / 1_073_741_824.0, dest.display());
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
