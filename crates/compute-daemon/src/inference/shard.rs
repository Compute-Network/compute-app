//! Model shard download and cache management.
//!
//! Handles downloading model shards from HuggingFace or CDN,
//! verifying checksums, and managing the local cache.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// A downloadable model shard.
#[derive(Debug, Clone)]
pub struct ShardDownload {
    /// Model identifier.
    pub model_id: String,
    /// Shard filename (e.g. "llama-3.1-70b-q4-layers-16-31.gguf").
    pub filename: String,
    /// Download URL.
    pub url: String,
    /// Expected SHA256 checksum.
    pub sha256: String,
    /// Expected file size in bytes.
    pub size_bytes: u64,
    /// Layer range this shard covers.
    pub start_layer: u32,
    pub end_layer: u32,
}

/// Manages the local shard cache.
pub struct ShardCache {
    cache_dir: PathBuf,
}

impl ShardCache {
    /// Create a new shard cache at the given directory.
    pub fn new(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    /// Default cache directory (~/.compute/models/).
    pub fn default_dir() -> PathBuf {
        crate::config::config_dir().unwrap_or_else(|| PathBuf::from("~/.compute")).join("models")
    }

    /// Ensure the cache directory exists.
    pub fn ensure_dir(&self) -> Result<()> {
        std::fs::create_dir_all(&self.cache_dir).context("Failed to create shard cache directory")
    }

    /// Check if a shard is already cached and valid.
    pub fn is_cached(&self, shard: &ShardDownload) -> bool {
        let path = self.shard_path(shard);
        if !path.exists() {
            return false;
        }

        // Check file size matches
        if let Ok(metadata) = std::fs::metadata(&path)
            && shard.size_bytes > 0
            && metadata.len() != shard.size_bytes
        {
            debug!(
                "Cached shard {} has wrong size (expected {}, got {})",
                shard.filename,
                shard.size_bytes,
                metadata.len()
            );
            return false;
        }

        true
    }

    /// Get the local path for a shard.
    pub fn shard_path(&self, shard: &ShardDownload) -> PathBuf {
        self.cache_dir.join(&shard.model_id).join(&shard.filename)
    }

    /// Download a shard if not already cached. Returns the local path.
    pub async fn ensure_shard(&self, shard: &ShardDownload) -> Result<PathBuf> {
        let path = self.shard_path(shard);

        if self.is_cached(shard) {
            info!("Shard {} already cached at {}", shard.filename, path.display());
            return Ok(path);
        }

        self.ensure_dir()?;

        // Create model subdirectory
        let model_dir = self.cache_dir.join(&shard.model_id);
        std::fs::create_dir_all(&model_dir)?;

        info!(
            "Downloading shard {} ({:.1} GB) from {}",
            shard.filename,
            shard.size_bytes as f64 / 1_073_741_824.0,
            shard.url
        );

        download_file(&shard.url, &path).await?;

        // Verify checksum if provided
        if !shard.sha256.is_empty() {
            info!("Verifying checksum for {}", shard.filename);
            let actual = sha256_file(&path)?;
            if actual != shard.sha256 {
                // Delete corrupt file
                let _ = std::fs::remove_file(&path);
                anyhow::bail!(
                    "Checksum mismatch for {}: expected {}, got {}",
                    shard.filename,
                    shard.sha256,
                    actual
                );
            }
            debug!("Checksum OK: {}", shard.sha256);
        }

        info!("Shard {} downloaded to {}", shard.filename, path.display());
        Ok(path)
    }

    /// List all cached model shards.
    pub fn list_cached(&self) -> Vec<(String, Vec<String>)> {
        let mut result = Vec::new();

        if let Ok(entries) = std::fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    let model_id = entry.file_name().to_string_lossy().to_string();
                    let mut files = Vec::new();

                    if let Ok(shard_entries) = std::fs::read_dir(entry.path()) {
                        for shard in shard_entries.flatten() {
                            files.push(shard.file_name().to_string_lossy().to_string());
                        }
                    }

                    if !files.is_empty() {
                        result.push((model_id, files));
                    }
                }
            }
        }

        result
    }

    /// Get total cache size in bytes.
    pub fn cache_size_bytes(&self) -> u64 {
        dir_size(&self.cache_dir)
    }

    /// Remove a specific model's cached shards.
    pub fn remove_model(&self, model_id: &str) -> Result<()> {
        let model_dir = self.cache_dir.join(model_id);
        if model_dir.exists() {
            std::fs::remove_dir_all(&model_dir)
                .context(format!("Failed to remove cache for {model_id}"))?;
            info!("Removed cached shards for {model_id}");
        }
        Ok(())
    }

    /// Clear the entire cache.
    pub fn clear(&self) -> Result<()> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir)?;
            std::fs::create_dir_all(&self.cache_dir)?;
            info!("Shard cache cleared");
        }
        Ok(())
    }
}

/// Download a file from URL to disk with streaming.
async fn download_file(url: &str, dest: &Path) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600)) // 1 hour for large files
        .build()?;

    let resp = client.get(url).send().await.context("Download request failed")?;

    if !resp.status().is_success() {
        anyhow::bail!("Download failed: HTTP {}", resp.status());
    }

    let total_size = resp.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;

    // Write to a temp file first, then rename (atomic)
    let tmp_path = dest.with_extension("tmp");
    let mut file =
        tokio::fs::File::create(&tmp_path).await.context("Failed to create temp file")?;

    use tokio::io::AsyncWriteExt;
    let mut stream = resp.bytes_stream();
    use futures_util::StreamExt;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Error reading download stream")?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;

        // Log progress every ~100MB
        if total_size > 0 && downloaded % (100 * 1024 * 1024) < chunk.len() as u64 {
            let pct = (downloaded as f64 / total_size as f64) * 100.0;
            debug!("Download progress: {pct:.0}% ({downloaded} / {total_size})");
        }
    }

    file.flush().await?;
    drop(file);

    // Rename temp file to final destination
    tokio::fs::rename(&tmp_path, dest).await.context("Failed to move downloaded file")?;

    Ok(())
}

/// Compute SHA256 of a file.
fn sha256_file(path: &Path) -> Result<String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024]; // 8MB chunks

    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    Ok(hasher.finalize_hex())
}

/// Simple SHA-256 implementation (no external dep needed for this).
/// Uses the system `shasum` command as a pragmatic approach.
struct Sha256 {
    data: Vec<u8>,
}

impl Sha256 {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn update(&mut self, chunk: &[u8]) {
        self.data.extend_from_slice(chunk);
    }

    fn finalize_hex(self) -> String {
        // Use system shasum for now — swap for ring/sha2 crate if needed
        use std::io::Write;
        let mut child = std::process::Command::new("shasum")
            .args(["-a", "256"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("shasum not found");

        if let Some(ref mut stdin) = child.stdin {
            let _ = stdin.write_all(&self.data);
        }

        let output = child.wait_with_output().expect("shasum failed");
        let hash = String::from_utf8_lossy(&output.stdout);
        hash.split_whitespace().next().unwrap_or("").to_string()
    }
}

/// Recursively compute directory size.
fn dir_size(path: &Path) -> u64 {
    if !path.exists() {
        return 0;
    }
    std::fs::read_dir(path)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| {
                    let p = e.path();
                    if p.is_dir() {
                        dir_size(&p)
                    } else {
                        e.metadata().map(|m| m.len()).unwrap_or(0)
                    }
                })
                .sum()
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_cache_default_dir() {
        let dir = ShardCache::default_dir();
        assert!(dir.to_string_lossy().contains("models"));
    }

    #[test]
    fn test_shard_path() {
        let cache = ShardCache::new(PathBuf::from("/tmp/compute-test-cache"));
        let shard = ShardDownload {
            model_id: "llama-70b".into(),
            filename: "shard-0-39.gguf".into(),
            url: "https://example.com/shard.gguf".into(),
            sha256: "abc123".into(),
            size_bytes: 1024,
            start_layer: 0,
            end_layer: 39,
        };

        let path = cache.shard_path(&shard);
        assert_eq!(path, PathBuf::from("/tmp/compute-test-cache/llama-70b/shard-0-39.gguf"));
    }

    #[test]
    fn test_is_cached_missing() {
        let cache = ShardCache::new(PathBuf::from("/tmp/compute-nonexistent-cache-dir"));
        let shard = ShardDownload {
            model_id: "test".into(),
            filename: "test.gguf".into(),
            url: String::new(),
            sha256: String::new(),
            size_bytes: 0,
            start_layer: 0,
            end_layer: 0,
        };
        assert!(!cache.is_cached(&shard));
    }
}
