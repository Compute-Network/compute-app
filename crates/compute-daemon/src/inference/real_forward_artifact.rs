use anyhow::Result;
use stage_forward_lab::StageLayout;
use std::path::{Path, PathBuf};

use crate::inference::engine::ShardConfig;

#[derive(Debug, Clone)]
pub struct RealForwardStageLoadSpec {
    pub config: ShardConfig,
    pub stage_dir: PathBuf,
    pub index_path: PathBuf,
    pub vocab_path: Option<PathBuf>,
    pub vocab_scores_path: Option<PathBuf>,
    pub layout: StageLayout,
}

impl RealForwardStageLoadSpec {
    pub fn from_shard_config(config: &ShardConfig) -> Result<Self> {
        let stage_dir = resolve_packed_stage_dir(config)?;
        let index_name = find_stage_index(&stage_dir)?;
        let index_path = stage_dir.join(index_name);

        let vocab_path = stage_dir.join("vocab.json");
        let vocab_path = vocab_path.exists().then_some(vocab_path);

        let vocab_scores_path = stage_dir.join("vocab_scores.json");
        let vocab_scores_path = vocab_scores_path.exists().then_some(vocab_scores_path);

        let layout = StageLayout {
            model_id: config.model_id.clone(),
            stage_id: format!("stage-{}-{}", config.start_layer, config.end_layer),
            start_layer: config.start_layer,
            end_layer: config.end_layer,
            is_head: config.is_first_stage,
            is_tail: config.is_last_stage,
        };

        Ok(Self {
            config: config.clone(),
            stage_dir,
            index_path,
            vocab_path,
            vocab_scores_path,
            layout,
        })
    }
}

pub(crate) fn resolve_packed_stage_dir(config: &ShardConfig) -> Result<PathBuf> {
    let shard_path = Path::new(&config.shard_path);
    if shard_path.is_dir() {
        return Ok(shard_path.to_path_buf());
    }
    if shard_path.is_file() && shard_path.extension().is_some_and(|e| e == "json") {
        return shard_path
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| anyhow::anyhow!("Cannot resolve parent of {}", shard_path.display()));
    }
    let stage_root =
        std::env::var_os("COMPUTE_STAGE_ROOT").map(PathBuf::from).unwrap_or_else(|| {
            dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join(".compute").join("stages")
        });
    let compute_dir = stage_root
        .join(&config.model_id)
        .join(format!("packed-stage-{}-{}", config.start_layer, config.end_layer));
    if compute_dir.exists() {
        return Ok(compute_dir);
    }
    anyhow::bail!(
        "Could not find packed stage directory for {} layers {}-{} (tried {} and {})",
        config.model_id,
        config.start_layer,
        config.end_layer,
        shard_path.display(),
        compute_dir.display()
    )
}

pub(crate) fn find_stage_index(dir: &Path) -> Result<String> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".index.json") {
            return Ok(name);
        }
    }
    anyhow::bail!("No .index.json file found in {}", dir.display())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn test_config(shard_path: PathBuf) -> ShardConfig {
        ShardConfig {
            model_id: "gemma-4-e4b-q4".into(),
            shard_path,
            start_layer: 0,
            end_layer: 20,
            total_layers: 42,
            is_first_stage: true,
            is_last_stage: false,
            max_batch_size: 16,
            context_length: 8192,
        }
    }

    #[test]
    fn load_spec_resolves_index_and_layout_from_index_path() {
        let temp = tempdir().unwrap();
        let stage_dir = temp.path().join("packed-stage-0-20");
        std::fs::create_dir_all(&stage_dir).unwrap();
        let index_path = stage_dir.join("stage-1-required.index.json");
        std::fs::write(&index_path, "{}").unwrap();
        std::fs::write(stage_dir.join("vocab.json"), "{}").unwrap();
        std::fs::write(stage_dir.join("vocab_scores.json"), "[]").unwrap();

        let spec = RealForwardStageLoadSpec::from_shard_config(&test_config(index_path)).unwrap();
        assert_eq!(spec.stage_dir, stage_dir);
        assert!(spec.index_path.ends_with("stage-1-required.index.json"));
        assert!(spec.vocab_path.is_some());
        assert!(spec.vocab_scores_path.is_some());
        assert_eq!(spec.layout.stage_id, "stage-0-20");
        assert!(spec.layout.is_head);
        assert!(!spec.layout.is_tail);
    }

    #[test]
    fn load_spec_accepts_stage_directory_directly() {
        let temp = tempdir().unwrap();
        let stage_dir = temp.path().join("packed-stage-21-41");
        std::fs::create_dir_all(&stage_dir).unwrap();
        std::fs::write(stage_dir.join("stage-2-required.index.json"), "{}").unwrap();

        let mut config = test_config(stage_dir.clone());
        config.start_layer = 21;
        config.end_layer = 41;
        config.is_first_stage = false;
        config.is_last_stage = true;

        let spec = RealForwardStageLoadSpec::from_shard_config(&config).unwrap();
        assert_eq!(spec.stage_dir, stage_dir);
        assert!(spec.vocab_path.is_none());
        assert!(spec.vocab_scores_path.is_none());
        assert_eq!(spec.layout.stage_id, "stage-21-41");
        assert!(!spec.layout.is_head);
        assert!(spec.layout.is_tail);
    }
}
