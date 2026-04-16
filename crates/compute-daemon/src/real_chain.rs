use anyhow::Result;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct RealStageArtifactSpec {
    pub artifact_path: PathBuf,
    pub start_layer: u32,
    pub end_layer: u32,
}

pub fn stage_dir_from_artifact(path: &Path) -> Result<PathBuf> {
    if path.is_dir() {
        return Ok(path.canonicalize()?);
    }
    path.parent()
        .ok_or_else(|| anyhow::anyhow!("Cannot resolve parent of {}", path.display()))?
        .canonicalize()
        .map_err(Into::into)
}

pub fn prepare_temp_stage_root(
    model_name: &str,
    specs: &[RealStageArtifactSpec],
) -> Result<PathBuf> {
    let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let root = env::temp_dir()
        .join(format!("compute-stage-root-{}-{unique}", std::process::id()))
        .join("stages");
    let model_root = root.join(model_name);
    fs::create_dir_all(&model_root)?;
    for spec in specs {
        let stage_dir = stage_dir_from_artifact(&spec.artifact_path)?;
        mirror_dir(
            &stage_dir,
            &model_root.join(format!("packed-stage-{}-{}", spec.start_layer, spec.end_layer)),
        )?;
    }
    Ok(root)
}

pub fn restore_stage_root(previous: Option<std::ffi::OsString>) {
    if let Some(previous) = previous {
        unsafe {
            env::set_var("COMPUTE_STAGE_ROOT", previous);
        }
    } else {
        unsafe {
            env::remove_var("COMPUTE_STAGE_ROOT");
        }
    }
}

fn mirror_dir(src: &Path, dst: &Path) -> Result<()> {
    if dst.exists() {
        fs::remove_dir_all(dst)?;
    }
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }

    #[cfg(unix)]
    {
        if std::os::unix::fs::symlink(src, dst).is_ok() {
            return Ok(());
        }
    }

    copy_dir_all(src, dst)
}

fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let target = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_all(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}
