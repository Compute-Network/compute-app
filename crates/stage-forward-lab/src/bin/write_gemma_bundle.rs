use anyhow::Result;
use stage_forward_lab::gguf::GgufFile;
use std::path::PathBuf;

fn main() -> Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap()
                .join(".compute")
                .join("models")
                .join("gemma-4-E4B-it-Q4_K_M.gguf")
        });

    let out_dir = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "/Users/macintosh/Documents/projects/Compute/compute-backend/out/gemma-e4b-2stage",
            )
        });

    let file = GgufFile::parse_file(&model_path)?;
    let splits = file
        .suggest_even_stage_split(2)
        .ok_or_else(|| anyhow::anyhow!("Could not infer a 2-stage split from GGUF"))?;
    let plan = file.plan_for_splits(&splits);
    let written = plan.write_bundle(&file, &out_dir)?;

    println!("bundle root   : {}", written.root_dir.display());
    println!("manifest      : {}", written.manifest_path.display());
    for path in written.stage_manifest_paths {
        println!("stage manifest: {}", path.display());
    }

    Ok(())
}
