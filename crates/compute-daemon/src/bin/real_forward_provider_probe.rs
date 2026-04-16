use anyhow::Result;
use compute_daemon::hardware::detect;
use compute_daemon::inference::engine::ShardConfig;
use compute_daemon::inference::stage_backend::{StageBackendKind, StageExecutionBackend};
use std::env;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let shard_path = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "/Users/macintosh/Documents/projects/Compute/compute-backend/out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json",
            )
        });
    let model_id = args.get(2).cloned().unwrap_or_else(|| "gemma-4-e4b-q4".to_string());
    let start_layer = args.get(3).and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
    let end_layer = args.get(4).and_then(|v| v.parse::<u32>().ok()).unwrap_or(20);
    let stage_role = args.get(5).map(|v| v.as_str()).unwrap_or("head");
    let stage_acceleration = args.get(6).map(|v| v.as_str()).unwrap_or("auto");
    let stage_provider = args.get(7).map(|v| v.as_str()).unwrap_or("auto");
    let prompt = args.get(8).cloned();

    let hw = detect();
    let mut backend = StageExecutionBackend::new_for_hardware(
        &hw,
        StageBackendKind::RealForward,
        stage_acceleration,
        stage_provider,
    );

    let is_head = matches!(stage_role, "head" | "first");
    let is_tail = matches!(stage_role, "tail" | "last");
    let config = ShardConfig {
        model_id: model_id.clone(),
        shard_path: shard_path.clone(),
        start_layer,
        end_layer,
        total_layers: end_layer + 1,
        is_first_stage: is_head,
        is_last_stage: is_tail,
        max_batch_size: 16,
        context_length: 8192,
    };

    println!("=== Real-Forward Provider Probe ===");
    println!("model        : {model_id}");
    println!("shard        : {}", shard_path.display());
    println!("layers       : {start_layer}-{end_layer}");
    println!("stage role   : {stage_role}");
    println!("requested    : {stage_acceleration}");
    println!("provider req : {stage_provider}");
    println!("backend      : {}", backend.backend_label());

    match backend.load_shard(&config).await {
        Ok(()) => {
            println!("load         : ok");
        }
        Err(err) => {
            println!("load         : error");
            println!("error        : {err}");
            anyhow::bail!(err);
        }
    }

    if let Some(prompt) = prompt {
        if is_head {
            let activation =
                backend.begin_prompt("provider-probe".into(), &prompt, Some(1), 0).await?;
            println!("prompt       : {:?}", prompt);
            println!("activation   : shape={:?} bytes={}", activation.shape, activation.data.len());
        } else {
            println!("prompt       : skipped (stage is not head)");
        }
    }

    Ok(())
}
