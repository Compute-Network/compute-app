use anyhow::Result;
use compute_daemon::config::Config;
use compute_daemon::hardware::HardwareInfo;
use compute_daemon::real_chain::{
    RealStageArtifactSpec, prepare_temp_stage_root, restore_stage_root, stage_dir_from_artifact,
};
use compute_daemon::stage_runtime::{StagePrototypeSpec, start_stage_prototype_with_bind_addr};
use std::env;
use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let artifact_path = args.get(1).map(PathBuf::from);
    let bind_addr: SocketAddr = args
        .get(2)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or_else(|| "127.0.0.1:9090".parse().expect("valid default bind addr"));
    let stage_index = args.get(3).and_then(|value| value.parse::<u32>().ok()).unwrap_or(1);
    let total_stages = args.get(4).and_then(|value| value.parse::<u32>().ok()).unwrap_or(2);
    let start_layer = args.get(5).and_then(|value| value.parse::<u32>().ok()).unwrap_or(21);
    let end_layer = args.get(6).and_then(|value| value.parse::<u32>().ok()).unwrap_or(41);
    let model_name = args.get(7).cloned().unwrap_or_else(|| "gemma-4-e4b-q4".to_string());
    let downstream_addr = args.get(8).cloned();

    println!("=== Real Stage Runtime Host ===");
    println!("artifact path : {:?}", artifact_path);
    println!("bind addr     : {}", bind_addr);
    println!("stage index   : {}", stage_index);
    println!("total stages  : {}", total_stages);
    println!("layers        : {}-{}", start_layer, end_layer);
    println!("model         : {}", model_name);
    println!("downstream    : {:?}", downstream_addr);

    let previous_stage_root = env::var_os("COMPUTE_STAGE_ROOT");
    let temp_stage_root = if let Some(artifact_path) = artifact_path.as_ref() {
        let artifact_dir = stage_dir_from_artifact(artifact_path)?;
        let stage_root = prepare_temp_stage_root(
            &model_name,
            &[RealStageArtifactSpec { artifact_path: artifact_dir, start_layer, end_layer }],
        )?;
        // This process owns the hosted runtime for its lifetime.
        unsafe {
            env::set_var("COMPUTE_STAGE_ROOT", &stage_root);
        }
        println!("stage root    : {}", stage_root.display());
        Some(stage_root)
    } else {
        println!("stage root    : <env/default>");
        None
    };

    let mut config = Config::default();
    config.experimental.stage_backend = "real_forward".to_string();
    let hw = HardwareInfo::empty();
    let handle = start_stage_prototype_with_bind_addr(
        &config,
        &hw,
        StagePrototypeSpec {
            pipeline_id: "real-stage-runtime-host".into(),
            model_name: model_name.clone(),
            shard_id: format!("stage-{stage_index}"),
            start_layer,
            end_layer,
            stage_index,
            total_stages,
            upstream_addr: None,
            downstream_addr,
        },
        bind_addr,
    )
    .await?;

    println!("listen addr   : {}", handle.listen_addr());
    println!("status        : ready");
    println!("ctrl-c        : stop");

    tokio::signal::ctrl_c().await?;

    handle.stop().await;
    restore_stage_root(previous_stage_root);
    if let Some(stage_root) = temp_stage_root {
        let _ = fs::remove_dir_all(stage_root);
    }
    println!("status        : stopped");
    Ok(())
}
