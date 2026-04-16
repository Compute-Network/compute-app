use anyhow::Result;
use compute_daemon::config::Config;
use compute_daemon::hardware::HardwareInfo;
use compute_daemon::stage_runtime::{StagePrototypeSpec, start_stage_prototype_with_bind_addr};
use std::env;
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let bind_addr: SocketAddr = args
        .get(1)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or_else(|| "127.0.0.1:9090".parse().expect("valid default bind addr"));
    let stage_index = args.get(2).and_then(|value| value.parse::<u32>().ok()).unwrap_or(0);
    let total_stages = args.get(3).and_then(|value| value.parse::<u32>().ok()).unwrap_or(2);
    let start_layer = args.get(4).and_then(|value| value.parse::<u32>().ok()).unwrap_or(0);
    let end_layer = args.get(5).and_then(|value| value.parse::<u32>().ok()).unwrap_or(20);
    let model_name = args.get(6).cloned().unwrap_or_else(|| "gemma-4-e4b-q4".to_string());
    let downstream_addr = args.get(7).cloned();

    println!("=== Prototype Stage Runtime Host ===");
    println!("bind addr     : {}", bind_addr);
    println!("stage index   : {}", stage_index);
    println!("total stages  : {}", total_stages);
    println!("layers        : {}-{}", start_layer, end_layer);
    println!("model         : {}", model_name);
    println!("downstream    : {:?}", downstream_addr);

    let mut config = Config::default();
    config.experimental.stage_backend = "prototype".to_string();
    let hw = HardwareInfo::empty();
    let handle = start_stage_prototype_with_bind_addr(
        &config,
        &hw,
        StagePrototypeSpec {
            pipeline_id: "prototype-stage-runtime-host".into(),
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
    println!("status        : stopped");
    Ok(())
}
