use anyhow::{Context, Result};
use compute_daemon::inference::ggml_stage_executor::build_ggml_stage_executor;
use compute_daemon::inference::ggml_stage_worker::{
    GgmlStageWorkerInitSpec, GgmlStageWorkerRequest, GgmlStageWorkerWireResponse,
    handle_stage_worker_request, read_framed_json,
};
use std::env;
use std::io::Write;
use std::net::TcpListener;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let init_json = args
        .windows(2)
        .find_map(|window| (window[0] == "--init-json").then(|| window[1].clone()))
        .context("missing required --init-json argument")?;
    let request_json = args
        .windows(2)
        .find_map(|window| (window[0] == "--request-json").then(|| window[1].clone()));
    let response_file = args
        .windows(2)
        .find_map(|window| (window[0] == "--response-file").then(|| window[1].clone()));
    let listen_addr = args
        .windows(2)
        .find_map(|window| (window[0] == "--listen-addr").then(|| window[1].clone()));
    let ready_file =
        args.windows(2).find_map(|window| (window[0] == "--ready-file").then(|| window[1].clone()));
    let request_file = args
        .windows(2)
        .find_map(|window| (window[0] == "--request-file").then(|| window[1].clone()));

    let init: GgmlStageWorkerInitSpec =
        serde_json::from_str(&init_json).context("parse --init-json payload")?;
    let mut executor = build_ggml_stage_executor(&init)?;

    if let Some(listen_addr) = listen_addr {
        let listener = TcpListener::bind(&listen_addr)
            .with_context(|| format!("bind ggml stage worker listener at {listen_addr}"))?;
        let bound_addr = listener.local_addr().context("resolve worker listener address")?;
        if let Some(ready_file) = ready_file {
            std::fs::write(&ready_file, bound_addr.to_string())
                .with_context(|| format!("write worker ready file to {ready_file}"))?;
        }
        for stream in listener.incoming() {
            let mut stream = stream.context("accept ggml stage worker connection")?;
            let request: GgmlStageWorkerRequest =
                read_framed_json(&mut stream).context("read worker framed request")?;
            let response = handle_stage_worker_request(executor.as_mut(), request)?;
            let wire_response = GgmlStageWorkerWireResponse::from(response);
            let payload = serde_json::to_vec(&wire_response)?;
            let len = u64::try_from(payload.len()).context("worker response frame exceeds u64")?;
            stream.write_all(&len.to_le_bytes())?;
            stream.write_all(&payload)?;
            stream.flush()?;
        }
        return Ok(());
    }

    if let Some(request_json) =
        request_json.or_else(|| request_file.and_then(|path| std::fs::read_to_string(path).ok()))
    {
        let request: GgmlStageWorkerRequest =
            serde_json::from_str(&request_json).context("parse --request-json payload")?;
        let response = handle_stage_worker_request(executor.as_mut(), request)?;
        let wire_response = GgmlStageWorkerWireResponse::from(response);
        if let Some(response_file) = response_file {
            std::fs::write(&response_file, serde_json::to_vec(&wire_response)?)
                .with_context(|| format!("write worker response file to {response_file}"))?;
        } else {
            println!("{}", serde_json::to_string(&wire_response)?);
        }
        return Ok(());
    }

    println!("=== GGML Stage Worker Host ===");
    println!("model        : {}", init.model_id);
    println!("stage        : {}", init.stage_id);
    println!("role         : {}", init.role);
    println!("stage dir    : {}", init.stage_dir.display());
    println!("index        : {}", init.index_path.display());
    println!(
        "tokenizer    : vocab={} vocab_scores={}",
        init.vocab_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".into()),
        init.vocab_scores_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".into())
    );
    println!("runtime      : {}", init.runtime.summary_label());
    println!("contract     : {}", init.contract.summary_label());
    println!("executor     : {}", executor.plan().summary_label());
    println!("status       : ready");
    println!("next         : swap in ggml-backed executor behind this worker contract");

    Ok(())
}
