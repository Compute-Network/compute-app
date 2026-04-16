// Integration probe for the new ManagedTailNode + ManagedHeadGatewayStack split.
// Spawns:
//   - tail-only stage worker on 127.0.0.1:0 (simulates the remote tail machine)
//   - head + gateway pointing at the tail worker (simulates the head machine)
// Connects a GatewayServiceClient and runs a complete() round-trip.
use anyhow::{Context, Result};
use llama_stage_backend::{
    GatewayServiceClient, ManagedGatewayLaunchSpec, ManagedHeadGatewayStack, ManagedTailNode,
    default_gemma_model_path,
};
use std::time::Instant;

fn main() -> Result<()> {
    let model_path = default_gemma_model_path();
    if !model_path.exists() {
        anyhow::bail!("model not found: {}", model_path.display());
    }
    eprintln!("[probe] model = {}", model_path.display());

    // Pretend gemma-4-e4b-q4: total_layers=42, head=0..=20, tail=21..=41.
    let head_start = 0u32;
    let head_end = 20u32;
    let tail_start = 21u32;
    let tail_end = 41u32;

    let launch_spec = ManagedGatewayLaunchSpec::default();

    eprintln!("[probe] spawning tail worker (layers {tail_start}-{tail_end})");
    let t0 = Instant::now();
    let tail = ManagedTailNode::spawn(
        model_path.clone(),
        "127.0.0.1:0",
        tail_start,
        tail_end,
        &launch_spec,
    )
    .context("spawning tail node")?;
    eprintln!(
        "[probe] tail listening on {} ({:.1}s)",
        tail.addr(),
        t0.elapsed().as_secs_f64()
    );

    eprintln!("[probe] spawning head + gateway (head layers {head_start}-{head_end}, tail={})", tail.addr());
    let t1 = Instant::now();
    let stack = ManagedHeadGatewayStack::spawn_with_remote_tail(
        model_path.clone(),
        head_start,
        head_end,
        tail.addr(),
        false,
        &launch_spec,
    )
    .context("spawning head + gateway")?;
    eprintln!(
        "[probe] gateway ready at {} ({:.1}s)",
        stack.gateway_addr(),
        t1.elapsed().as_secs_f64()
    );

    eprintln!("[probe] connecting client to gateway");
    let mut client = GatewayServiceClient::connect(stack.gateway_addr())
        .context("connect gateway client")?;

    let prompt = "The capital of France is".to_string();
    eprintln!("[probe] complete prompt={prompt:?}");
    let t2 = Instant::now();
    let completion = client
        .complete("split-probe-1".to_string(), prompt.clone(), 24)
        .context("complete request")?;
    let total = t2.elapsed().as_secs_f64();

    eprintln!("[probe] tokens={} text={:?}", completion.completion_tokens, completion.text);
    eprintln!(
        "[probe] timings ttft={:?}ms total={:?}ms",
        completion.timings.ttft_ms, completion.timings.total_ms
    );
    let tps = if total > 0.0 {
        completion.completion_tokens as f64 / total
    } else {
        0.0
    };
    eprintln!("[probe] elapsed={:.2}s tps={:.2}", total, tps);

    if completion.completion_tokens == 0 {
        anyhow::bail!("no tokens generated");
    }
    Ok(())
}
