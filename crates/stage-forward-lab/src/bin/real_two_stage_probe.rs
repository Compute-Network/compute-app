use anyhow::Result;
use stage_forward_lab::real_forward::RealGemmaBackend;
use stage_forward_lab::{StageForwardBackend, StageLayout};
use std::env;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let stage1_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json")
    });
    let stage2_path = args.get(2).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("out/gemma-e4b-2stage/packed-stage-2/stage-2-required.index.json")
    });
    let prompt = args.get(3).cloned().unwrap_or_else(|| "Hello".to_string());
    let vocab_path = args.get(4).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("out/gemma-e4b-2stage/vocab.json")
    });
    let scores_path = PathBuf::from("out/gemma-e4b-2stage/vocab_scores.json");

    println!("=== Real 2-Stage Gemma Forward ===");
    println!("stage 1  : {}", stage1_path.display());
    println!("stage 2  : {}", stage2_path.display());
    println!("prompt   : {:?}", prompt);
    println!();

    let t0 = Instant::now();
    let mut head = RealGemmaBackend::new(&stage1_path);
    if vocab_path.exists() {
        let sp = if scores_path.exists() { Some(scores_path.as_path()) } else { None };
        head.load_tokenizer(&vocab_path, sp)?;
    }
    head.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-1".into(),
        start_layer: 0,
        end_layer: 20,
        is_head: true,
        is_tail: false,
    })?;
    println!("head loaded      : {}ms", t0.elapsed().as_millis());

    let t1 = Instant::now();
    let mut tail = RealGemmaBackend::new(&stage2_path);
    if vocab_path.exists() {
        let sp = if scores_path.exists() { Some(scores_path.as_path()) } else { None };
        tail.load_tokenizer(&vocab_path, sp)?;
        println!("tokenizer loaded : {}", vocab_path.display());
    }
    tail.load_layout(StageLayout {
        model_id: "gemma-4-e4b-q4".into(),
        stage_id: "stage-2".into(),
        start_layer: 21,
        end_layer: 41,
        is_head: false,
        is_tail: true,
    })?;
    println!("tail loaded      : {}ms", t1.elapsed().as_millis());
    println!();

    let t_head = Instant::now();
    let head_output = head.begin_prompt("probe-req", &prompt, Some(1), 0)?;
    let head_ms = t_head.elapsed().as_millis();

    let head_state: Vec<f32> = head_output.bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    println!("=== Stage 1 (Head) ===");
    println!("forward time     : {}ms", head_ms);
    println!("hidden_dim       : {}", head_output.hidden_dim);
    println!("stage_trace      : {:?}", head_output.stage_trace);
    print_stats("head", &head_state);
    println!();

    let t_tail = Instant::now();
    let tail_output = tail.continue_forward(head_output)?;
    let tail_ms = t_tail.elapsed().as_millis();

    let tail_state: Vec<f32> = tail_output.bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    println!("=== Stage 2 (Tail Forward) ===");
    println!("forward time     : {}ms", tail_ms);
    println!("hidden_dim       : {}", tail_output.hidden_dim);
    println!("stage_trace      : {:?}", tail_output.stage_trace);
    print_stats("tail_fwd", &tail_state);
    println!();

    let t_sample = Instant::now();
    let sample = tail.sample_tail(tail_output)?;
    let sample_ms = t_sample.elapsed().as_millis();

    println!("=== Sampling ===");
    println!("sample time      : {}ms", sample_ms);
    println!("text             : {:?}", sample.text);
    println!("tokens           : {}", sample.completion_tokens);
    println!();

    println!("=== Total ===");
    println!("head forward     : {}ms", head_ms);
    println!("tail forward     : {}ms", tail_ms);
    println!("sampling         : {}ms", sample_ms);
    println!("total            : {}ms", head_ms + tail_ms + sample_ms);

    Ok(())
}

fn print_stats(label: &str, state: &[f32]) {
    let finite = state.iter().filter(|v| v.is_finite()).count();
    let nan = state.iter().filter(|v| v.is_nan()).count();
    let inf = state.iter().filter(|v| v.is_infinite()).count();
    let min = state.iter().copied().fold(f32::INFINITY, f32::min);
    let max = state.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = state.iter().sum::<f32>() / state.len() as f32;
    let rms = (state.iter().map(|v| v * v).sum::<f32>() / state.len() as f32).sqrt();
    let preview: Vec<String> = state[..8.min(state.len())].iter().map(|v| format!("{:.4}", v)).collect();

    println!("{} finite      : {}/{}", label, finite, state.len());
    println!("{} nan/inf     : {}/{}", label, nan, inf);
    println!("{} range       : [{:.4}, {:.4}]", label, min, max);
    println!("{} mean/rms    : {:.4} / {:.4}", label, mean, rms);
    println!("{} preview     : [{}]", label, preview.join(", "));
}
