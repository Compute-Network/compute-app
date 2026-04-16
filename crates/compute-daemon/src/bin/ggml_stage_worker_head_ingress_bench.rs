use anyhow::{Result, bail};
use compute_daemon::inference::engine::ShardConfig;
use compute_daemon::inference::ggml_runtime::detect_ggml_runtime_plan;
use compute_daemon::inference::ggml_stage_executor::{
    GgmlHeadIngressProfile, GgmlStageExecutorKind, build_ggml_stage_executor,
};
use compute_daemon::inference::ggml_stage_worker::GgmlStageWorkerInitSpec;
use compute_daemon::inference::real_forward_artifact::RealForwardStageLoadSpec;
use compute_daemon::inference::stage_acceleration::StageAccelerationTarget;
use stage_forward_lab::prompt_suite::{ValidationPromptSuiteMode, validation_prompt_cases};
use std::env;
use std::path::PathBuf;

fn target_from_str(value: &str) -> StageAccelerationTarget {
    match value {
        "cpu" => StageAccelerationTarget::Cpu,
        "cuda" => StageAccelerationTarget::Cuda,
        "vulkan" => StageAccelerationTarget::Vulkan,
        "directml" => StageAccelerationTarget::DirectMl,
        _ => StageAccelerationTarget::Metal,
    }
}

#[derive(Default)]
struct HeadIngressProfileTotals {
    total_us: u128,
    embed_token_gather_us: u128,
    ple_token_gather_us: u128,
    ple_model_proj_us: u128,
    ple_normalize_combine_us: u128,
    prompt_aux_encode_us: u128,
    hidden_encode_us: u128,
    payload_frame_us: u128,
    other_us: u128,
    iterations: u64,
}

impl HeadIngressProfileTotals {
    fn record(&mut self, profile: &GgmlHeadIngressProfile) {
        self.total_us += u128::from(profile.total_us);
        self.embed_token_gather_us += u128::from(profile.embed_token_gather_us.unwrap_or(0));
        self.ple_token_gather_us += u128::from(profile.ple_token_gather_us.unwrap_or(0));
        self.ple_model_proj_us += u128::from(profile.ple_model_proj_us.unwrap_or(0));
        self.ple_normalize_combine_us += u128::from(profile.ple_normalize_combine_us.unwrap_or(0));
        self.prompt_aux_encode_us += u128::from(profile.prompt_aux_encode_us.unwrap_or(0));
        self.hidden_encode_us += u128::from(profile.hidden_encode_us.unwrap_or(0));
        self.payload_frame_us += u128::from(profile.payload_frame_us.unwrap_or(0));
        self.other_us += u128::from(profile.other_us.unwrap_or(0));
        self.iterations += u64::from(profile.iterations);
    }

    fn avg(bucket_total_us: u128, iterations: u64) -> f64 {
        bucket_total_us as f64 / iterations.max(1) as f64
    }

    fn hottest_bucket(&self) -> (&'static str, f64) {
        let mut buckets = [
            ("embed_token_gather", Self::avg(self.embed_token_gather_us, self.iterations)),
            ("ple_token_gather", Self::avg(self.ple_token_gather_us, self.iterations)),
            ("ple_model_proj", Self::avg(self.ple_model_proj_us, self.iterations)),
            ("ple_normalize_combine", Self::avg(self.ple_normalize_combine_us, self.iterations)),
            ("prompt_aux_encode", Self::avg(self.prompt_aux_encode_us, self.iterations)),
            ("hidden_encode", Self::avg(self.hidden_encode_us, self.iterations)),
            ("payload_frame", Self::avg(self.payload_frame_us, self.iterations)),
            ("other", Self::avg(self.other_us, self.iterations)),
        ];
        buckets.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        buckets[0]
    }
}

fn print_profile(label: &str, profile: &GgmlHeadIngressProfile) {
    println!(
        "{label:<18} avg_total_us={:>10.1} payload={} hidden={} aux={}",
        profile.avg_total_us(),
        profile.payload_bytes,
        profile.hidden_state_bytes,
        profile.aux_bytes
    );
    if profile.embed_token_gather_us.is_some() {
        println!(
            "{:<18} embed={:>10.1} ple_token={:>10.1} ple_proj={:>10.1} ple_norm={:>10.1}",
            "",
            GgmlHeadIngressProfile::avg_bucket_us(
                profile.embed_token_gather_us,
                profile.iterations
            )
            .unwrap_or_default(),
            GgmlHeadIngressProfile::avg_bucket_us(profile.ple_token_gather_us, profile.iterations)
                .unwrap_or_default(),
            GgmlHeadIngressProfile::avg_bucket_us(profile.ple_model_proj_us, profile.iterations)
                .unwrap_or_default(),
            GgmlHeadIngressProfile::avg_bucket_us(
                profile.ple_normalize_combine_us,
                profile.iterations
            )
            .unwrap_or_default(),
        );
        println!(
            "{:<18} aux_encode={:>10.1} hidden_encode={:>10.1} payload_frame={:>10.1} other={:>10.1}",
            "",
            GgmlHeadIngressProfile::avg_bucket_us(profile.prompt_aux_encode_us, profile.iterations)
                .unwrap_or_default(),
            GgmlHeadIngressProfile::avg_bucket_us(profile.hidden_encode_us, profile.iterations)
                .unwrap_or_default(),
            GgmlHeadIngressProfile::avg_bucket_us(profile.payload_frame_us, profile.iterations)
                .unwrap_or_default(),
            GgmlHeadIngressProfile::avg_bucket_us(profile.other_us, profile.iterations)
                .unwrap_or_default(),
        );
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let shard_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(
            "../compute-backend/out/gemma-e4b-2stage/packed-stage-1/stage-1-required.index.json",
        )
    });
    let model_id = args.get(2).cloned().unwrap_or_else(|| "gemma-4-e4b-q4".to_string());
    let start_layer = args.get(3).and_then(|value| value.parse::<u32>().ok()).unwrap_or(0);
    let end_layer = args.get(4).and_then(|value| value.parse::<u32>().ok()).unwrap_or(20);
    let stage_role = args.get(5).map(|value| value.as_str()).unwrap_or("head");
    let requested = args.get(6).map(|value| value.as_str()).unwrap_or("metal");
    let suite_mode = args
        .get(7)
        .and_then(|value| ValidationPromptSuiteMode::parse(value))
        .unwrap_or(ValidationPromptSuiteMode::Core);
    let iterations = args.get(8).and_then(|value| value.parse::<u32>().ok()).unwrap_or(20);
    let case_filter = args.get(9).cloned();

    let is_head = matches!(stage_role, "head" | "first" | "single");
    let is_tail = matches!(stage_role, "tail" | "last" | "single");
    if !is_head {
        bail!("head ingress bench requires a head or single stage, got `{stage_role}`");
    }

    let load_spec = RealForwardStageLoadSpec::from_shard_config(&ShardConfig {
        model_id: model_id.clone(),
        shard_path: shard_path.clone(),
        start_layer,
        end_layer,
        total_layers: end_layer + 1,
        is_first_stage: is_head,
        is_last_stage: is_tail,
        max_batch_size: 16,
        context_length: 8192,
    })?;
    let runtime = detect_ggml_runtime_plan(target_from_str(requested));

    let mut ggml_init =
        GgmlStageWorkerInitSpec::from_load_spec(&load_spec, &runtime, GgmlStageExecutorKind::Ggml);
    ggml_init.debug_layer_cap = Some(0);
    let mut cpu_init = GgmlStageWorkerInitSpec::from_load_spec(
        &load_spec,
        &runtime,
        GgmlStageExecutorKind::ReferenceCpu,
    );
    cpu_init.debug_layer_cap = Some(0);

    let mut ggml_executor = build_ggml_stage_executor(&ggml_init)?;
    let mut cpu_executor = build_ggml_stage_executor(&cpu_init)?;
    let mut ggml_totals = HeadIngressProfileTotals::default();
    let mut cpu_totals = HeadIngressProfileTotals::default();

    println!("=== GGML Stage Worker Head Ingress Bench ===");
    println!("model        : {model_id}");
    println!("shard        : {}", shard_path.display());
    println!("layers       : {start_layer}-{end_layer}");
    println!("stage role   : {stage_role}");
    println!("suite mode   : {}", suite_mode.as_str());
    println!("target       : {requested}");
    println!("iterations   : {}", iterations.max(1));
    println!("runtime      : {}", runtime.summary_label());
    println!("ggml plan    : {}", ggml_executor.plan().summary_label());
    println!("cpu-ref plan : {}", cpu_executor.plan().summary_label());
    if let Some(case_filter) = &case_filter {
        println!("case filter  : {case_filter}");
    }
    println!();

    for case in validation_prompt_cases(suite_mode) {
        if let Some(case_filter) = &case_filter
            && case.name != case_filter
        {
            continue;
        }

        let token_ids = ggml_executor.tokenize_generation_prompt(case.prompt)?;
        let max_tokens = Some(1);
        let _ = ggml_executor.profile_begin_token_ids_ingress(&token_ids, max_tokens, 1)?;
        let _ = cpu_executor.profile_begin_token_ids_ingress(&token_ids, max_tokens, 1)?;
        let ggml_profile =
            ggml_executor.profile_begin_token_ids_ingress(&token_ids, max_tokens, iterations)?;
        let cpu_profile =
            cpu_executor.profile_begin_token_ids_ingress(&token_ids, max_tokens, iterations)?;
        ggml_totals.record(&ggml_profile);
        cpu_totals.record(&cpu_profile);

        let speedup = if ggml_profile.avg_total_us() > 0.0 {
            cpu_profile.avg_total_us() / ggml_profile.avg_total_us()
        } else {
            0.0
        };
        let hottest_bucket = [
            (
                "embed_token_gather",
                GgmlHeadIngressProfile::avg_bucket_us(
                    ggml_profile.embed_token_gather_us,
                    ggml_profile.iterations,
                )
                .unwrap_or_default(),
            ),
            (
                "ple_token_gather",
                GgmlHeadIngressProfile::avg_bucket_us(
                    ggml_profile.ple_token_gather_us,
                    ggml_profile.iterations,
                )
                .unwrap_or_default(),
            ),
            (
                "ple_model_proj",
                GgmlHeadIngressProfile::avg_bucket_us(
                    ggml_profile.ple_model_proj_us,
                    ggml_profile.iterations,
                )
                .unwrap_or_default(),
            ),
            (
                "ple_normalize_combine",
                GgmlHeadIngressProfile::avg_bucket_us(
                    ggml_profile.ple_normalize_combine_us,
                    ggml_profile.iterations,
                )
                .unwrap_or_default(),
            ),
            (
                "prompt_aux_encode",
                GgmlHeadIngressProfile::avg_bucket_us(
                    ggml_profile.prompt_aux_encode_us,
                    ggml_profile.iterations,
                )
                .unwrap_or_default(),
            ),
            (
                "hidden_encode",
                GgmlHeadIngressProfile::avg_bucket_us(
                    ggml_profile.hidden_encode_us,
                    ggml_profile.iterations,
                )
                .unwrap_or_default(),
            ),
            (
                "payload_frame",
                GgmlHeadIngressProfile::avg_bucket_us(
                    ggml_profile.payload_frame_us,
                    ggml_profile.iterations,
                )
                .unwrap_or_default(),
            ),
            (
                "other",
                GgmlHeadIngressProfile::avg_bucket_us(
                    ggml_profile.other_us,
                    ggml_profile.iterations,
                )
                .unwrap_or_default(),
            ),
        ]
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(("n/a", 0.0));

        println!(
            "case         : {} tokens={} speedup={:.2}x hottest={} ({:.1} us)",
            case.name,
            token_ids.len(),
            speedup,
            hottest_bucket.0,
            hottest_bucket.1
        );
        print_profile("cpu-ref-worker", &cpu_profile);
        print_profile("ggml-worker", &ggml_profile);
        println!();
    }

    let cpu_avg = HeadIngressProfileTotals::avg(cpu_totals.total_us, cpu_totals.iterations);
    let ggml_avg = HeadIngressProfileTotals::avg(ggml_totals.total_us, ggml_totals.iterations);
    let hottest_bucket = ggml_totals.hottest_bucket();
    println!("=== Aggregate ===");
    println!("cpu-ref avg_total_us : {:.1}", cpu_avg);
    println!("ggml avg_total_us    : {:.1}", ggml_avg);
    println!(
        "speedup              : {:.2}x",
        if ggml_avg > 0.0 { cpu_avg / ggml_avg } else { 0.0 }
    );
    println!("hottest ggml bucket  : {} ({:.1} us)", hottest_bucket.0, hottest_bucket.1);

    Ok(())
}
