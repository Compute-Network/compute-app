use anyhow::{Result, bail};
use crate::{
    PackedTensorEntry, StageTensorStore, StageModelView, StageLayout,
    StageTensor, StageSample, StageForwardBackend, PayloadKind,
    LayerOperatorView, LayerExecutionProgram,
    quants,
};
use crate::real_math::{self, GemmaLayerConfig};
use crate::tokenizer::GemmaTokenizer;

pub struct RealGemmaBackend {
    index_path: std::path::PathBuf,
    layout: Option<StageLayout>,
    store: Option<StageTensorStore>,
    model_view: Option<StageModelView>,
    config: Option<GemmaLayerConfig>,
    rope_freqs: Option<Vec<f32>>,
    token_embd: Option<Vec<u8>>,
    token_embd_type: u32,
    vocab_size: usize,
    vocab_tokens: Option<Vec<String>>,
    tokenizer: Option<GemmaTokenizer>,
    ple_token_embd: Option<Vec<u8>>,
    ple_token_embd_type: u32,
    ple_model_proj: Option<(usize, usize, Vec<f32>)>,
    ple_proj_norm: Option<Vec<f32>>,
    ple_dim: usize,
    ple_num_layers: usize,
}

impl RealGemmaBackend {
    pub fn new(index_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            index_path: index_path.into(),
            layout: None,
            store: None,
            model_view: None,
            config: None,
            rope_freqs: None,
            token_embd: None,
            token_embd_type: 0,
            vocab_size: 0,
            vocab_tokens: None,
            tokenizer: None,
            ple_token_embd: None,
            ple_token_embd_type: 0,
            ple_model_proj: None,
            ple_proj_norm: None,
            ple_dim: 0,
            ple_num_layers: 0,
        }
    }

    pub fn load_vocab_json(&mut self, path: &std::path::Path) -> Result<()> {
        let data = std::fs::read(path)?;
        let tokens: Vec<String> = serde_json::from_slice(&data)?;
        self.vocab_tokens = Some(tokens);
        Ok(())
    }

    pub fn set_vocab(&mut self, tokens: Vec<String>) {
        self.vocab_tokens = Some(tokens);
    }

    pub fn load_tokenizer(
        &mut self,
        vocab_path: &std::path::Path,
        scores_path: Option<&std::path::Path>,
    ) -> Result<()> {
        let tok = GemmaTokenizer::load(vocab_path, scores_path)?;
        self.vocab_tokens = Some(tok.id_to_token().to_vec());
        self.tokenizer = Some(tok);
        Ok(())
    }

    fn tokenize_prompt(&self, prompt: &str) -> Vec<u32> {
        if let Some(tok) = &self.tokenizer {
            tok.encode_with_bos(prompt)
        } else {
            Self::simple_tokenize(prompt, self.vocab_size.max(1))
        }
    }

    fn decode_token(&self, id: u32) -> String {
        if let Some(vocab) = &self.vocab_tokens {
            if (id as usize) < vocab.len() {
                return vocab[id as usize].clone();
            }
        }
        if id < 128 {
            return (id as u8 as char).to_string();
        }
        format!("<{}>", id)
    }

    fn layout(&self) -> Result<&StageLayout> {
        self.layout.as_ref().ok_or_else(|| anyhow::anyhow!("No layout loaded"))
    }

    fn store(&self) -> Result<&StageTensorStore> {
        self.store.as_ref().ok_or_else(|| anyhow::anyhow!("No store loaded"))
    }

    fn model_view(&self) -> Result<&StageModelView> {
        self.model_view.as_ref().ok_or_else(|| anyhow::anyhow!("No model view loaded"))
    }

    fn config(&self) -> Result<&GemmaLayerConfig> {
        self.config.as_ref().ok_or_else(|| anyhow::anyhow!("No config loaded"))
    }

    fn decode_f32_vector(store: &StageTensorStore, entry: &PackedTensorEntry) -> Result<Vec<f32>> {
        if entry.ggml_type != quants::GGML_TYPE_F32 {
            bail!("Expected F32 tensor for {}, got type {}", entry.name, entry.ggml_type);
        }
        let bytes = store.read(&entry.name)?;
        quants::dequantize_f32_tensor(&bytes)
    }

    fn decode_matrix(store: &StageTensorStore, entry: &PackedTensorEntry) -> Result<(usize, usize, Vec<f32>)> {
        if entry.dimensions.len() != 2 {
            bail!("Expected 2D tensor for {}, got {}D", entry.name, entry.dimensions.len());
        }
        let in_dim = entry.dimensions[0] as usize;
        let out_dim = entry.dimensions[1] as usize;
        let bytes = store.read(&entry.name)?;
        let matrix = quants::dequantize_tensor(entry.ggml_type, &bytes)?;
        if matrix.len() != in_dim * out_dim {
            bail!(
                "Matrix {} decoded to {} elements but expected {}",
                entry.name, matrix.len(), in_dim * out_dim
            );
        }
        Ok((out_dim, in_dim, matrix))
    }

    fn embed_tokens(&self, token_ids: &[u32], hidden_dim: usize) -> Result<Vec<Vec<f32>>> {
        let embd_raw = self.token_embd.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No token embedding loaded"))?;
        let mut embeddings = Vec::with_capacity(token_ids.len());
        for &tid in token_ids {
            let row = quants::dequantize_row(self.token_embd_type, embd_raw, tid as usize, hidden_dim)?;
            let scale = (hidden_dim as f32).sqrt();
            let scaled: Vec<f32> = row.iter().map(|v| v * scale).collect();
            embeddings.push(scaled);
        }
        Ok(embeddings)
    }

    fn compute_ple_inputs(
        &self,
        token_ids: &[u32],
        inputs_embeds: &[Vec<f32>],
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let ple_raw = self.ple_token_embd.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No PLE token embedding loaded"))?;
        let ple_dim = self.ple_dim;
        let num_layers = self.ple_num_layers;
        if ple_dim == 0 || num_layers == 0 {
            bail!("PLE not configured (ple_dim={}, num_layers={})", ple_dim, num_layers);
        }
        let total_ple_dim = num_layers * ple_dim;
        let embed_scale = (ple_dim as f32).sqrt();
        let proj_scale = (inputs_embeds[0].len() as f32).powf(-0.5);
        let combine_scale = (2.0f32).powf(-0.5);

        let (proj_out, proj_in, proj_mat) = self.ple_model_proj.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No PLE model projection loaded"))?;
        let proj_norm = self.ple_proj_norm.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No PLE projection norm loaded"))?;

        let seq_len = token_ids.len();
        let mut result: Vec<Vec<Vec<f32>>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let token_row = quants::dequantize_row(
                self.ple_token_embd_type, ple_raw, token_ids[t] as usize, total_ple_dim,
            )?;
            let token_scaled: Vec<f32> = token_row.iter().map(|v| v * embed_scale).collect();

            let context_proj_raw = real_math::matmul(proj_mat, &inputs_embeds[t], *proj_out, *proj_in);
            let context_proj_scaled: Vec<f32> = context_proj_raw.iter().map(|v| v * proj_scale).collect();

            let mut per_layer_slices = Vec::with_capacity(num_layers);
            for layer_i in 0..num_layers {
                let start = layer_i * ple_dim;
                let end = start + ple_dim;
                let mut proj_slice = context_proj_scaled[start..end].to_vec();
                real_math::rms_norm_inplace(&mut proj_slice, proj_norm, 1e-6);
                let combined: Vec<f32> = proj_slice.iter()
                    .zip(&token_scaled[start..end])
                    .map(|(p, t)| (p + t) * combine_scale)
                    .collect();
                per_layer_slices.push(combined);
            }
            result.push(per_layer_slices);
        }
        Ok(result)
    }

    fn simple_tokenize(prompt: &str, vocab_size: usize) -> Vec<u32> {
        prompt.bytes().map(|b| (b as u32) % vocab_size as u32).collect()
    }

    fn layer_config(layer: &LayerOperatorView, fallback: &GemmaLayerConfig) -> GemmaLayerConfig {
        let hidden_dim = layer.attn_q.as_ref()
            .and_then(|e| e.dimensions.first().copied())
            .unwrap_or(fallback.hidden_dim as u64) as usize;
        let q_dim = layer.attn_q.as_ref()
            .and_then(|e| e.dimensions.get(1).copied())
            .unwrap_or((fallback.n_heads * fallback.head_dim) as u64) as usize;
        let k_dim = layer.attn_k.as_ref()
            .and_then(|e| e.dimensions.get(1).copied())
            .unwrap_or((fallback.n_kv_heads * fallback.head_dim) as u64) as usize;
        let ffn_dim = layer.ffn_up.as_ref()
            .and_then(|e| e.dimensions.get(1).copied())
            .unwrap_or(fallback.ffn_dim as u64) as usize;
        GemmaLayerConfig::from_dims(hidden_dim, q_dim, k_dim, ffn_dim)
    }

    fn forward_layer_seq(
        &self,
        states: &mut [Vec<f32>],
        layer: &LayerOperatorView,
        _program: &LayerExecutionProgram,
        position_offset: u32,
        ple_inputs: Option<&[Vec<f32>]>,
    ) -> Result<()> {
        let store = self.store()?;
        let base_config = self.config()?;
        let config = Self::layer_config(layer, base_config);
        let seq_len = states.len();

        let attn_norm_weight = layer.attn_norm.as_ref()
            .map(|e| Self::decode_f32_vector(store, e)).transpose()?;
        let q_matrix = layer.attn_q.as_ref()
            .map(|e| Self::decode_matrix(store, e)).transpose()?;
        let k_matrix = layer.attn_k.as_ref()
            .map(|e| Self::decode_matrix(store, e)).transpose()?;
        let v_matrix = layer.attn_v.as_ref()
            .map(|e| Self::decode_matrix(store, e)).transpose()?;
        let q_norm_weight = layer.attn_q_norm.as_ref()
            .map(|e| Self::decode_f32_vector(store, e)).transpose()?;
        let k_norm_weight = layer.attn_k_norm.as_ref()
            .map(|e| Self::decode_f32_vector(store, e)).transpose()?;
        let o_matrix = layer.attn_output.as_ref()
            .map(|e| Self::decode_matrix(store, e)).transpose()?;
        let post_attn_norm_weight = layer.post_attention_norm.as_ref()
            .map(|e| Self::decode_f32_vector(store, e)).transpose()?;
        let ffn_norm_weight = layer.ffn_norm.as_ref()
            .map(|e| Self::decode_f32_vector(store, e)).transpose()?;
        let gate_matrix = layer.ffn_gate.as_ref()
            .map(|e| Self::decode_matrix(store, e)).transpose()?;
        let up_matrix = layer.ffn_up.as_ref()
            .map(|e| Self::decode_matrix(store, e)).transpose()?;
        let down_matrix = layer.ffn_down.as_ref()
            .map(|e| Self::decode_matrix(store, e)).transpose()?;
        let post_ffn_norm_weight = layer.post_ffw_norm.as_ref()
            .map(|e| Self::decode_f32_vector(store, e)).transpose()?;
        let layer_scale = layer.layer_output_scale.as_ref().map(|e| {
            let bytes = store.read(&e.name).ok()?;
            if bytes.len() >= 4 {
                Some(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            } else {
                None
            }
        }).flatten();

        let inp_gate_matrix = layer.inp_gate.as_ref()
            .map(|e| Self::decode_matrix(store, e)).transpose()?;
        let proj_matrix = layer.proj.as_ref()
            .map(|e| Self::decode_matrix(store, e)).transpose()?;
        let post_norm_weight = layer.post_norm.as_ref()
            .map(|e| Self::decode_f32_vector(store, e)).transpose()?;
        let has_ple = inp_gate_matrix.is_some() && proj_matrix.is_some() && ple_inputs.is_some();

        let has_attn = q_matrix.is_some() && k_matrix.is_some() && v_matrix.is_some();
        let has_ffn = gate_matrix.is_some() && up_matrix.is_some() && down_matrix.is_some();

        let mut k_cache: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let mut v_cache: Vec<Vec<f32>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let residual_pre_attn = states[t].clone();

            if let Some(ref w) = attn_norm_weight {
                real_math::rms_norm_inplace(&mut states[t], w, config.eps);
            }

            if has_attn {
                let (q_out, q_in, q_mat) = q_matrix.as_ref().unwrap();
                let (k_out, k_in, k_mat) = k_matrix.as_ref().unwrap();
                let (v_out, v_in, v_mat) = v_matrix.as_ref().unwrap();

                let mut q = real_math::matmul(q_mat, &states[t], *q_out, *q_in);
                let mut k = real_math::matmul(k_mat, &states[t], *k_out, *k_in);
                let v = real_math::matmul(v_mat, &states[t], *v_out, *v_in);


                if let Some(ref w) = q_norm_weight {
                    real_math::per_head_rms_norm(&mut q, w, config.n_heads, config.head_dim);
                }
                if let Some(ref w) = k_norm_weight {
                    real_math::per_head_rms_norm(&mut k, w, config.n_kv_heads, config.head_dim);
                }

                if let Some(ref freqs) = self.rope_freqs {
                    real_math::rope_apply(
                        &mut q, &mut k, freqs, position_offset + t as u32,
                        config.n_heads, config.n_kv_heads, config.head_dim,
                    );
                }

                k_cache.push(k);
                v_cache.push(v);

                let attn_out = real_math::gqa_attention_seq(
                    &q, &k_cache, &v_cache,
                    config.n_heads, config.n_kv_heads, config.head_dim,
                );

                let attn_proj = if let Some((o_out, o_in, o_mat)) = o_matrix.as_ref() {
                    real_math::matmul(o_mat, &attn_out, *o_out, *o_in)
                } else {
                    attn_out
                };

                if let Some(ref w) = post_attn_norm_weight {
                    let normed = real_math::rms_norm(&attn_proj, w, config.eps);
                    states[t] = real_math::vec_add(&residual_pre_attn, &normed);
                } else {
                    states[t] = real_math::vec_add(&residual_pre_attn, &attn_proj);
                }
            } else {
                states[t] = residual_pre_attn;
            }

            let residual_pre_ffn = states[t].clone();

            if let Some(ref w) = ffn_norm_weight {
                real_math::rms_norm_inplace(&mut states[t], w, config.eps);
            }

            if has_ffn {
                let (g_out, g_in, g_mat) = gate_matrix.as_ref().unwrap();
                let (u_out, u_in, u_mat) = up_matrix.as_ref().unwrap();
                let (d_out, d_in, d_mat) = down_matrix.as_ref().unwrap();

                let gate = real_math::matmul(g_mat, &states[t], *g_out, *g_in);
                let up = real_math::matmul(u_mat, &states[t], *u_out, *u_in);
                let gate_activated = real_math::silu(&gate);
                let ffn_hidden = real_math::vec_mul(&gate_activated, &up);
                let ffn_out = real_math::matmul(d_mat, &ffn_hidden, *d_out, *d_in);

                if let Some(ref w) = post_ffn_norm_weight {
                    let normed = real_math::rms_norm(&ffn_out, w, config.eps);
                    states[t] = real_math::vec_add(&residual_pre_ffn, &normed);
                } else {
                    states[t] = real_math::vec_add(&residual_pre_ffn, &ffn_out);
                }
            }

            if has_ple {
                let residual_pre_ple = states[t].clone();
                let gated = real_math::matmul(
                    &inp_gate_matrix.as_ref().unwrap().2,
                    &states[t],
                    inp_gate_matrix.as_ref().unwrap().0,
                    inp_gate_matrix.as_ref().unwrap().1,
                );
                let activated = real_math::gelu_pytorch_tanh(&gated);
                let ple_input = &ple_inputs.unwrap()[t];
                let modulated = real_math::vec_mul(&activated, ple_input);
                let projected = real_math::matmul(
                    &proj_matrix.as_ref().unwrap().2,
                    &modulated,
                    proj_matrix.as_ref().unwrap().0,
                    proj_matrix.as_ref().unwrap().1,
                );
                if let Some(ref w) = post_norm_weight {
                    let normed = real_math::rms_norm(&projected, w, config.eps);
                    states[t] = real_math::vec_add(&residual_pre_ple, &normed);
                } else {
                    states[t] = real_math::vec_add(&residual_pre_ple, &projected);
                }
            }

            if let Some(scale) = layer_scale {
                for v in states[t].iter_mut() {
                    *v *= scale;
                }
            }
        }

        Ok(())
    }

    fn forward_layer(
        &self,
        state: &mut Vec<f32>,
        layer: &LayerOperatorView,
        program: &LayerExecutionProgram,
        position: u32,
        ple_input: Option<&[Vec<f32>]>,
    ) -> Result<()> {
        let mut states = vec![state.clone()];
        self.forward_layer_seq(&mut states, layer, program, position, ple_input)?;
        *state = states.into_iter().next().unwrap();
        Ok(())
    }

    fn encode_hidden_state(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn decode_hidden_state(bytes: &[u8], hidden_dim: usize) -> Result<Vec<f32>> {
        if bytes.len() != hidden_dim * 4 {
            bail!(
                "Hidden-state byte length {} does not match hidden_dim * 4 = {}",
                bytes.len(), hidden_dim * 4
            );
        }
        let mut values = Vec::with_capacity(hidden_dim);
        for chunk in bytes.chunks_exact(4) {
            values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(values)
    }
}

impl StageForwardBackend for RealGemmaBackend {
    fn load_layout(&mut self, layout: StageLayout) -> Result<()> {
        let store = StageTensorStore::load(&self.index_path)?;
        store.validate_offsets()?;

        let model_view = store.model_view();

        let first_layer = model_view.operator_layers.first();

        let hidden_dim = first_layer
            .and_then(|l| l.attn_q.as_ref())
            .and_then(|e| e.dimensions.first().copied())
            .unwrap_or(2560) as usize;
        let q_dim = first_layer
            .and_then(|l| l.attn_q.as_ref())
            .and_then(|e| e.dimensions.get(1).copied())
            .unwrap_or(2048) as usize;
        let k_dim = first_layer
            .and_then(|l| l.attn_k.as_ref())
            .and_then(|e| e.dimensions.get(1).copied())
            .unwrap_or(512) as usize;
        let ffn_dim = first_layer
            .and_then(|l| l.ffn_up.as_ref())
            .and_then(|e| e.dimensions.get(1).copied())
            .unwrap_or(10240) as usize;

        let config = GemmaLayerConfig::from_dims(hidden_dim, q_dim, k_dim, ffn_dim);

        let rope_freqs = model_view.positional.iter()
            .find(|e| e.name == "rope_freqs.weight")
            .map(|e| Self::decode_f32_vector(&store, e))
            .transpose()?;

        let (token_embd, token_embd_type, vocab_size) = {
            let entry = model_view.prompt_ingress.iter()
                .chain(model_view.shared_auxiliary.iter())
                .chain(model_view.tail_only.iter())
                .find(|e| e.name == "token_embd.weight")
                .or_else(|| store.entry("token_embd.weight"));
            if let Some(entry) = entry {
                let raw = store.read(&entry.name)?;
                let vocab = if entry.dimensions.len() == 2 {
                    entry.dimensions[1] as usize
                } else {
                    0
                };
                (Some(raw), entry.ggml_type, vocab)
            } else {
                (None, 0, 0)
            }
        };

        let ple_token_entry = store.entry("per_layer_token_embd.weight");
        let (ple_token_embd, ple_token_embd_type) = if let Some(entry) = ple_token_entry {
            let raw = store.read(&entry.name)?;
            (Some(raw), entry.ggml_type)
        } else {
            (None, 0)
        };

        let ple_model_proj = store.entry("per_layer_model_proj.weight")
            .map(|e| Self::decode_matrix(&store, e))
            .transpose()?;

        let ple_proj_norm = store.entry("per_layer_proj_norm.weight")
            .map(|e| Self::decode_f32_vector(&store, e))
            .transpose()?;

        let ple_dim = ple_proj_norm.as_ref().map(|w| w.len()).unwrap_or(0);
        let ple_num_layers = if ple_dim > 0 {
            ple_model_proj.as_ref()
                .map(|(out, _, _)| *out / ple_dim)
                .unwrap_or(0)
        } else {
            0
        };

        self.config = Some(config);
        self.rope_freqs = rope_freqs;
        self.token_embd = token_embd;
        self.token_embd_type = token_embd_type;
        self.vocab_size = vocab_size;
        self.ple_token_embd = ple_token_embd;
        self.ple_token_embd_type = ple_token_embd_type;
        self.ple_model_proj = ple_model_proj;
        self.ple_proj_norm = ple_proj_norm;
        self.ple_dim = ple_dim;
        self.ple_num_layers = ple_num_layers;
        self.layout = Some(layout);
        self.model_view = Some(model_view);
        self.store = Some(store);
        Ok(())
    }

    fn begin_prompt(
        &self,
        request_id: &str,
        prompt: &str,
        max_tokens: Option<u32>,
        _hidden_dim_hint: usize,
    ) -> Result<StageTensor> {
        let layout = self.layout()?;
        let model_view = self.model_view()?;
        let config = self.config()?;

        if !layout.is_head {
            bail!("Only the head stage may accept prompt ingress");
        }

        let hidden_dim = config.hidden_dim;

        let token_ids = self.tokenize_prompt(prompt);
        if token_ids.is_empty() {
            bail!("Empty prompt");
        }

        let mut states = self.embed_tokens(&token_ids, hidden_dim)?;

        let ple_all = if self.ple_dim > 0 {
            Some(self.compute_ple_inputs(&token_ids, &states)?)
        } else {
            None
        };

        let start_layer = layout.start_layer as usize;
        for (idx, (layer, program)) in model_view.operator_layers.iter()
            .zip(model_view.execution_programs.iter())
            .enumerate()
        {
            let global_layer = start_layer + idx;
            let ple_for_layer: Option<Vec<Vec<f32>>> = ple_all.as_ref().map(|all| {
                all.iter().map(|token_layers| {
                    if global_layer < token_layers.len() {
                        token_layers[global_layer].clone()
                    } else {
                        vec![0.0; self.ple_dim]
                    }
                }).collect()
            });
            let ple_ref = ple_for_layer.as_deref();
            self.forward_layer_seq(&mut states, layer, program, 0, ple_ref)?;
        }

        let state = states.last().unwrap();
        let bytes = Self::encode_hidden_state(state);

        Ok(StageTensor {
            request_id: request_id.to_string(),
            kind: PayloadKind::HiddenState,
            stage_trace: vec![layout.stage_id.clone()],
            hidden_dim,
            bytes,
            prompt_text: Some(prompt.to_string()),
            max_tokens,
            continuation: None,
            transient: None,
            carry: None,
        })
    }

    fn continue_forward(&self, input: StageTensor) -> Result<StageTensor> {
        let layout = self.layout()?;
        let model_view = self.model_view()?;

        if input.kind != PayloadKind::HiddenState {
            bail!("Stage forward requires hidden-state payloads");
        }
        if layout.is_head {
            bail!("Head stage should use begin_prompt, not continue_forward");
        }

        let mut state = Self::decode_hidden_state(&input.bytes, input.hidden_dim)?;

        let ple_all = if self.ple_dim > 0 && self.token_embd.is_some() {
            if let Some(ref prompt) = input.prompt_text {
                let token_ids = self.tokenize_prompt(prompt);
                if !token_ids.is_empty() {
                    let embeds = self.embed_tokens(&token_ids, input.hidden_dim)?;
                    let ple = self.compute_ple_inputs(&token_ids, &embeds)?;
                    let last_token_ple = ple.last().cloned();
                    last_token_ple
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let start_layer = layout.start_layer;
        for (idx, (layer, program)) in model_view.operator_layers.iter()
            .zip(model_view.execution_programs.iter())
            .enumerate()
        {
            let global_layer = start_layer as usize + idx;
            let ple_for_layer: Option<Vec<Vec<f32>>> = ple_all.as_ref().map(|token_layers| {
                let ple_vec = if global_layer < token_layers.len() {
                    token_layers[global_layer].clone()
                } else {
                    vec![0.0; self.ple_dim]
                };
                vec![ple_vec]
            });
            let ple_ref = ple_for_layer.as_deref();
            self.forward_layer(&mut state, layer, program, start_layer + idx as u32, ple_ref)?;
        }

        let mut stage_trace = input.stage_trace;
        stage_trace.push(layout.stage_id.clone());

        let bytes = Self::encode_hidden_state(&state);

        Ok(StageTensor {
            request_id: input.request_id,
            kind: PayloadKind::HiddenState,
            stage_trace,
            hidden_dim: input.hidden_dim,
            bytes,
            prompt_text: input.prompt_text,
            max_tokens: input.max_tokens,
            continuation: None,
            transient: None,
            carry: None,
        })
    }

    fn sample_tail(&self, input: StageTensor) -> Result<StageSample> {
        let layout = self.layout()?;
        let store = self.store()?;
        let model_view = self.model_view()?;
        let config = self.config()?;

        if !layout.is_tail {
            bail!("Only the tail stage may sample output");
        }
        if input.kind != PayloadKind::HiddenState {
            bail!("Tail sampling requires hidden-state payloads");
        }

        let mut state = Self::decode_hidden_state(&input.bytes, input.hidden_dim)?;

        if let Some(entry) = model_view.tail_only.iter().find(|e| e.name == "output_norm.weight") {
            let w = Self::decode_f32_vector(store, entry)?;
            real_math::rms_norm_inplace(&mut state, &w, config.eps);
        }

        let _token_embd_entry = model_view.prompt_ingress.iter()
            .find(|e| e.name == "token_embd.weight");

        let text = if let Some(entry) = store.entry("token_embd.weight") {
            let raw = store.read("token_embd.weight")?;
            let hidden_dim = config.hidden_dim;
            let vocab_size = if entry.dimensions.len() == 2 {
                entry.dimensions[1] as usize
            } else {
                0
            };

            if vocab_size > 0 {
                let max_tokens_count = input.max_tokens.unwrap_or(1).max(1) as usize;
                let mut generated = Vec::new();

                for _ in 0..max_tokens_count {
                    let mut best_score = f32::NEG_INFINITY;
                    let mut best_id = 0usize;

                    let batch = 1024;
                    for batch_start in (0..vocab_size).step_by(batch) {
                        let batch_end = (batch_start + batch).min(vocab_size);
                        for row in batch_start..batch_end {
                            let emb = quants::dequantize_row(
                                entry.ggml_type, &raw, row, hidden_dim,
                            )?;
                            let score: f32 = emb.iter()
                                .zip(state.iter())
                                .map(|(e, s)| e * s)
                                .sum();
                            if score > best_score {
                                best_score = score;
                                best_id = row;
                            }
                        }
                    }
                    generated.push(best_id as u32);
                }

                if let Some(tok) = &self.tokenizer {
                    tok.decode_ids(&generated)
                } else {
                    generated.iter()
                        .map(|&id| self.decode_token(id))
                        .collect::<Vec<_>>()
                        .join("")
                }
            } else {
                "?".to_string()
            }
        } else {
            "?".to_string()
        };

        Ok(StageSample {
            request_id: input.request_id,
            model_id: layout.model_id.clone(),
            text,
            completion_tokens: 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PackedStageIndex, PackedTensorEntry};
    use std::fs;
    use tempfile::tempdir;

    fn write_f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn real_backend_loads_and_runs_minimal_stage() {
        let temp = tempdir().unwrap();
        let index_path = temp.path().join("stage-1-required.index.json");
        let pack_path = temp.path().join("stage-1-required.pack");

        let hidden_dim = 4;
        let vocab_size = 8;
        let n_heads = 2;
        let head_dim = 2;
        let n_kv_heads = 1;
        let ffn_dim = 8;

        let mut pack_data = Vec::new();
        let mut tensors = Vec::new();
        let mut offset = 0u64;

        let add_tensor = |name: &str, dims: Vec<u64>, data: &[f32],
                               pack: &mut Vec<u8>, tensors: &mut Vec<PackedTensorEntry>, off: &mut u64| {
            let bytes = write_f32_bytes(data);
            let entry = PackedTensorEntry {
                name: name.to_string(),
                pack_offset: *off,
                byte_len: bytes.len() as u64,
                source_file_offset: 0,
                dimensions: dims,
                ggml_type: quants::GGML_TYPE_F32,
            };
            pack.extend_from_slice(&bytes);
            *off += bytes.len() as u64;
            tensors.push(entry);
        };

        let embd: Vec<f32> = (0..hidden_dim * vocab_size)
            .map(|i| (i as f32 * 0.1) - 0.5)
            .collect();
        add_tensor("token_embd.weight",
            vec![hidden_dim as u64, vocab_size as u64],
            &embd, &mut pack_data, &mut tensors, &mut offset);

        let rope: Vec<f32> = vec![1.0; head_dim / 2];
        add_tensor("rope_freqs.weight",
            vec![rope.len() as u64],
            &rope, &mut pack_data, &mut tensors, &mut offset);

        let norm_w = vec![0.0f32; hidden_dim];
        add_tensor("blk.0.attn_norm.weight",
            vec![hidden_dim as u64],
            &norm_w, &mut pack_data, &mut tensors, &mut offset);

        let identity_4x4: Vec<f32> = (0..hidden_dim).flat_map(|r| {
            (0..hidden_dim).map(move |c| if r == c { 1.0 } else { 0.0 })
        }).collect();

        add_tensor("blk.0.attn_q.weight",
            vec![hidden_dim as u64, (n_heads * head_dim) as u64],
            &identity_4x4, &mut pack_data, &mut tensors, &mut offset);

        let k_weights: Vec<f32> = (0..hidden_dim * n_kv_heads * head_dim)
            .map(|i| if i % (hidden_dim + 1) == 0 { 1.0 } else { 0.0 })
            .collect();
        add_tensor("blk.0.attn_k.weight",
            vec![hidden_dim as u64, (n_kv_heads * head_dim) as u64],
            &k_weights, &mut pack_data, &mut tensors, &mut offset);

        add_tensor("blk.0.attn_v.weight",
            vec![hidden_dim as u64, (n_kv_heads * head_dim) as u64],
            &k_weights, &mut pack_data, &mut tensors, &mut offset);

        let q_norm = vec![0.0f32; head_dim];
        add_tensor("blk.0.attn_q_norm.weight",
            vec![head_dim as u64],
            &q_norm, &mut pack_data, &mut tensors, &mut offset);
        add_tensor("blk.0.attn_k_norm.weight",
            vec![head_dim as u64],
            &q_norm, &mut pack_data, &mut tensors, &mut offset);

        add_tensor("blk.0.attn_output.weight",
            vec![(n_heads * head_dim) as u64, hidden_dim as u64],
            &identity_4x4, &mut pack_data, &mut tensors, &mut offset);

        let post_attn_norm = vec![0.0f32; hidden_dim];
        add_tensor("blk.0.post_attention_norm.weight",
            vec![hidden_dim as u64],
            &post_attn_norm, &mut pack_data, &mut tensors, &mut offset);
        add_tensor("blk.0.ffn_norm.weight",
            vec![hidden_dim as u64],
            &post_attn_norm, &mut pack_data, &mut tensors, &mut offset);

        let ffn_gate: Vec<f32> = (0..hidden_dim * ffn_dim)
            .map(|i| if i % (hidden_dim + 1) == 0 { 0.5 } else { 0.0 })
            .collect();
        add_tensor("blk.0.ffn_gate.weight",
            vec![hidden_dim as u64, ffn_dim as u64],
            &ffn_gate, &mut pack_data, &mut tensors, &mut offset);
        add_tensor("blk.0.ffn_up.weight",
            vec![hidden_dim as u64, ffn_dim as u64],
            &ffn_gate, &mut pack_data, &mut tensors, &mut offset);

        let ffn_down: Vec<f32> = (0..ffn_dim * hidden_dim)
            .map(|i| if i % (ffn_dim + 1) == 0 { 0.5 } else { 0.0 })
            .collect();
        add_tensor("blk.0.ffn_down.weight",
            vec![ffn_dim as u64, hidden_dim as u64],
            &ffn_down, &mut pack_data, &mut tensors, &mut offset);

        add_tensor("blk.0.post_ffw_norm.weight",
            vec![hidden_dim as u64],
            &post_attn_norm, &mut pack_data, &mut tensors, &mut offset);

        let scale = vec![1.0f32];
        add_tensor("blk.0.layer_output_scale.weight",
            vec![1],
            &scale, &mut pack_data, &mut tensors, &mut offset);

        fs::write(&pack_path, &pack_data).unwrap();
        fs::write(&index_path, serde_json::to_vec_pretty(&PackedStageIndex {
            model_name: "test-gemma".into(),
            architecture: "gemma4".into(),
            stage_index: 0,
            role: "head".into(),
            total_bytes: offset,
            tensor_count: tensors.len(),
            tensors,
        }).unwrap()).unwrap();

        let mut backend = RealGemmaBackend::new(&index_path);
        backend.load_layout(StageLayout {
            model_id: "test-gemma".into(),
            stage_id: "stage-0".into(),
            start_layer: 0,
            end_layer: 0,
            is_head: true,
            is_tail: false,
        }).unwrap();

        let tensor = backend.begin_prompt("test-req", "hi", Some(1), 0).unwrap();
        assert_eq!(tensor.kind, PayloadKind::HiddenState);
        assert_eq!(tensor.hidden_dim, hidden_dim);
        assert_eq!(tensor.bytes.len(), hidden_dim * 4);

        let state = RealGemmaBackend::decode_hidden_state(&tensor.bytes, hidden_dim).unwrap();
        assert!(state.iter().all(|v| v.is_finite()), "State contains non-finite values: {:?}", state);
    }

    struct TestStageBuilder {
        pack_data: Vec<u8>,
        tensors: Vec<PackedTensorEntry>,
        offset: u64,
    }

    impl TestStageBuilder {
        fn new() -> Self {
            Self { pack_data: Vec::new(), tensors: Vec::new(), offset: 0 }
        }

        fn add_f32(&mut self, name: &str, dims: Vec<u64>, data: &[f32]) {
            let bytes = write_f32_bytes(data);
            self.tensors.push(PackedTensorEntry {
                name: name.to_string(),
                pack_offset: self.offset,
                byte_len: bytes.len() as u64,
                source_file_offset: 0,
                dimensions: dims,
                ggml_type: quants::GGML_TYPE_F32,
            });
            self.pack_data.extend_from_slice(&bytes);
            self.offset += bytes.len() as u64;
        }

        fn write(self, dir: &std::path::Path, stage_name: &str, role: &str, stage_index: u32)
            -> std::path::PathBuf
        {
            let index_path = dir.join(format!("{}-required.index.json", stage_name));
            let pack_path = dir.join(format!("{}-required.pack", stage_name));
            fs::write(&pack_path, &self.pack_data).unwrap();
            fs::write(&index_path, serde_json::to_vec_pretty(&PackedStageIndex {
                model_name: "test-gemma".into(),
                architecture: "gemma4".into(),
                stage_index,
                role: role.into(),
                total_bytes: self.offset,
                tensor_count: self.tensors.len(),
                tensors: self.tensors,
            }).unwrap()).unwrap();
            index_path
        }
    }

    fn add_layer_tensors(
        b: &mut TestStageBuilder, layer: usize,
        hidden_dim: usize, n_heads: usize, head_dim: usize,
        n_kv_heads: usize, ffn_dim: usize,
    ) {
        let q_dim = n_heads * head_dim;
        let k_dim = n_kv_heads * head_dim;
        let prefix = format!("blk.{}", layer);

        let norm_w = vec![0.0f32; hidden_dim];
        b.add_f32(&format!("{prefix}.attn_norm.weight"), vec![hidden_dim as u64], &norm_w);

        let identity: Vec<f32> = (0..hidden_dim * q_dim)
            .map(|i| if i / q_dim == i % q_dim { 0.1 } else { 0.0 })
            .collect();
        b.add_f32(&format!("{prefix}.attn_q.weight"),
            vec![hidden_dim as u64, q_dim as u64], &identity[..hidden_dim * q_dim]);

        let k_w: Vec<f32> = (0..hidden_dim * k_dim)
            .map(|i| if i / k_dim == i % k_dim { 0.1 } else { 0.0 })
            .collect();
        b.add_f32(&format!("{prefix}.attn_k.weight"),
            vec![hidden_dim as u64, k_dim as u64], &k_w);
        b.add_f32(&format!("{prefix}.attn_v.weight"),
            vec![hidden_dim as u64, k_dim as u64], &k_w);

        let q_norm = vec![0.0f32; head_dim];
        b.add_f32(&format!("{prefix}.attn_q_norm.weight"), vec![head_dim as u64], &q_norm);
        b.add_f32(&format!("{prefix}.attn_k_norm.weight"), vec![head_dim as u64], &q_norm);

        let out_w: Vec<f32> = (0..q_dim * hidden_dim)
            .map(|i| if i / hidden_dim == i % hidden_dim { 0.1 } else { 0.0 })
            .collect();
        b.add_f32(&format!("{prefix}.attn_output.weight"),
            vec![q_dim as u64, hidden_dim as u64], &out_w);

        b.add_f32(&format!("{prefix}.post_attention_norm.weight"), vec![hidden_dim as u64], &norm_w);
        b.add_f32(&format!("{prefix}.ffn_norm.weight"), vec![hidden_dim as u64], &norm_w);

        let ffn_w: Vec<f32> = (0..hidden_dim * ffn_dim)
            .map(|i| if i / ffn_dim == i % ffn_dim { 0.1 } else { 0.0 })
            .collect();
        b.add_f32(&format!("{prefix}.ffn_gate.weight"),
            vec![hidden_dim as u64, ffn_dim as u64], &ffn_w);
        b.add_f32(&format!("{prefix}.ffn_up.weight"),
            vec![hidden_dim as u64, ffn_dim as u64], &ffn_w);

        let ffn_down: Vec<f32> = (0..ffn_dim * hidden_dim)
            .map(|i| if i / hidden_dim == i % hidden_dim { 0.1 } else { 0.0 })
            .collect();
        b.add_f32(&format!("{prefix}.ffn_down.weight"),
            vec![ffn_dim as u64, hidden_dim as u64], &ffn_down);

        b.add_f32(&format!("{prefix}.post_ffw_norm.weight"), vec![hidden_dim as u64], &norm_w);
        b.add_f32(&format!("{prefix}.layer_output_scale.weight"), vec![1], &[1.0]);
    }

    #[test]
    fn real_two_stage_roundtrip_produces_finite_output() {
        let temp = tempdir().unwrap();
        let hidden_dim = 8;
        let n_heads = 2;
        let head_dim = 4;
        let n_kv_heads = 1;
        let ffn_dim = 16;
        let vocab_size = 32;

        let mut head_builder = TestStageBuilder::new();
        let embd: Vec<f32> = (0..hidden_dim * vocab_size)
            .map(|i| (i as f32 * 0.37).sin() * 0.5)
            .collect();
        head_builder.add_f32("token_embd.weight",
            vec![hidden_dim as u64, vocab_size as u64], &embd);
        let rope = vec![1.0f32; head_dim / 2];
        head_builder.add_f32("rope_freqs.weight", vec![rope.len() as u64], &rope);
        add_layer_tensors(&mut head_builder, 0, hidden_dim, n_heads, head_dim, n_kv_heads, ffn_dim);
        add_layer_tensors(&mut head_builder, 1, hidden_dim, n_heads, head_dim, n_kv_heads, ffn_dim);
        let head_path = head_builder.write(temp.path(), "stage-0", "head", 0);

        let mut tail_builder = TestStageBuilder::new();
        let tail_rope = vec![1.0f32; head_dim / 2];
        tail_builder.add_f32("rope_freqs.weight", vec![tail_rope.len() as u64], &tail_rope);
        add_layer_tensors(&mut tail_builder, 2, hidden_dim, n_heads, head_dim, n_kv_heads, ffn_dim);
        add_layer_tensors(&mut tail_builder, 3, hidden_dim, n_heads, head_dim, n_kv_heads, ffn_dim);
        let norm_w = vec![0.0f32; hidden_dim];
        tail_builder.add_f32("output_norm.weight", vec![hidden_dim as u64], &norm_w);
        let tail_path = tail_builder.write(temp.path(), "stage-1", "tail", 1);

        let mut head = RealGemmaBackend::new(&head_path);
        head.load_layout(StageLayout {
            model_id: "test-gemma".into(),
            stage_id: "stage-0".into(),
            start_layer: 0,
            end_layer: 1,
            is_head: true,
            is_tail: false,
        }).unwrap();

        let mut tail = RealGemmaBackend::new(&tail_path);
        tail.load_layout(StageLayout {
            model_id: "test-gemma".into(),
            stage_id: "stage-1".into(),
            start_layer: 2,
            end_layer: 3,
            is_head: false,
            is_tail: true,
        }).unwrap();

        let head_output = head.begin_prompt("req-1", "hello world", Some(1), 0).unwrap();
        assert_eq!(head_output.kind, PayloadKind::HiddenState);
        assert_eq!(head_output.hidden_dim, hidden_dim);

        let head_state = RealGemmaBackend::decode_hidden_state(&head_output.bytes, hidden_dim).unwrap();
        assert!(head_state.iter().all(|v| v.is_finite()),
            "Head output has non-finite values: {:?}", head_state);

        let tail_input = tail.continue_forward(head_output).unwrap();
        assert_eq!(tail_input.kind, PayloadKind::HiddenState);
        assert_eq!(tail_input.hidden_dim, hidden_dim);
        assert_eq!(tail_input.stage_trace, vec!["stage-0", "stage-1"]);

        let tail_state = RealGemmaBackend::decode_hidden_state(&tail_input.bytes, hidden_dim).unwrap();
        assert!(tail_state.iter().all(|v| v.is_finite()),
            "Tail output has non-finite values: {:?}", tail_state);

        assert_ne!(head_state, tail_state, "Tail stage should transform the hidden state");
    }
}
