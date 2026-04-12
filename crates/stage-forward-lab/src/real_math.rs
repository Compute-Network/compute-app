use anyhow::{Result, bail};

const GEMMA_RMS_NORM_EPS: f32 = 1e-6;

pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    let wlen = weight.len();
    x.iter()
        .enumerate()
        .map(|(i, v)| v * scale * (1.0 + weight[i % wlen]))
        .collect()
}

pub fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    let wlen = weight.len();
    for (i, v) in x.iter_mut().enumerate() {
        *v = *v * scale * (1.0 + weight[i % wlen]);
    }
}

pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v / (1.0 + (-v).exp())).collect()
}

pub fn gelu_pytorch_tanh(x: &[f32]) -> Vec<f32> {
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    x.iter().map(|&v| {
        let inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
        0.5 * v * (1.0 + inner.tanh())
    }).collect()
}

pub fn softmax(x: &[f32]) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / x.len() as f32; x.len()];
    }
    exps.iter().map(|v| v / sum).collect()
}

pub fn matmul(matrix: &[f32], input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    assert!(
        matrix.len() >= out_dim * in_dim,
        "matmul: matrix.len()={} but out_dim*in_dim={} (out={}, in={})",
        matrix.len(), out_dim * in_dim, out_dim, in_dim
    );
    assert!(
        input.len() >= in_dim,
        "matmul: input.len()={} but in_dim={} (out_dim={})",
        input.len(), in_dim, out_dim
    );
    let mut output = vec![0.0f32; out_dim];
    for row in 0..out_dim {
        let mut acc = 0.0f32;
        let row_offset = row * in_dim;
        for col in 0..in_dim {
            acc += matrix[row_offset + col] * input[col];
        }
        output[row] = acc;
    }
    output
}

pub fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

pub fn vec_add_inplace(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += y;
    }
}

pub fn vec_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

pub fn vec_scale(x: &[f32], s: f32) -> Vec<f32> {
    x.iter().map(|v| v * s).collect()
}

pub fn rope_apply(
    q: &mut [f32],
    k: &mut [f32],
    freqs: &[f32],
    position: u32,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    let pos = position as f32;
    rope_apply_heads(q, freqs, pos, n_heads, head_dim);
    rope_apply_heads(k, freqs, pos, n_kv_heads, head_dim);
}

fn rope_apply_heads(x: &mut [f32], freqs: &[f32], pos: f32, n_heads: usize, head_dim: usize) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let freq = if i < freqs.len() {
                freqs[i]
            } else {
                1.0 / (10000.0f32).powf(2.0 * i as f32 / head_dim as f32)
            };
            let theta = pos * freq;
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let x0 = x[base + i];
            let x1 = x[base + half + i];
            x[base + i] = x0 * cos_t - x1 * sin_t;
            x[base + half + i] = x0 * sin_t + x1 * cos_t;
        }
    }
}

pub fn gqa_attention(
    _q: &[f32],
    _k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let groups = n_heads / n_kv_heads;
    let mut output = vec![0.0f32; n_heads * head_dim];

    for h in 0..n_heads {
        let kv_h = h / groups;
        let v_offset = kv_h * head_dim;
        let out_offset = h * head_dim;
        for d in 0..head_dim {
            output[out_offset + d] = v[v_offset + d];
        }
    }
    output
}

pub fn gqa_attention_seq(
    q: &[f32],
    k_cache: &[Vec<f32>],
    v_cache: &[Vec<f32>],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let groups = n_heads / n_kv_heads;
    let seq_len = k_cache.len();
    let mut output = vec![0.0f32; n_heads * head_dim];

    for h in 0..n_heads {
        let kv_h = h / groups;
        let q_offset = h * head_dim;

        let mut scores = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let k_offset = kv_h * head_dim;
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += q[q_offset + d] * k_cache[t][k_offset + d];
            }
            score /= (head_dim as f32).sqrt();
            scores.push(score);
        }

        let weights = softmax(&scores);

        let out_offset = h * head_dim;
        for t in 0..seq_len {
            let v_offset = kv_h * head_dim;
            for d in 0..head_dim {
                output[out_offset + d] += weights[t] * v_cache[t][v_offset + d];
            }
        }
    }
    output
}

pub fn per_head_rms_norm(x: &mut [f32], weight: &[f32], n_heads: usize, head_dim: usize) {
    for h in 0..n_heads {
        let base = h * head_dim;
        let slice = &x[base..base + head_dim];
        let mean_sq = slice.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
        let scale = 1.0 / (mean_sq + GEMMA_RMS_NORM_EPS).sqrt();
        for i in 0..head_dim {
            x[base + i] *= scale * (1.0 + weight[i % weight.len()]);
        }
    }
}

pub fn embedding_lookup(
    embd_data: &[f32],
    token_id: u32,
    hidden_dim: usize,
    vocab_size: usize,
) -> Result<Vec<f32>> {
    let id = token_id as usize;
    if id >= vocab_size {
        bail!(
            "Token ID {} exceeds vocab size {}",
            token_id,
            vocab_size
        );
    }
    let start = id * hidden_dim;
    let end = start + hidden_dim;
    if end > embd_data.len() {
        bail!(
            "Embedding data too short: need {} but have {}",
            end,
            embd_data.len()
        );
    }
    Ok(embd_data[start..end].to_vec())
}

pub fn embedding_lookup_sum(
    embd_data: &[f32],
    token_ids: &[u32],
    hidden_dim: usize,
    vocab_size: usize,
) -> Result<Vec<f32>> {
    if token_ids.is_empty() {
        bail!("No token IDs provided");
    }
    let first = embedding_lookup(embd_data, token_ids[0], hidden_dim, vocab_size)?;
    if token_ids.len() == 1 {
        return Ok(first);
    }
    let mut sum = first;
    for &tid in &token_ids[1..] {
        let emb = embedding_lookup(embd_data, tid, hidden_dim, vocab_size)?;
        vec_add_inplace(&mut sum, &emb);
    }
    Ok(sum)
}

pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

pub fn top_k_sample(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

#[derive(Debug, Clone)]
pub struct GemmaLayerConfig {
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub eps: f32,
}

impl GemmaLayerConfig {
    pub fn from_dims(hidden_dim: usize, q_dim: usize, k_dim: usize, ffn_dim: usize) -> Self {
        let head_dim = Self::infer_head_dim(q_dim, k_dim);
        let n_heads = if head_dim > 0 { q_dim / head_dim } else { 1 };
        let n_kv_heads = if head_dim > 0 { (k_dim / head_dim).max(1) } else { 1 };
        Self {
            hidden_dim,
            n_heads: n_heads.max(1),
            n_kv_heads,
            head_dim: head_dim.max(1),
            ffn_dim,
            eps: GEMMA_RMS_NORM_EPS,
        }
    }

    fn infer_head_dim(q_dim: usize, k_dim: usize) -> usize {
        if k_dim == 0 || q_dim == 0 {
            return q_dim.max(k_dim).max(1);
        }
        let gcd = gcd(q_dim, k_dim);
        if gcd >= 64 {
            gcd
        } else if q_dim % 256 == 0 && k_dim % 256 == 0 {
            256
        } else {
            gcd.max(1)
        }
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_identity_weights() {
        let x = vec![3.0, 4.0];
        let w = vec![0.0, 0.0];
        let out = rms_norm(&x, &w, 1e-6);
        let expected_scale = 1.0 / ((9.0 + 16.0) / 2.0 + 1e-6f32).sqrt();
        assert!((out[0] - 3.0 * expected_scale).abs() < 1e-5);
        assert!((out[1] - 4.0 * expected_scale).abs() < 1e-5);
    }

    #[test]
    fn softmax_basic() {
        let x = vec![1.0, 2.0, 3.0];
        let s = softmax(&x);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(s[2] > s[1] && s[1] > s[0]);
    }

    #[test]
    fn silu_zero() {
        let x = vec![0.0];
        let s = silu(&x);
        assert!((s[0]).abs() < 1e-6);
    }

    #[test]
    fn matmul_identity() {
        let mat = vec![1.0, 0.0, 0.0, 1.0];
        let input = vec![3.0, 5.0];
        let out = matmul(&mat, &input, 2, 2);
        assert_eq!(out, vec![3.0, 5.0]);
    }

    #[test]
    fn embedding_lookup_basic() {
        let embd = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let v = embedding_lookup(&embd, 1, 3, 2).unwrap();
        assert_eq!(v, vec![0.4, 0.5, 0.6]);
    }

    #[test]
    fn argmax_basic() {
        let logits = vec![0.1, 0.9, 0.3];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn rope_preserves_norm() {
        let head_dim = 8;
        let mut q = vec![1.0; head_dim];
        let mut k = vec![1.0; head_dim];
        let freqs = vec![1.0; head_dim / 2];
        let norm_before_q: f32 = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        rope_apply(&mut q, &mut k, &freqs, 0, 1, 1, head_dim);
        let norm_after_q: f32 = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm_before_q - norm_after_q).abs() < 1e-4);
    }

    #[test]
    fn gqa_attention_single_token() {
        let head_dim = 4;
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![0.0, 0.0, 1.0, 0.0];
        let out = gqa_attention(&q, &k, &v, 2, 1, head_dim);
        assert_eq!(out.len(), 8);
        assert_eq!(&out[0..4], &[0.0, 0.0, 1.0, 0.0]);
        assert_eq!(&out[4..8], &[0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn per_head_rms_norm_basic() {
        let mut x = vec![3.0, 4.0, 1.0, 2.0];
        let w = vec![0.0, 0.0];
        per_head_rms_norm(&mut x, &w, 2, 2);
        let norm0 = (x[0] * x[0] + x[1] * x[1]).sqrt();
        let norm1 = (x[2] * x[2] + x[3] * x[3]).sqrt();
        assert!((norm0 - (2.0f32).sqrt()).abs() < 0.01);
        assert!((norm1 - (2.0f32).sqrt()).abs() < 0.01);
    }
}
