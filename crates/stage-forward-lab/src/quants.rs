use anyhow::{Result, bail};
use half::f16;

pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;
pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q4_K: u32 = 12;
pub const GGML_TYPE_Q5_K: u32 = 13;
pub const GGML_TYPE_Q6_K: u32 = 14;
pub const GGML_TYPE_BF16: u32 = 30;

const BLOCK_Q4_K_SIZE: usize = 2 * 2 + K_SCALE_SIZE + QK_K / 2;
const BLOCK_Q5_K_SIZE: usize = 2 * 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2;
const BLOCK_Q6_K_SIZE: usize = 2 + QK_K / 16 + 3 * QK_K / 4;

pub fn ggml_type_name(ggml_type: u32) -> &'static str {
    match ggml_type {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        30 => "BF16",
        _ => "UNKNOWN",
    }
}

pub fn dequantize_tensor(ggml_type: u32, bytes: &[u8]) -> Result<Vec<f32>> {
    match ggml_type {
        GGML_TYPE_F32 => dequantize_f32_tensor(bytes),
        GGML_TYPE_F16 => dequantize_f16_tensor(bytes),
        GGML_TYPE_Q4_K => dequantize_q4_k_tensor(bytes),
        GGML_TYPE_Q5_K => dequantize_q5_k_tensor(bytes),
        GGML_TYPE_Q6_K => dequantize_q6_k_tensor(bytes),
        GGML_TYPE_BF16 => dequantize_bf16_tensor(bytes),
        _ => bail!(
            "Unsupported GGML type {} ({})",
            ggml_type,
            ggml_type_name(ggml_type)
        ),
    }
}

pub fn dequantize_f32_tensor(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        bail!("F32 tensor length {} is not divisible by 4", bytes.len());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

pub fn dequantize_f16_tensor(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 2 != 0 {
        bail!("F16 tensor length {} is not divisible by 2", bytes.len());
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        out.push(fp16_to_f32([chunk[0], chunk[1]]));
    }
    Ok(out)
}

pub fn dequantize_bf16_tensor(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 2 != 0 {
        bail!("BF16 tensor length {} is not divisible by 2", bytes.len());
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(bf16_to_f32(bits));
    }
    Ok(out)
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub fn dequantize_q4_k_tensor(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % BLOCK_Q4_K_SIZE != 0 {
        bail!(
            "Q4_K tensor length {} is not divisible by block size {}",
            bytes.len(),
            BLOCK_Q4_K_SIZE
        );
    }
    let mut out = vec![0.0f32; bytes.len() / BLOCK_Q4_K_SIZE * QK_K];
    for (block_idx, block) in bytes.chunks_exact(BLOCK_Q4_K_SIZE).enumerate() {
        dequantize_q4_k_block(block, &mut out[block_idx * QK_K..(block_idx + 1) * QK_K])?;
    }
    Ok(out)
}

pub fn dequantize_q5_k_tensor(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % BLOCK_Q5_K_SIZE != 0 {
        bail!(
            "Q5_K tensor length {} is not divisible by block size {}",
            bytes.len(),
            BLOCK_Q5_K_SIZE
        );
    }
    let mut out = vec![0.0f32; bytes.len() / BLOCK_Q5_K_SIZE * QK_K];
    for (block_idx, block) in bytes.chunks_exact(BLOCK_Q5_K_SIZE).enumerate() {
        dequantize_q5_k_block(block, &mut out[block_idx * QK_K..(block_idx + 1) * QK_K])?;
    }
    Ok(out)
}

pub fn dequantize_q5_k_block(block: &[u8], out: &mut [f32]) -> Result<()> {
    if block.len() != BLOCK_Q5_K_SIZE || out.len() != QK_K {
        bail!("Q5_K block decode shape mismatch");
    }
    let d = fp16_to_f32([block[0], block[1]]);
    let dmin = fp16_to_f32([block[2], block[3]]);
    let scales = &block[4..4 + K_SCALE_SIZE];
    let qh = &block[4 + K_SCALE_SIZE..4 + K_SCALE_SIZE + QK_K / 8];
    let qs = &block[4 + K_SCALE_SIZE + QK_K / 8..];

    let mut is = 0usize;
    let mut q_offset = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;

    for j in (0..QK_K).step_by(64) {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let d1 = d * sc1 as f32;
        let m1 = dmin * m1 as f32;
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc2 as f32;
        let m2 = dmin * m2 as f32;
        for l in 0..32 {
            let q_lo = qs[q_offset + l];
            let qh_byte = qh[l];
            let hbit1 = if qh_byte & u1 != 0 { 16 } else { 0 };
            let hbit2 = if qh_byte & u2 != 0 { 16 } else { 0 };
            out[j + l * 2] = d1 * ((q_lo & 0x0F) as f32 + hbit1 as f32) - m1;
            out[j + l * 2 + 1] = d2 * ((q_lo >> 4) as f32 + hbit2 as f32) - m2;
        }
        q_offset += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
    Ok(())
}

pub fn dequantize_q6_k_tensor(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % BLOCK_Q6_K_SIZE != 0 {
        bail!(
            "Q6_K tensor length {} is not divisible by block size {}",
            bytes.len(),
            BLOCK_Q6_K_SIZE
        );
    }
    let mut out = vec![0.0f32; bytes.len() / BLOCK_Q6_K_SIZE * QK_K];
    for (block_idx, block) in bytes.chunks_exact(BLOCK_Q6_K_SIZE).enumerate() {
        dequantize_q6_k_block(block, &mut out[block_idx * QK_K..(block_idx + 1) * QK_K])?;
    }
    Ok(out)
}

pub fn dequantize_q4_k_block(block: &[u8], out: &mut [f32]) -> Result<()> {
    if block.len() != BLOCK_Q4_K_SIZE || out.len() != QK_K {
        bail!("Q4_K block decode shape mismatch");
    }
    let d = fp16_to_f32([block[0], block[1]]);
    let dmin = fp16_to_f32([block[2], block[3]]);
    let scales = &block[4..4 + K_SCALE_SIZE];
    let qs = &block[4 + K_SCALE_SIZE..];

    let mut is = 0usize;
    let mut q_offset = 0usize;
    for j in (0..QK_K).step_by(64) {
        let (sc1, m1) = get_scale_min_k4(is, scales);
        let d1 = d * sc1 as f32;
        let m1 = dmin * m1 as f32;
        let (sc2, m2) = get_scale_min_k4(is + 1, scales);
        let d2 = d * sc2 as f32;
        let m2 = dmin * m2 as f32;
        for l in 0..32 {
            let q = qs[q_offset + l];
            out[j + l] = d1 * (q & 0x0F) as f32 - m1;
            out[j + 32 + l] = d2 * (q >> 4) as f32 - m2;
        }
        q_offset += 32;
        is += 2;
    }
    Ok(())
}

pub fn dequantize_q6_k_block(block: &[u8], out: &mut [f32]) -> Result<()> {
    if block.len() != BLOCK_Q6_K_SIZE || out.len() != QK_K {
        bail!("Q6_K block decode shape mismatch");
    }
    let ql = &block[..QK_K / 2];
    let qh = &block[QK_K / 2..QK_K / 2 + QK_K / 4];
    let scales_start = QK_K / 2 + QK_K / 4;
    let scales = &block[scales_start..scales_start + QK_K / 16];
    let d = fp16_to_f32([block[BLOCK_Q6_K_SIZE - 2], block[BLOCK_Q6_K_SIZE - 1]]);

    let mut y_offset = 0usize;
    let mut ql_offset = 0usize;
    let mut qh_offset = 0usize;
    let mut sc_offset = 0usize;
    for _ in (0..QK_K).step_by(128) {
        for l in 0..32 {
            let is = l / 16;
            let qh_byte = qh[qh_offset + l];
            let q1 =
                (((ql[ql_offset + l] & 0x0F) | (((qh_byte >> 0) & 0x03) << 4)) as i32 - 32) as f32;
            let q2 = (((ql[ql_offset + 32 + l] & 0x0F) | (((qh_byte >> 2) & 0x03) << 4)) as i32
                - 32) as f32;
            let q3 = ((((ql[ql_offset + l] >> 4) & 0x0F) | (((qh_byte >> 4) & 0x03) << 4)) as i32
                - 32) as f32;
            let q4 = ((((ql[ql_offset + 32 + l] >> 4) & 0x0F) | (((qh_byte >> 6) & 0x03) << 4))
                as i32
                - 32) as f32;

            out[y_offset + l] = d * (scales[sc_offset + is] as i8 as f32) * q1;
            out[y_offset + 32 + l] = d * (scales[sc_offset + is + 2] as i8 as f32) * q2;
            out[y_offset + 64 + l] = d * (scales[sc_offset + is + 4] as i8 as f32) * q3;
            out[y_offset + 96 + l] = d * (scales[sc_offset + is + 6] as i8 as f32) * q4;
        }
        y_offset += 128;
        ql_offset += 64;
        qh_offset += 32;
        sc_offset += 8;
    }
    Ok(())
}

fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn fp16_to_f32(bytes: [u8; 2]) -> f32 {
    f16::from_bits(u16::from_le_bytes(bytes)).to_f32()
}

pub fn bytes_per_row(ggml_type: u32, row_elements: usize) -> Result<usize> {
    match ggml_type {
        GGML_TYPE_F32 => Ok(row_elements * 4),
        GGML_TYPE_F16 | GGML_TYPE_BF16 => Ok(row_elements * 2),
        GGML_TYPE_Q4_K => {
            if row_elements % QK_K != 0 {
                bail!("Q4_K row length {} not divisible by {}", row_elements, QK_K);
            }
            Ok(row_elements / QK_K * BLOCK_Q4_K_SIZE)
        }
        GGML_TYPE_Q5_K => {
            if row_elements % QK_K != 0 {
                bail!("Q5_K row length {} not divisible by {}", row_elements, QK_K);
            }
            Ok(row_elements / QK_K * BLOCK_Q5_K_SIZE)
        }
        GGML_TYPE_Q6_K => {
            if row_elements % QK_K != 0 {
                bail!("Q6_K row length {} not divisible by {}", row_elements, QK_K);
            }
            Ok(row_elements / QK_K * BLOCK_Q6_K_SIZE)
        }
        _ => bail!("Unsupported GGML type {} for row size", ggml_type),
    }
}

pub fn dequantize_row(ggml_type: u32, raw: &[u8], row_idx: usize, row_elements: usize) -> Result<Vec<f32>> {
    let bpr = bytes_per_row(ggml_type, row_elements)?;
    let start = row_idx * bpr;
    let end = start + bpr;
    if end > raw.len() {
        bail!(
            "Row {} needs bytes {}..{} but tensor has only {} bytes",
            row_idx, start, end, raw.len()
        );
    }
    dequantize_tensor(ggml_type, &raw[start..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q6_k_zero_block_dequantizes_to_zeroes() {
        let block = vec![0u8; BLOCK_Q6_K_SIZE];
        let mut out = vec![1.0f32; QK_K];
        dequantize_q6_k_block(&block, &mut out).unwrap();
        assert!(out.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn q4_k_zero_block_dequantizes_to_zeroes() {
        let block = vec![0u8; BLOCK_Q4_K_SIZE];
        let mut out = vec![1.0f32; QK_K];
        dequantize_q4_k_block(&block, &mut out).unwrap();
        assert!(out.iter().all(|value| *value == 0.0));
    }
}
