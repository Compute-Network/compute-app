pub mod wallet;

/// Validate a Solana public address (base58, 32-44 characters).
pub fn is_valid_address(address: &str) -> bool {
    if address.is_empty() || address.len() < 32 || address.len() > 44 {
        return false;
    }
    // Base58 character set (excludes 0, O, I, l)
    address
        .chars()
        .all(|c| c.is_ascii_alphanumeric() && c != '0' && c != 'O' && c != 'I' && c != 'l')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_address() {
        // Valid base58 string of correct length
        assert!(is_valid_address("DRpbCBMxVnDK7maPM5tGv6MvB3v1sRMC86PZ8okm21hy"));
        assert!(is_valid_address("So11111111111111111111111111111112"));
    }

    #[test]
    fn test_invalid_address_empty() {
        assert!(!is_valid_address(""));
    }

    #[test]
    fn test_invalid_address_too_short() {
        assert!(!is_valid_address("abc123"));
    }

    #[test]
    fn test_invalid_address_too_long() {
        assert!(!is_valid_address("DRpbCBMxVnDK7maPM5tGv6MvB3v1sRMC86PZ8okm21hyEXTRA"));
    }

    #[test]
    fn test_invalid_address_bad_chars() {
        // Contains '0' which is not in base58
        assert!(!is_valid_address("DRpbCBMxVnDK7maPM5tGv6MvB3v1sRMC86PZ80km21hy"));
        // Contains 'O' which is not in base58
        assert!(!is_valid_address("DRpbCBMxVnDK7maPM5tGv6MvB3v1sRMC86PZ8Okm21hy"));
    }
}
