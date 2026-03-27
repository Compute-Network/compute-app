use serde::{Deserialize, Serialize};

/// Read-only wallet info. Compute never holds private keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletInfo {
    pub address: String,
    pub sol_balance: Option<f64>,
    pub compute_balance: Option<f64>,
}

impl WalletInfo {
    pub fn new(address: String) -> Self {
        Self { address, sol_balance: None, compute_balance: None }
    }

    /// Check if a wallet address has been set.
    pub fn is_configured(&self) -> bool {
        !self.address.is_empty()
    }

    /// Mock wallet info for initial development.
    pub fn mock() -> Self {
        Self {
            address: "COMPUTE...mock".into(),
            sol_balance: Some(2.45),
            compute_balance: Some(12_450.0),
        }
    }
}
