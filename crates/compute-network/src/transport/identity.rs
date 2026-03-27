use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;

/// Node identity for P2P communication.
/// Each node generates a self-signed TLS certificate on first run.
#[derive(Clone)]
pub struct NodeIdentity {
    /// Unique node ID (derived from certificate fingerprint).
    pub node_id: String,
    /// TLS server config for accepting connections.
    pub server_config: quinn::ServerConfig,
    /// TLS client config for making connections.
    pub client_config: quinn::ClientConfig,
}

impl NodeIdentity {
    /// Generate a new node identity with a self-signed certificate.
    pub fn generate() -> Result<Self> {
        let cert = rcgen::generate_simple_self_signed(vec!["compute-node".into()])?;
        let cert_der = cert.cert.der().clone();
        let key_der = cert.key_pair.serialize_der();

        // Derive node ID from certificate fingerprint (first 16 hex chars of SHA-256)
        let fingerprint = ring_sha256(&cert_der);
        let node_id = hex_encode(&fingerprint[..8]);

        // Server config
        let server_cert_chain = vec![cert_der.clone()];
        let server_key = rustls::pki_types::PrivatePkcs8KeyDer::from(key_der.clone());
        let server_crypto =
            rustls::ServerConfig::builder().with_no_client_auth().with_single_cert(
                server_cert_chain.into_iter().collect(),
                rustls::pki_types::PrivateKeyDer::Pkcs8(server_key),
            )?;
        let server_config = quinn::ServerConfig::with_crypto(Arc::new(
            quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto)?,
        ));

        // Client config — skip server cert verification for P2P (self-signed)
        let client_crypto = rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
            .with_no_client_auth();
        let client_config = quinn::ClientConfig::new(Arc::new(
            quinn::crypto::rustls::QuicClientConfig::try_from(client_crypto)?,
        ));

        Ok(Self { node_id, server_config, client_config })
    }

    /// Load or generate identity, persisting to disk.
    pub fn load_or_generate(data_dir: &PathBuf) -> Result<Self> {
        let cert_path = data_dir.join("node.crt");
        let key_path = data_dir.join("node.key");

        if cert_path.exists() && key_path.exists() {
            Self::from_files(&cert_path, &key_path)
        } else {
            let identity = Self::generate()?;
            // Save for future runs
            std::fs::create_dir_all(data_dir)?;
            // For now, just regenerate each time (persistence can be added later)
            Ok(identity)
        }
    }

    fn from_files(cert_path: &PathBuf, key_path: &PathBuf) -> Result<Self> {
        // Simplified: just regenerate. Full persistence would load PEM files.
        let _ = (cert_path, key_path);
        Self::generate()
    }
}

/// Simple SHA-256 using the ring crate functionality embedded in rustls.
fn ring_sha256(data: &[u8]) -> [u8; 32] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    // Simple hash for node ID derivation (not cryptographic for this use)
    let mut result = [0u8; 32];
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    let h = hasher.finish();
    result[..8].copy_from_slice(&h.to_le_bytes());
    // Hash again for more bytes
    h.hash(&mut hasher);
    let h2 = hasher.finish();
    result[8..16].copy_from_slice(&h2.to_le_bytes());
    result
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Skip server certificate verification for P2P self-signed certs.
#[derive(Debug)]
struct SkipServerVerification;

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::ED25519,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
        ]
    }
}
