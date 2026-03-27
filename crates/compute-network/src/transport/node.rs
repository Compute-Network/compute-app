use anyhow::{Context, Result};
use quinn::{Connection, Endpoint};
use std::net::SocketAddr;
use tracing::info;

use super::identity::NodeIdentity;
use super::protocol::{self, PipelineMessage};

/// A P2P transport node that can both listen for and initiate QUIC connections.
pub struct TransportNode {
    identity: NodeIdentity,
    endpoint: Endpoint,
    listen_addr: SocketAddr,
}

impl TransportNode {
    /// Create a new transport node bound to the given address.
    pub async fn bind(addr: SocketAddr) -> Result<Self> {
        let identity = NodeIdentity::generate()?;

        let endpoint = Endpoint::server(identity.server_config.clone(), addr)
            .context("Failed to bind QUIC endpoint")?;

        let listen_addr = endpoint.local_addr()?;

        info!("Transport node {} listening on {}", identity.node_id, listen_addr);

        Ok(Self { identity, endpoint, listen_addr })
    }

    /// Get this node's ID.
    pub fn node_id(&self) -> &str {
        &self.identity.node_id
    }

    /// Get the listening address.
    pub fn listen_addr(&self) -> SocketAddr {
        self.listen_addr
    }

    /// Connect to a remote peer.
    pub async fn connect(&self, addr: SocketAddr) -> Result<PeerConnection> {
        let mut endpoint = self.endpoint.clone();
        endpoint.set_default_client_config(self.identity.client_config.clone());

        let connection = endpoint
            .connect(addr, "compute-node")
            .context("Failed to initiate QUIC connection")?
            .await
            .context("QUIC handshake failed")?;

        info!("Connected to peer at {}", addr);

        Ok(PeerConnection { connection })
    }

    /// Accept incoming connections. Returns the next connection when one arrives.
    pub async fn accept(&self) -> Result<PeerConnection> {
        let incoming =
            self.endpoint.accept().await.ok_or_else(|| anyhow::anyhow!("Endpoint closed"))?;

        let connection = incoming.await.context("Failed to accept connection")?;

        info!("Accepted connection from {}", connection.remote_address());

        Ok(PeerConnection { connection })
    }

    /// Gracefully close the endpoint.
    pub fn close(&self) {
        self.endpoint.close(quinn::VarInt::from_u32(0), b"shutdown");
    }
}

/// A connection to a pipeline peer.
pub struct PeerConnection {
    connection: Connection,
}

impl PeerConnection {
    /// Get the remote address.
    pub fn remote_addr(&self) -> SocketAddr {
        self.connection.remote_address()
    }

    /// Open a new bidirectional stream for sending a message.
    pub async fn send_message(&self, msg: &PipelineMessage) -> Result<()> {
        let (mut send, _recv) = self.connection.open_bi().await.context("Failed to open stream")?;

        protocol::write_message(&mut send, msg).await?;
        send.finish()?;

        Ok(())
    }

    /// Open a unidirectional stream for sending activations (fire-and-forget).
    pub async fn send_activations(&self, msg: &PipelineMessage) -> Result<()> {
        let mut send = self.connection.open_uni().await.context("Failed to open uni stream")?;

        protocol::write_message(&mut send, msg).await?;
        send.finish()?;

        Ok(())
    }

    /// Accept an incoming unidirectional stream and read a message.
    pub async fn recv_activations(&self) -> Result<PipelineMessage> {
        let mut recv = self.connection.accept_uni().await.context("Failed to accept uni stream")?;

        protocol::read_message(&mut recv).await
    }

    /// Accept an incoming bidirectional stream and read a message.
    pub async fn recv_message(&self) -> Result<(PipelineMessage, quinn::SendStream)> {
        let (send, mut recv) =
            self.connection.accept_bi().await.context("Failed to accept bi stream")?;

        let msg = protocol::read_message(&mut recv).await?;
        Ok((msg, send))
    }

    /// Close this connection gracefully.
    pub fn close(&self) {
        self.connection.close(quinn::VarInt::from_u32(0), b"done");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_transport_node_bind() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let node = TransportNode::bind(addr).await.unwrap();
        assert!(!node.node_id().is_empty());
        assert_ne!(node.listen_addr().port(), 0);
        node.close();
    }

    #[tokio::test]
    async fn test_connect_and_send() {
        // Start a listener
        let server_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let server = TransportNode::bind(server_addr).await.unwrap();
        let server_addr = server.listen_addr();

        // Connect from a client
        let client_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let client = TransportNode::bind(client_addr).await.unwrap();

        // Spawn server accept
        let server_handle = tokio::spawn(async move {
            let peer = server.accept().await.unwrap();
            let msg = peer.recv_activations().await.unwrap();
            match msg {
                PipelineMessage::Ping(ping) => {
                    assert_eq!(ping.node_id, "test");
                }
                _ => panic!("Expected Ping"),
            }
            server.close();
        });

        // Client sends
        let conn = client.connect(server_addr).await.unwrap();
        let msg = PipelineMessage::Ping(protocol::PingMessage {
            node_id: "test".into(),
            timestamp_ms: 12345,
        });
        conn.send_activations(&msg).await.unwrap();

        // Wait for server to process before closing
        server_handle.await.unwrap();
        conn.close();
        client.close();
    }
}
