use anyhow::{Context, Result};
use bollard::Docker;
use bollard::container::{
    Config, CreateContainerOptions, StartContainerOptions, StopContainerOptions,
};
use bollard::image::CreateImageOptions;
use bollard::models::HostConfig;
use futures_util::StreamExt;
use tracing::{error, info};

/// Docker container manager for workload execution.
pub struct ContainerManager {
    docker: Docker,
}

/// Container status information.
#[derive(Debug, Clone)]
pub struct ContainerInfo {
    pub id: String,
    pub name: String,
    pub image: String,
    pub status: ContainerStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContainerStatus {
    Running,
    Stopped,
    Creating,
    Error(String),
}

impl ContainerManager {
    /// Create a new container manager, connecting to the local Docker daemon.
    pub async fn new() -> Result<Self> {
        let docker = Docker::connect_with_local_defaults()
            .context("Failed to connect to Docker. Is Docker running?")?;

        // Verify connection
        docker.ping().await.context("Docker ping failed. Is the Docker daemon running?")?;

        Ok(Self { docker })
    }

    /// Check if Docker is available and responsive.
    pub async fn health_check(&self) -> Result<String> {
        let version = self.docker.version().await?;
        Ok(version.version.unwrap_or_else(|| "unknown".into()))
    }

    /// Pull a container image if not already present.
    pub async fn pull_image(&self, image: &str) -> Result<()> {
        info!("Pulling image: {image}");

        let options = CreateImageOptions { from_image: image, ..Default::default() };

        let mut stream = self.docker.create_image(Some(options), None, None);

        while let Some(result) = stream.next().await {
            match result {
                Ok(info) => {
                    if let Some(status) = info.status {
                        tracing::debug!("Pull: {status}");
                    }
                }
                Err(e) => {
                    error!("Pull error: {e}");
                    return Err(e.into());
                }
            }
        }

        info!("Image pulled: {image}");
        Ok(())
    }

    /// Create and start a container for a workload.
    pub async fn start_workload(
        &self,
        name: &str,
        image: &str,
        gpu_enabled: bool,
        env_vars: Vec<String>,
        port_bindings: Vec<(u16, u16)>,
    ) -> Result<String> {
        let mut host_config = HostConfig {
            // Security: no privileged mode, no host networking
            privileged: Some(false),
            network_mode: Some("bridge".into()),
            // Memory limit: 16GB default
            memory: Some(16 * 1024 * 1024 * 1024),
            // Restart policy: unless stopped
            restart_policy: Some(bollard::models::RestartPolicy {
                name: Some(bollard::models::RestartPolicyNameEnum::UNLESS_STOPPED),
                maximum_retry_count: Some(3),
            }),
            ..Default::default()
        };

        // GPU passthrough for NVIDIA
        if gpu_enabled {
            host_config.device_requests = Some(vec![bollard::models::DeviceRequest {
                driver: Some("nvidia".into()),
                count: Some(-1), // All GPUs
                capabilities: Some(vec![vec!["gpu".into()]]),
                ..Default::default()
            }]);
        }

        // Port bindings
        if !port_bindings.is_empty() {
            let mut bindings = std::collections::HashMap::new();
            for (host_port, container_port) in &port_bindings {
                bindings.insert(
                    format!("{container_port}/tcp"),
                    Some(vec![bollard::models::PortBinding {
                        host_ip: Some("127.0.0.1".into()),
                        host_port: Some(host_port.to_string()),
                    }]),
                );
            }
            host_config.port_bindings = Some(bindings);
        }

        let container_config = Config {
            image: Some(image.to_string()),
            env: Some(env_vars),
            host_config: Some(host_config),
            ..Default::default()
        };

        let options = CreateContainerOptions { name, platform: None };

        info!("Creating container: {name} from {image}");
        let container = self
            .docker
            .create_container(Some(options), container_config)
            .await
            .context("Failed to create container")?;

        self.docker
            .start_container(&container.id, None::<StartContainerOptions<String>>)
            .await
            .context("Failed to start container")?;

        info!("Container started: {} ({})", name, &container.id[..12]);
        Ok(container.id)
    }

    /// Stop a running container.
    pub async fn stop_container(&self, container_id: &str) -> Result<()> {
        info!("Stopping container: {}", &container_id[..12.min(container_id.len())]);

        let options = StopContainerOptions { t: 10 };

        self.docker
            .stop_container(container_id, Some(options))
            .await
            .context("Failed to stop container")?;

        Ok(())
    }

    /// Remove a container.
    pub async fn remove_container(&self, container_id: &str) -> Result<()> {
        self.docker
            .remove_container(
                container_id,
                Some(bollard::container::RemoveContainerOptions {
                    force: true,
                    ..Default::default()
                }),
            )
            .await
            .context("Failed to remove container")?;

        Ok(())
    }

    /// List all Compute-managed containers.
    pub async fn list_containers(&self) -> Result<Vec<ContainerInfo>> {
        let containers = self
            .docker
            .list_containers(Some(bollard::container::ListContainersOptions::<String> {
                all: true,
                ..Default::default()
            }))
            .await?;

        let mut infos = Vec::new();
        for c in containers {
            let name = c
                .names
                .as_ref()
                .and_then(|n| n.first())
                .map(|n| n.trim_start_matches('/').to_string())
                .unwrap_or_default();

            // Only show compute-managed containers
            if !name.starts_with("compute-") {
                continue;
            }

            let status = match c.state.as_deref() {
                Some("running") => ContainerStatus::Running,
                Some("exited") | Some("dead") => ContainerStatus::Stopped,
                Some("created") => ContainerStatus::Creating,
                _ => ContainerStatus::Error("unknown state".into()),
            };

            infos.push(ContainerInfo {
                id: c.id.unwrap_or_default(),
                name,
                image: c.image.unwrap_or_default(),
                status,
            });
        }

        Ok(infos)
    }
}
