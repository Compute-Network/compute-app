# Cost Estimates — Infrastructure Scaling

GPU compute is provided by node operators at no cost to Compute. These estimates cover only the central infrastructure.

---

## 10 Nodes

| Component | Service | Cost/mo |
|-----------|---------|---------|
| Orchestrator + API | Railway (hobby) | $5-10 |
| Database | Supabase (free tier) | $0 |
| Solana RPC | Helius/QuickNode free tier | $0 |
| Binary CDN | GitHub Releases + Cloudflare R2 | $0 |
| Domain | Cloudflare | ~$1 |
| Monitoring | Not needed yet | $0 |
| **Total** | | **~$5-15/mo** |

**Serving capacity:** ~2 pipelines of 5 nodes. ~4 tok/s aggregate on 70B models (~10-20 tok/s on smaller models with fewer stages). Handful of API consumers doing batch work.

---

## 100 Nodes

| Component | Service | Cost/mo |
|-----------|---------|---------|
| Orchestrator + API | Railway (Pro, larger instance) | $20-40 |
| Database | Supabase Pro | $25 |
| Solana RPC | Paid tier (depending on reward frequency) | $0-50 |
| Binary CDN | GitHub Releases + Cloudflare R2 | $0-5 |
| Domain | Cloudflare | ~$1 |
| Monitoring | Grafana Cloud (free tier) | $0 |
| **Total** | | **~$50-120/mo** |

**Serving capacity:** ~20 pipelines, ~40 tok/s aggregate on 70B. 100-500 concurrent API consumers for batch workloads. Revenue at $0.10/M tokens covers infra easily.

---

## 1,000 Nodes

| Component | Service | Cost/mo |
|-----------|---------|---------|
| Orchestrator + API | Railway (2 services, auto-scaled) | $80-200 |
| Database | Supabase Pro | $25-75 |
| Solana RPC | Paid tier | $50-200 |
| Load balancer | Cloudflare (in front of API) | $0-20 |
| Binary CDN | Cloudflare R2 | $5-15 |
| Monitoring | Grafana Cloud | $0-50 |
| **Total** | | **~$200-550/mo** |

**Serving capacity:** ~200 pipelines, ~400 tok/s aggregate on 70B. Thousands of concurrent API consumers. Conservative pricing ($0.05/M tokens) at 400 tok/s ≈ $50K+/mo revenue potential if pipelines stay busy.

---

## 10,000+ Nodes

| Component | Service | Cost/mo |
|-----------|---------|---------|
| Orchestrator | Fly.io (multi-region, nodes connect to nearest) | $300-800 |
| API | Railway or Fly.io (auto-scaled) | $100-300 |
| Database | Supabase Pro or Railway managed Postgres | $75-200 |
| Cache/pub-sub | Upstash Redis (serverless) | $30-100 |
| Solana RPC | Dedicated node or premium tier | $200-500 |
| Load balancer | Cloudflare | $20-50 |
| Binary CDN | Cloudflare R2 | $15-40 |
| Monitoring | Grafana Cloud (paid) | $50-150 |
| **Total** | | **~$800-2,000/mo** |

**Serving capacity:** ~2,000 pipelines, ~4,000 tok/s aggregate. At this scale, infrastructure costs are <1% of potential revenue.

---

## Scaling path summary

Infrastructure costs are trivial relative to the free GPU compute from node operators:

| Scale | Infra cost | Revenue potential | Ratio |
|-------|-----------|-------------------|-------|
| 10 nodes | ~$10/mo | Negligible (testing) | — |
| 100 nodes | ~$80/mo | ~$2-5K/mo | 1-2% |
| 1,000 nodes | ~$400/mo | ~$50K+/mo | <1% |
| 10,000 nodes | ~$1,500/mo | ~$500K+/mo | <0.5% |

Revenue estimates assume conservative pricing ($0.05-0.10/M tokens) and moderate pipeline utilization (~50%). Actual revenue depends heavily on demand and pipeline uptime.

---

## Service choices rationale

- **Railway** for orchestrator/API: deploys from GitHub, auto-scales replicas, handles WebSocket connections, usage-based pricing. Upgrade path to Fly.io for multi-region at 10K+ nodes.
- **Supabase** for database: Postgres with built-in auth (including Solana wallet auth for future web claim portal), realtime subscriptions for live dashboards, Row Level Security.
- **Cloudflare** for DNS/CDN/edge: free tier is generous, R2 for binary hosting, can add edge caching on API later.
- **Grafana Cloud** for monitoring: free tier covers early stages, upgrade when needed.
