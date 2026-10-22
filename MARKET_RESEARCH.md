# Market research: "OpenRouter for compute" (2026-10-22)

## Already exists (don't compete head-on)
| Player | What it is | Gap |
|---|---|---|
| **Shadeform** | One API to provision GPUs across ~20–30 clouds, ~7–12% markup (per third-party pages) | Raw VM provisioning; no job sizing, no free tiers, not agent-native |
| **SkyPilot** (OSS) | Multi-cloud orchestrator on your own accounts, failover across providers | YAML/infra-heavy; you must bring cloud accounts + know what GPU you need |
| **dstack** | Orchestrator + "dstack sky" broker | Marketplace rated weaker by SemiAnalysis |
| **Prime Intellect, Hyperbolic, SF Compute, Vast** | GPU marketplaces / exchanges | Sell capacity; you still pick the GPU and babysit |
| **RunPod MCP** (official, ~50 tools), **HF Jobs** (`hf jobs run --flavor …`, needs Pro), **Modal skills** | Agent access to *one* provider each | Single-vendor; agent can't compare options |
| **Tinker** (Thinking Machines) | Fine-tuning API billed per token, not per GPU-hour | Different abstraction; one vendor |

## Real pain (evidence)
- **Money/access:** 85% of surveyed academic NLP researchers had **zero cloud compute budget** (arXiv 2410.23261). H100 1-yr rental prices rose ~40% from Oct 2025 to Mar 2026 (SemiAnalysis via Seeking Alpha).
- **Sizing:** Wrong GPU → CUDA OOM → restart on a bigger instance (common pitfall in vendor guides).
- **Waste:** Forgotten idle instances, spot preemption without checkpoints.
- **Agents + spend:** Agents leave GPUs running; one reported $6.5K AWS bill (unverified). Blogs and papers say **no mature tool enforces per-job GPU budgets for coding agents**; people stitch together IAM, quotas and auto-termination by hand.
- **Free compute is scattered:** Kaggle, Colab, Lightning (15 credits/mo for edu), Modal credits, NVIDIA Inception, cloud research credits (AWS research credits take 90–120 days to review). Aggregators ignore free tiers; credits are provider-locked and expire.

## Wedge
Not "another GPU marketplace." Instead: a **compute broker for agents**, i.e. a CLI + MCP server that:
1. **Estimates** what a job needs (VRAM, hours) from model size + method (LoRA/QLoRA/full) + tokens
2. **Quotes** every way to run it, including free tiers and per-token fine-tune APIs, from live public data
3. **Launches** with a hard budget cap + idle auto-kill
4. **Reports** cost, logs and artifacts back to the agent

Users: small research teams, students, indie builders whose agents (Claude Code, Codex, Cursor) need to run training jobs.

## Sources
- https://www.spheron.network/blog/spheron-vs-shadeform/
- https://www.spheron.network/blog/skypilot-multi-cloud-gpu-orchestration-guide/
- https://newsletter.semianalysis.com/p/clustermax-20-the-industry-standard
- https://www.runpod.io/blog/manage-your-runpod-infrastructure-from-any-ai-assistant-introducing-the-runpod-mcp-server
- https://huggingface.co/docs/huggingface_hub/v1.1.7/guides/jobs
- https://www.beam.cloud/blog/tinker-model-pricing
- https://arxiv.org/html/2410.23261v1
- https://seekingalpha.com/news/4572260
- https://www.runpod.io/articles/guides/cloud-gpu-mistakes-to-avoid
- https://www.developersdigest.tech/blog/ai-infrastructure-agents-need-spend-guardrails
- https://www.qovery.com/blog/governed-audited-agent-access-mistral-large-4-cloud-spend
- https://github.com/PathOnAI/computing_resources
- https://www.primeintellect.ai/blog/compute
- https://www.latent.space/p/sfcompute
