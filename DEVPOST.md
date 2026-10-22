# Devpost submission: copy/paste

**Project name:** gpuroute

**Tagline:** Your agent can write training code, but it can't safely run it. gpuroute sizes, prices and launches GPU training jobs for AI agents, free options first, under a budget they can't raise.

**Track:** Hacking with Public Data APIs

**Source code:** https://github.com/sissississi-013/gpuroute

**Presentation:** [share the deck link from claude.ai, or export it to PDF/PPTX and upload]

---

## Inspiration
Coding agents can write a fine-tuning script in seconds. The moment it needs a GPU, a human has to step back in: figure out how much VRAM it needs, compare a dozen clouds, and decide whether to trust the agent with a cloud account. We researched where people actually struggle:
- Sizing training "by inference intuition is wrong by an order of magnitude," which leads to out-of-memory crashes and restarts
- Every provider's agent tooling (RunPod MCP, Hugging Face Jobs, Modal skills) covers only its own platform
- There's "no mature tool that enforces per-job GPU budgets for coding agents," and forgotten GPUs keep billing ($400 over a weekend for an idle H100)
- 85% of surveyed academic NLP researchers have $0 cloud budget, while free compute (local GPUs, Kaggle, Colab, free credits) is scattered

## What it does
gpuroute is an **MCP server + CLI + local dashboard**:
- **estimate_job**: VRAM needed (weights/optimizer, activations, overhead) and time on 16 GPUs, before renting anything
- **quote_job**: ranks every way to run the job by what you'd actually pay: your own Mac/GPU ($0, private), Kaggle, Colab, Modal (free credits), and the **live Vast.ai GPU market**
- **launch_job**: runs a LoRA fine-tune under a hard budget, or waits for human approval above a threshold
- **job_status**: step, loss curve, cost so far, artifact path. No SSH babysitting
- **connect_provider**: the agent sets up the cloud GPU itself and hands the human only the steps that must be theirs (sign-in, payment method), already opened in the browser
- **free_inference_models**: checks OpenRouter for models that are free right now, because the cheapest fine-tune is the one you don't need

## How we built it
Python. The MCP Python SDK for the agent interface, `rich` for the CLI, a stdlib HTTP server + vanilla JS for the dashboard, `transformers` + `peft` for training. Launch adapters for local Apple MPS/CUDA and Modal.

The budget is enforced in **three layers**:
1. **Policy**: per-job cap, monthly allowance, budget ≥ estimate
2. **Infrastructure**: Modal's function timeout = budget ÷ price per second, so the provider kills the container even if the agent stops polling. Local jobs have a watchdog
3. **Settlement**: actual GPU-seconds × price goes into an append-only audit log

Agents get an allowance, never credentials. There is deliberately no MCP tool to approve jobs or change the policy.

## Public data used
- **Vast.ai public GPU pricing feed** (hourly): live price + availability for 15 GPU types
- **OpenRouter models API**: which models are free to call today
- **Modal pricing**: per-second GPU prices
- **NVIDIA datasheets**: TFLOPS and VRAM for the estimator
- **Hugging Face Hub**: public datasets/models for training

## Responsible data use
- Vast's feed allows display with attribution but not redistribution, so we keep only the latest snapshot, always attribute it, and never archive or redistribute it
- Every launch states **what data leaves the device** before it runs, and logs it
- Local-first: when your machine can do it, it's $0 and the data never leaves
- Dashboard binds to localhost only and rejects cross-origin requests

## Challenges
- Small models can't saturate a GPU, so we calibrated the estimator's utilization curve on a real M2 run (~12% MFU at 135M params)
- A stalled model download slipped past the in-loop time cap, so we added a watchdog plus an external kill
- Modal needs a payment method before GPU use. connect_provider detects this in 3 seconds without spending anything

## Accomplishments
- A real agent (Claude Code) used it correctly on the first try: it quoted, picked the free local GPU, launched with a $1 cap and reported status
- A real LoRA fine-tune on the M2 GPU: loss 2.75 → 2.23, $0.00, settled in the audit log
- A 30-second cap stopped a job at 24 seconds, as designed

## What's next
Launch adapters for Vast.ai and Kaggle, QLoRA launches, estimates that learn from your own past runs, and team allowances.

## Built with
python, mcp, modal, pytorch, transformers, peft, hugging-face, vast-ai, openrouter, claude-code

*Built with AI-assisted development (Claude Code); open-source libraries credited in the README.*
