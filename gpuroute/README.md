# gpuroute

**The cheapest GPU is the one you already have and aren't using.**

Compute sits idle everywhere: teammates' laptops, the gaming PC at home, the lab workstation, unused cloud credits. Meanwhile agents only know about the one machine they run on. gpuroute gives agents (Claude Code, Codex, Cursor, anything that speaks MCP) a **map of every machine and provider they can use**, routes any GPU workload to the best idle one, and keeps spending and owner rules in human hands. Workloads include transcription, embeddings, image generation, rendering, batch inference and fine-tuning.

**Live site:** https://gpuroute-site.vercel.app (public snapshot; run `gpuroute ui` for your live dashboard)

Built for the National Renaissance Hackathon at super{set}. Track: **Hacking with Public Data APIs**. It is an MCP server and a CLI built on live public pricing and model data.

```
                        ┌─ shared nodes: teammates' idle GPUs  (gpuroute share, owner rules)
agent ──MCP──▶ compute_map ─┼─ this machine
   │                    ├─ cloud: Modal ($30/mo free credits)
   │                    ├─ spare-GPU market: Vast.ai live prices (1,600+ GPUs listed)
   │                    └─ free tiers: Kaggle, Colab
   ├──▶ run_workload(script, deps, files)  → best idle node → outputs come back
   └──▶ quote_job / launch_job (fine-tunes) → hard budget cap, human approval above a threshold
```

### Get started (one command)

```bash
uvx --from "git+https://github.com/sissississi-013/gpuroute#subdirectory=gpuroute" gpuroute init
```
Checks this machine, connects gpuroute to Claude Code / Codex (prints the config for Cursor), and shows the three commands you need. Then ask your agent: *"What compute can you use right now?"*

### Share a machine (one command, nothing to pre-install but uv)

```bash
uvx --from "git+https://github.com/sissississi-013/gpuroute#subdirectory=gpuroute" gpuroute share --name iana
```

The owner's rules are enforced on their machine: jobs run only while it's idle (CPU/GPU under 60% busy, read from the GPU driver), each job is capped (30 min by default), one job runs at a time, and every job is printed in the owner's terminal with who sent it and what it installs. Ctrl-C stops sharing. See [SHARE.md](../SHARE.md).

### Run anything on the best idle GPU

```bash
gpuroute join "iana@172.20.x.x:7400#<token>"   # the line `share` printed
gpuroute map                                            # everything reachable + how busy it is
gpuroute run examples/benchmark.py                      # deps read from imports → idle node → results back
gpuroute run examples/embed.py --file notes.md
```

Jobs are plain Python scripts that declare their own pip dependencies (`uv` installs them in a throwaway env on the target) and write results to `./out`.

## The problems it solves

| Pain | Evidence | What gpuroute does |
|---|---|---|
| "Which GPU do I need?" People size training like inference, run out of memory, and restart on a bigger GPU | A whole genre of VRAM calculators; sizing training "by inference intuition is wrong by an order of magnitude" | `estimate_job`: VRAM with breakdown + time on 16 GPUs, before renting anything |
| Agents can't compare providers | RunPod, Hugging Face Jobs and Modal each ship single-vendor agent tools | `quote_job`: one ranked list across this machine, free tiers, Modal and the live Vast.ai market |
| You can't trust an agent with your cloud account | "No mature tool enforces per-job GPU budgets for coding agents"; reported runaway agent bills | Agents get an **allowance**, not credentials. Per-job caps, monthly caps, human approval above a threshold |
| Forgotten GPUs keep billing | $400 weekend on an idle H100 (r/mlops) | The budget becomes the provider's **hard timeout**: the container dies at budget / price even if the agent stops polling |
| 85% of academic NLP researchers have $0 cloud budget | arXiv 2410.23261 | Free first: your own Mac/GPU, Modal free credits, Kaggle/Colab |
| Training outlives the agent session; SSH babysitting is fragile | Claude Code issues #67123, #52790 | Submit → job id → poll structured status. No SSH |

## Quick start

```bash
cd gpuroute
uv venv --python 3.12 && uv pip install -e ".[local]"   # [local] = torch etc. for training on this machine

.venv/bin/gpuroute quote Qwen/Qwen2.5-0.5B-Instruct --examples 2000 --budget 1
.venv/bin/gpuroute launch HuggingFaceTB/SmolLM2-135M-Instruct mlabonne/guanaco-llama2-1k --budget 0.5 --examples 120 --seq-len 256
.venv/bin/gpuroute status <job_id> --watch
.venv/bin/gpuroute ui            # dashboard at http://127.0.0.1:7311
```

### Give it to your agent (Claude Code)

```bash
claude mcp add gpuroute -- "$(pwd)/.venv/bin/gpuroute-mcp"
```

Then ask: *"Fine-tune Qwen/Qwen2.5-0.5B-Instruct on mlabonne/guanaco-llama2-1k, 200 rows, budget $1. Find the cheapest way and launch it."*

### MCP tools

| Tool | What it does |
|---|---|
| `compute_map` | Every machine/provider the agent can use and how busy each is right now |
| `run_workload` | Run any Python script (+ deps, + input files) on the best idle node or this machine |
| `read_output` | Read a finished job's text outputs |
| `estimate_job` | VRAM needed (weights/optimizer, activations, overhead) and estimated time on each GPU |
| `quote_job` | Every way to run it, ranked by what you actually pay, plus the remaining allowance |
| `launch_job` | Starts a LoRA fine-tune under a budget; refuses with a reason and cheaper options, or returns `pending_approval` |
| `job_status` | State, step, loss curve, tokens/s, cost so far, artifact path |
| `providers_status` | Which providers can launch right now, and what each is blocked on |
| `connect_provider` | **Self-serve setup.** Installs the client, starts sign-in, probes GPU access with a 5-second test call. If a human step remains (sign-in click, payment method), it opens that exact page in the user's browser and hands back one instruction, then verifies on the next call |
| `cancel_job`, `list_jobs`, `wallet` | Control and visibility |
| `free_inference_models` | Models OpenRouter serves at $0 right now: check this before fine-tuning at all |

There is deliberately **no MCP tool to approve jobs or change the policy**. Those are human-only (`gpuroute approve`, `gpuroute policy`, or the dashboard).

### Least-friction setup

The agent sets providers up itself. The human only does what legally has to be theirs:

```
you:    "Fine-tune Qwen2.5-1.5B on my data on a cloud GPU, budget $1"
agent:  connect_provider("modal")  → signed in ✓ · GPU probe → needs a payment method
        → opens modal.com/settings/<you>/usage in your browser
        "Add a payment method, then tell me. Est. $0.07 on an L4, from Modal's free credits; hard cap $1."
you:    "done"
agent:  connect_provider("modal") → GPU ✓ (Tesla T4) → launch_job(...) → job_status(...)
```

### Human commands

```
gpuroute setup [modal]               # provider readiness / connect one
gpuroute wallet                      # allowance this month
gpuroute policy --allowance 20 --approve-above 2 --max-job 5
gpuroute approve <job> | deny <job>  # jobs over the threshold wait for you
gpuroute log                         # append-only audit log of every spending decision
gpuroute jobs | cancel <job>
gpuroute free-models
```

## How the budget is enforced (three layers)

1. **Policy** (before launch): budget ≤ per-job cap, ≤ remaining monthly allowance, ≥ the estimate. Above the approval threshold, the job waits for a human.
2. **Infrastructure** (during the run): on Modal, the function timeout is `floor(budget / price_per_second)`, so the provider kills the container. Local runs stop themselves at their time cap.
3. **Settlement** (after): actual GPU seconds × price is written to the ledger; running paid jobs reserve their budget until they settle.

## Public data used

| Source | Used for | Terms |
|---|---|---|
| [Vast.ai public GPU pricing feed](https://vast.ai/pricing) (hourly JSON) | Live marketplace price + availability for 15 GPU types | Free for display and citation with attribution. gpuroute keeps only the latest snapshot (overwritten, never archived), always shows attribution and never redistributes. A production version would need Vast's data license |
| [OpenRouter models API](https://openrouter.ai/api/v1/models) | Which models are free to call right now | Public API |
| [Modal pricing](https://modal.com/pricing) | Per-second GPU list prices, free-credit allowance | Public page |
| NVIDIA datasheets | Dense bf16/fp16 TFLOPS + VRAM per GPU | Public |
| [Hugging Face Hub](https://huggingface.co) | Public training datasets and models | Per-dataset licenses |
| Kaggle / Colab docs | Free-tier quotas | Public docs |

## Responsible data use

- **Data locality is shown before anything runs.** Every launch states what leaves the device ("none: job runs on this machine", or "local file X (N rows) will be uploaded to Modal"), and it's recorded in the audit log.
- **Local-first.** When this machine can do the job, it's $0 and private, and it ranks first.
- **No credentials to agents.** The agent holds an allowance; provider credentials stay in the provider CLIs on the human's machine.
- **Dashboard binds to 127.0.0.1 only** and rejects cross-origin POSTs.

## How the estimates work

- VRAM = bytes/param by method (full 16 · LoRA 2 · QLoRA 0.56) × params + activations (gradient checkpointing; per-layer boundary tensors + one layer's working set + fp32 logits) × 1.15 + 1.5 GB overhead. On Apple MPS, LoRA weights are fp32 (4 bytes).
- Time = FLOPs/token (full 8 · LoRA/QLoRA 6, including recompute) × params × tokens ÷ (peak TFLOPS × MFU) + startup. MFU scales with model size (calibrated on a real 135M-param run on an M2: ~12%; 30% at ≥1B).
- Token count is an upper bound (assumes every row fills `seq_len`), so estimates err toward over-budgeting.

## Status / limitations

- Shared nodes: macOS and Linux; same-network (LAN/hotspot) connectivity with a bearer token. Jobs run as the owner's user, so share only with people you trust. Container sandboxing is next.
- Launch adapters: **local** (Apple MPS / CUDA), **shared nodes**, and **Modal**. Vast, Kaggle and Colab are quoted with how-to steps; launch adapters are next.
- Launch supports LoRA; estimate/quote also cover QLoRA and full fine-tuning.
- Modal needs a payment method on the account before it runs GPUs (usage is still covered by the $30/month free credits); `connect_provider` detects this and walks the user through it.

## Credits

Built with AI-assisted development (Claude Code). Uses `transformers`, `peft`, `torch`, `datasets`, the MCP Python SDK, `rich`, `httpx`, and Modal.
