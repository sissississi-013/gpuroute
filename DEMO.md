# 5-minute demo script

## Before you present (setup, ~2 min)
```bash
cd ~/national-renaissance-hackathon/gpuroute
.venv/bin/gpuroute ui                  # dashboard opens at http://127.0.0.1:7311, keep it in a browser tab
# Claude Code with gpuroute (in a second terminal):
cd ~/national-renaissance-hackathon && claude    # .mcp.json in this folder loads gpuroute
```
Optional: fresh dashboard for the demo: `mv ~/.gpuroute/jobs ~/.gpuroute/jobs.bak && mv ~/.gpuroute/ledger.jsonl ~/.gpuroute/ledger.bak`

## Script
**0:00 Slides 1–2 (45s).** "Agents write training code in seconds. When it needs a GPU: which one, which cloud, and would you give it your AWS keys?"

**0:45 Slide 3 (20s).** Four tools: estimate, quote, launch, status.

**1:05 Live, in Claude Code (2 min).** Type:
> Fine-tune HuggingFaceTB/SmolLM2-360M-Instruct on mlabonne/guanaco-llama2-1k, 200 rows, seq len 256. Budget $1. Find the cheapest way and launch it.

While it runs, narrate: it called quote_job (13 options ranked), picked this Mac at $0, launched with a $1 cap, and says what data leaves the device (none).

Switch to the dashboard: running job, progress bar, loss curve filling in, audit log.

**3:05 Guardrails (40s).** In Claude Code:
> Now fine-tune Qwen/Qwen2.5-1.5B-Instruct on 1000 rows on a cloud GPU, budget $3.

It hits the $2 approval threshold, and the job appears under **Needs your approval** on the dashboard. Click **Approve** or **Deny**. "The agent can't approve itself: there's no MCP tool for that." (If Modal has no card on file, this also shows connect_provider opening the billing page: one human step.)

**3:45 Slides 6–8 (45s).** Three-layer budget, public data + responsible use, market gap.

**4:30 Close (30s).** "Let agents train models. Keep the wallet." Q&A.

## Backup if wifi dies
- Models already cached: SmolLM2-135M, SmolLM2-360M, Qwen2.5-0.5B
- Quotes fall back to the cached Vast/OpenRouter snapshot
- CLI only: `gpuroute quote HuggingFaceTB/SmolLM2-360M-Instruct --examples 200 --seq-len 256 --budget 1`, then `gpuroute launch HuggingFaceTB/SmolLM2-360M-Instruct mlabonne/guanaco-llama2-1k --budget 1 --examples 200 --seq-len 256`, then `gpuroute status <id> --watch`
