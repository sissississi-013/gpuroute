# 5-minute demo script

## Setup (before presenting)
**Iana's laptop** (see SHARE.md):
```bash
uvx --from "git+https://github.com/sissississi-013/gpuroute#subdirectory=gpuroute" gpuroute share --name iana
```
Copy the `gpuroute join iana@…` line it prints.

**This Mac:**
```bash
cd ~/national-renaissance-hackathon/gpuroute
.venv/bin/gpuroute join "iana@<ip>:7400#<token>"                 # paste from Iana (keep the quotes)
.venv/bin/gpuroute map                                            # iana should say usable: yes
.venv/bin/gpuroute run examples/benchmark.py --on iana              # warm-up: caches torch on Iana's machine
.venv/bin/gpuroute nodes rm iana-sim                              # drop the simulated node
.venv/bin/gpuroute ui                                             # dashboard tab
cd .. && claude                                                   # Claude Code with gpuroute (.mcp.json here)
```
Warm-up matters: the first run on Iana's machine downloads torch (~1–2 min). Also run `examples/embed.py` once there to cache sentence-transformers.

## Script
**0:00 Slides 1–3 (60s).** "The cheapest GPU is the one you already have and aren't using." 35% cluster utilization, 1,650 spare GPUs right now, idle laptops everywhere. Agents can't see any of it.

**1:00 Live (2 min).** Iana's terminal visible. In Claude Code:
> I have notes at /Users/sissi/national-renaissance-hackathon/MARKET_RESEARCH.md. Build a semantic search index of them on a GPU using gpuroute/examples/embed.py. Don't hog my laptop if there's spare compute.

Narrate: compute_map → sees iana idle → run_workload ships script + file → **Iana's terminal prints the job** → results come back → agent reads demo_search.json via read_output. Show the dashboard: compute map, job, audit log.

**3:00 Owner control (30s).** Iana starts something heavy (or Ctrl-C): her node shows "not accepting" in the map, and the agent falls back to this machine. "The owner is always in control."

**3:30 Slides: budget + setup + data (45s).** Paid compute: hard cap = provider timeout, approval above $2, connect_provider makes setup one click.

**4:15 Market + close (45s).**

## Fallback if Iana's machine can't connect (venue wifi isolation)
Both on a phone hotspot. If still no luck, simulate on this Mac and say so:
```bash
.venv/bin/gpuroute share --name iana-sim --port 7400 --always &   # prints the add line
```
