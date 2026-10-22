# Demo script: 5 minutes total (≈2 min slides, ≈2 min live, ≈1 min Q&A)

## Before you go up (do this 10 minutes early)
1. **Iana's laptop joins the pool.** On your Mac, run `cd ~/national-renaissance-hackathon/gpuroute && .venv/bin/gpuroute hub`, copy the `uvx … share --pool "…"` line it prints, send it to Iana, and have her run it with `--name iana`.
   Then run `.venv/bin/gpuroute run examples/benchmark.py --on iana` once, so torch is already cached on her machine (the first run downloads ~1–2 min).
2. **Clean up the test nodes:** `pkill -f "gpuroute share"` (stops the simulated `friend-pc`), then `.venv/bin/gpuroute nodes rm iana-sim`.
3. **Open, in this order:**
   - the deck (claude.ai, Present mode)
   - Terminal A: `claude` started in `~/national-renaissance-hackathon` (gpuroute is connected)
   - Iana's `share` terminal visible (her screen next to yours)
   - browser tab: `http://127.0.0.1:7311/#map` (run `.venv/bin/gpuroute ui` if it's not up)
4. **Warm-up check:** in Claude Code ask "what compute can you use right now?". It should list local, iana (pool node), Modal and the satellites.

## Script

**0:00 – Slide 1 (cover), 10s**
"We're Sissi and Iana. gpuroute puts spare compute to work for AI agents."

**0:10 – Slide 2 (idle), 20s**
"Compute already exists and sits idle. One university cluster averaged 35% utilization over a year. Right now there are about 1,600 spare GPUs for rent on Vast alone. And every team has idle laptops and gaming PCs."

**0:30 – Slide 3 (blind), 15s**
"But an agent only knows the laptop it's running on. Everything else is invisible to it, and you can't safely hand it your cloud account."

**0:45 – Slide 4 (pains), 30s**
"We spent the morning on research, not guessing. Six real pains: nobody knows what GPU a job needs; every provider's agent tool covers only itself; there's no spend cap for agents; forgotten GPUs keep billing; most researchers have no budget at all; and long jobs outlive agent sessions. The right column is what we built for each."

**1:15 – Slide 6 (how it works), 15s**
"The agent calls our MCP tools, our router picks an idle target and enforces the budget, and the job runs on your machine, a teammate's machine, Modal, or the orbital market."

**1:30 – LIVE (Claude Code), about 2 minutes**

*Moment 1: a teammate's spare laptop (the core story)*
Type:
> Build a search index of my notes at /Users/sissi/national-renaissance-hackathon/MARKET_RESEARCH.md using gpuroute/examples/embed.py. Don't hog my laptop if there's spare compute.

While it runs: "It checked the compute map, saw Iana's laptop is idle, and sent the job there." **Point at Iana's terminal:** the job appears with who sent it and what it installs. "Iana stays in control: idle-only, time cap, Ctrl-C stops it." The results come back to my machine.

*Moment 2: picking real hardware*
Type:
> Now run gpuroute/examples/benchmark.py on the strongest GPU you can get, under $1.

It picks Modal H100: ~760 TFLOPS for well under a cent. "The $1 budget became Modal's hard timeout. Even if the agent dies, it can't overspend."

*Moment 3 (only if time remains): satellite*
Type:
> Run the same benchmark on satellite compute.

Say it honestly: "Orbital compute exists (Starcloud-1 has an H100 in orbit) but nobody sells it by API yet. The orbits, sunlight windows and ground-station passes are real and live; the listings are mock, and the job runs on a ground H100. The agent tells you that itself."

**3:30 – Browser, Map tab, 20s**
Click Starcloud-1 (live position, next downlink), then type "Frankfurt" in search. "Agents get this same map through world_compute, for example to keep data in Germany."

**3:50 – Slide 8 (proof) + close, 15s**
"Everything in this table ran today. Put spare compute to work."

**4:05 – Q&A**

## What's real vs not (say this if asked, don't oversell)
| Real, tested today | Not real yet |
|---|---|
| This Mac's GPU, pool nodes over the internet, Modal T4→B200 (H100 verified) | Orbital compute purchase: listings + prices are mock, jobs run on a ground H100 |
| Budget caps (Modal timeout, local watchdog), approvals, audit log | Vast.ai / Kaggle / Colab: quoted only, no launch adapter yet |
| Live public data: ATLAS, Epoch AI, TeleGeography, CelesTrak, Vast feed, OpenRouter | Pool nodes run jobs as the owner's user: share only with people you trust (container sandboxing is next) |
| A real agent made every choice in the proof table | Satellite downlink is a scheduled time, not an actual transfer |
| | The pool has only been tested between two processes on one Mac until Iana joins |

## Likely questions
- **"Why not SkyPilot or Shadeform?"** They provision cloud accounts you already pay for. We add machines you already own, free tiers, a spend cap agents can't raise, and an agent-native interface.
- **"Isn't running someone's code on my laptop risky?"** Today pools are for people you trust: a pool secret, the owner sees every job, idle-only, time caps. Containers are next.
- **"Is the satellite part real?"** The hardware and orbits are real; the marketplace is a working prototype waiting for an operator API.

## If something breaks
- **Iana's laptop can't connect:** the pool needs only outbound HTTPS, so try her phone hotspot. Otherwise skip Moment 1 and do Moment 2.
- **Modal is slow to start:** a cold start can take ~30s; keep talking over slide 6.
- **Wifi dies:** `gpuroute run examples/benchmark.py --on local` runs on this Mac offline (torch and models are cached).
