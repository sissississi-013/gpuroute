"""MCP server: lets any agent (Claude Code, Codex, Cursor) size, price and run
training jobs under a spending allowance it cannot raise itself.

There is deliberately no tool to approve jobs or change the policy. Those are
human-only actions in the CLI (`gpuroute approve`, `gpuroute policy`).
"""
from mcp.server.mcpserver import MCPServer

from . import data, jobs, ledger, providers
from .estimate import estimate
from .quote import quote

mcp = MCPServer("gpuroute", instructions=(
    "Use gpuroute whenever a task needs GPU or heavy compute: training, fine-tuning, transcription, "
    "embeddings, image/video generation, rendering, batch inference, and orbital compute. world_compute answers "
    "questions about where compute physically is (datacenters, frontier AI sites, undersea, orbit) (orbital_offers / "
    "book_orbital). Call compute_map first to see every "
    "machine you can use (including teammates' idle GPUs), use run_workload for arbitrary scripts and "
    "quote_job/launch_job for LoRA fine-tunes. "
    "If a provider isn't ready, call connect_provider: it does the setup itself and only hands the user "
    "the one step that needs a human (sign-in or payment method), already opened in their browser. "
    "Always call quote_job before launch_job, pick a budget that covers the estimate, and "
    "prefer $0 options (local machine, free credits). Never assume a job finished: poll "
    "job_status. If a launch returns pending_approval, tell the user the exact approve command."
))


@mcp.tool()
def estimate_job(model: str, method: str = "lora", examples: int = 1000, seq_len: int = 512,
                 epochs: int = 1, batch: int = 8, params_b: float | None = None) -> dict:
    """Estimate VRAM and runtime for a fine-tuning job before renting anything.

    model: Hugging Face model id (e.g. "Qwen/Qwen2.5-0.5B-Instruct"). For unknown models pass params_b.
    method: "lora", "qlora" or "full".
    examples/seq_len/epochs/batch: dataset rows, max tokens per row, passes over the data, batch size.
    Returns VRAM needed (with breakdown), whether each GPU fits, and estimated seconds per GPU.
    """
    return estimate(model, method, examples, seq_len, epochs, batch, params_b)


@mcp.tool()
def quote_job(model: str, method: str = "lora", examples: int = 1000, seq_len: int = 512,
              epochs: int = 1, budget_usd: float | None = None, params_b: float | None = None) -> dict:
    """Rank every way to run a job, cheapest first: this machine, free tiers (Kaggle, Colab),
    Modal (free credits, then list price), and live Vast.ai marketplace prices.

    Each option has est_minutes, cost_usd, out_of_pocket_usd and launchable (whether
    launch_job can start it). Also returns the allowance left in your wallet.
    """
    q = quote(model, method, examples, seq_len, epochs, params_b=params_b, budget_usd=budget_usd)
    est = q["estimate"]
    return {"vram_needed_gb": est["vram_needed_gb"], "tokens": est["tokens"],
            "assumptions": est["assumptions"], "options": q["options"][:12],
            "recommended_launch": q["recommended_launch"], "notes": q["notes"],
            "wallet": ledger.wallet()}


@mcp.tool()
def launch_job(model: str, dataset: str, budget_usd: float, examples: int = 1000, seq_len: int = 512,
               epochs: int = 1, provider: str = "auto", gpu: str | None = None,
               max_minutes: float | None = None) -> dict:
    """Start a LoRA fine-tune with a hard spending cap.

    dataset: a Hugging Face dataset id with a "text" column (e.g. "mlabonne/guanaco-llama2-1k"),
             or an absolute path to a local .jsonl file with {"text": ...} rows.
    budget_usd: the most this job may cost. On Modal, the provider kills the container when
                runtime reaches budget / price, so the cap holds even if you stop polling.
    provider: "auto" (cheapest launchable), "local" ($0, data stays on this machine) or "modal".
    Launches above the user's approval threshold return state "pending_approval"; relay the
    approve command to the user. Refusals explain why and list cheaper options.
    """
    return jobs.launch(model, dataset, budget_usd, examples=examples, seq_len=seq_len, epochs=epochs,
                       provider=provider, gpu=gpu, max_minutes=max_minutes, requested_by="agent:mcp")


@mcp.tool()
def compute_map() -> list:
    """Every compute resource you can use, and how busy each is right now: this machine, shared
    nodes (teammates' idle GPUs, lab boxes, gaming PCs running `gpuroute share`), cloud providers,
    the live spare-GPU market, and free tiers. Call this before assuming you only have this machine."""
    from . import pool
    return pool.compute_map()


@mcp.tool()
def run_workload(script: str, deps: list[str] = [], args: list[str] = [], files: list[str] = [],
                 needs_gpu: bool = True, min_vram_gb: float | None = None, max_minutes: float = 15,
                 target: str = "auto", budget_usd: float = 1.0) -> dict:
    """Run ANY Python workload (transcription, embeddings, image generation, rendering, batch
    inference, benchmarks, training) on the best available compute: an idle shared node first,
    then this machine. The node owner's rules (idle-only, minutes cap) are enforced on their side.

    script: absolute path to a .py file, or inline Python. It should write results into ./out
            (also $GPUROUTE_OUT); they are copied back here when the job ends.
    deps: pip packages the script needs. Leave empty to infer them from the script's imports.
    files: absolute paths of local input files to send along (≤50 MB total); they sit next to the script.
    target: "auto" (idle shared node, then this machine), "local", a node name from compute_map, or a
            cloud GPU: "modal:T4", "modal:L4", "modal:A10", "modal:L40S", "modal:A100-80GB", "modal:H100",
            "modal:H200", "modal:B200". Pick the GPU the workload needs; quote_job helps size training.
    budget_usd: hard cap for cloud GPUs (becomes the provider's timeout); above the user's threshold it
                waits for human approval. Free targets ignore it.
    """
    return jobs.run_workload(script, deps, args, files, needs_gpu=needs_gpu, min_vram_gb=min_vram_gb,
                             max_minutes=max_minutes, target=target, requested_by="agent:mcp",
                             budget_usd=budget_usd)


@mcp.tool()
def read_output(job_id: str, filename: str, max_chars: int = 20000) -> dict:
    """Read a text output (json, txt, csv, md) a finished job wrote to ./out. For binary files
    (e.g. .npy) returns the path and size instead."""
    from pathlib import Path
    out = ledger.JOBS / job_id / "out"
    f = out / Path(filename).name
    if not f.exists():
        return {"error": f"no output {filename!r}", "available": sorted(p.name for p in out.glob("*"))
                if out.exists() else []}
    if f.suffix.lower() in (".json", ".txt", ".csv", ".md", ".log", ".jsonl", ".srt", ".vtt"):
        text = f.read_text(errors="replace")
        return {"file": str(f), "bytes": f.stat().st_size, "content": text[:max_chars],
                "truncated": len(text) > max_chars}
    return {"file": str(f), "bytes": f.stat().st_size, "note": "binary file; use the path"}


@mcp.tool()
def world_compute(layer: str = "all", near: str | None = None, lat: float | None = None, lon: float | None = None,
                  radius_km: float = 500, search: str | None = None, owner: str | None = None,
                  country: str | None = None, limit: int = 15) -> dict:
    """Query the world compute map (the same data as the dashboard's Map tab): where compute physically is.
    layer: "all" | "datacenters" (6,131 facilities, ATLAS) | "frontier" (frontier AI sites with power in MW
    and owner, Epoch AI) | "undersea" (underwater datacenters) | "cables" (undersea internet cables) |
    "orbit" (satellites with compute: live position, sunlight windows, downlink passes).
    Filter by near (a city like "virginia", "frankfurt", "singapore") or lat/lon + radius_km, by search
    text, owner/operator, or country. Each layer says whether agents can actually use it: use this for
    questions like "where is GPU capacity near Frankfurt?", data-residency choices, or "what's in orbit now?"."""
    from . import world
    return world.query(layer, near, lat, lon, radius_km, search, owner, country, limit)


@mcp.tool()
def orbital_offers(hours: float = 12) -> dict:
    """Satellite compute marketplace: GPU time aboard satellites (only while sunlit, for solar power) and
    ground-station relay passes to downlink results, from different operators. Orbits, sunlight and passes
    are computed live from real orbital elements; operator listings and prices are mock (no operator sells
    this by API yet), so booked jobs are emulated on ground GPUs following the satellite's real timeline."""
    from . import orbital
    return orbital.offers(hours)


@mcp.tool()
def book_orbital(offer_id: str, script: str, deps: list[str] = [], args: list[str] = [],
                 files: list[str] = [], budget_usd: float = 1.0, max_minutes: float = 10) -> dict:
    """Book a compute offer from orbital_offers (type "compute", open now or within 20 min) and run a
    Python script in that satellite's window. Returns the booking (window, mock price, downlink pass)
    and a job_id to poll with job_status. Same script contract as run_workload."""
    from . import orbital
    return orbital.book(offer_id, script, deps, args, files, budget_usd, max_minutes, requested_by="agent:mcp")


@mcp.tool()
def providers_status() -> dict:
    """Which providers can launch jobs right now (local machine, Modal) and what each is blocked on."""
    return providers.status()


@mcp.tool()
def connect_provider(provider: str = "modal") -> dict:
    """Set up a provider so launch_job can use it. Does every automatable step itself: installs the
    client, starts sign-in, and verifies GPU access with a ~5-second test call (~$0.001 of free credits).
    If a human step remains (sign-in click or adding a payment method), it opens that exact page in the
    user's browser and returns human_action with a one-line instruction. Relay it, wait for the user
    to confirm, then call this again to verify."""
    return providers.connect(provider)


@mcp.tool()
def job_status(job_id: str) -> dict:
    """State, progress (step, loss, tokens/s), loss curve, cost so far, and the artifact path of a job.
    States: pending_approval, running, completed, stopped_at_budget, failed, cancelled, denied."""
    return jobs.status(job_id)


@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """Stop a running job now and settle what it cost."""
    return jobs.cancel(job_id)


@mcp.tool()
def list_jobs(limit: int = 10) -> list:
    """Recent jobs with their state and cost."""
    return [jobs._public(j) for j in ledger.all_jobs()[:limit]]


@mcp.tool()
def wallet() -> dict:
    """The agent's allowance this month: cap, spent, reserved by running jobs, available,
    and the threshold above which a human must approve."""
    return ledger.wallet()


@mcp.tool()
def free_inference_models(limit: int = 10) -> dict:
    """Models OpenRouter serves at $0 right now. Check this before training: if the task only
    needs inference (classify, summarize, extract), a free hosted model may beat fine-tuning."""
    models, total, _ = data.free_models(limit)
    return {"free_models_total": total, "models": models, "source": data.SOURCES["openrouter"]["attribution"]}


def main():
    mcp.run()


if __name__ == "__main__":
    main()
