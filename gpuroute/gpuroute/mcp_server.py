"""MCP server: lets any agent (Claude Code, Codex, Cursor) size, price and run
training jobs under a spending allowance it cannot raise itself.

There is deliberately no tool to approve jobs or change the policy. Those are
human-only actions in the CLI (`gpuroute approve`, `gpuroute policy`).
"""
from mcp.server.mcpserver import MCPServer

from . import data, jobs, ledger
from .estimate import estimate
from .quote import quote

mcp = MCPServer("gpuroute", instructions=(
    "Use gpuroute whenever a task needs GPU compute for training or fine-tuning. "
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
