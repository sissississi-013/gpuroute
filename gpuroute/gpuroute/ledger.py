"""The agent's allowance: spending policy, an append-only audit log, and job records.

The agent never holds cloud credentials. It holds an allowance. Every launch
reserves money against it, and anything above the approval threshold waits for
a human to approve it from their own terminal.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home() / ".gpuroute"
CONFIG = HOME / "config.json"
LEDGER = HOME / "ledger.jsonl"
JOBS = HOME / "jobs"

DEFAULT_CONFIG = {
    "monthly_allowance_usd": 10.0,   # hard cap on what agents may spend per month
    "approval_threshold_usd": 2.0,   # launches above this need a human `gpuroute approve`
    "max_job_usd": 5.0,              # no single job may reserve more than this
}


def config():
    HOME.mkdir(parents=True, exist_ok=True)
    if not CONFIG.exists():
        CONFIG.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
    return {**DEFAULT_CONFIG, **json.loads(CONFIG.read_text())}


def set_config(**kw):
    cfg = config()
    cfg.update({k: v for k, v in kw.items() if v is not None})
    CONFIG.write_text(json.dumps(cfg, indent=2))
    record("policy_changed", **{k: v for k, v in kw.items() if v is not None})
    return cfg


def record(event, **data):
    HOME.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, **data}
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def entries(limit=None):
    if not LEDGER.exists():
        return []
    lines = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    return lines[-limit:] if limit else lines


def save_job(job):
    JOBS.mkdir(parents=True, exist_ok=True)
    (JOBS / job["id"] / "job.json").parent.mkdir(exist_ok=True)
    (JOBS / job["id"] / "job.json").write_text(json.dumps(job, indent=2))


def load_job(job_id):
    path = JOBS / job_id / "job.json"
    if not path.exists():
        raise KeyError(f"no job {job_id}")
    return json.loads(path.read_text())


def all_jobs():
    if not JOBS.exists():
        return []
    jobs = [json.loads(p.read_text()) for p in JOBS.glob("*/job.json")]
    return sorted(jobs, key=lambda j: j["created"], reverse=True)


def wallet():
    """Allowance left this month = cap - settled spend - money reserved by running jobs."""
    cfg = config()
    month = time.strftime("%Y-%m")
    spent = reserved = 0.0
    for j in all_jobs():
        if not time.strftime("%Y-%m", time.localtime(j["created"])) == month:
            continue
        if j["state"] in ("pending_approval", "denied"):
            continue
        if j.get("actual_cost_usd") is not None:
            spent += j["actual_cost_usd"]
        elif j["state"] in ("launching", "running") and j.get("price_per_s"):
            reserved += j["budget_usd"]  # free (local) jobs reserve nothing
    left = cfg["monthly_allowance_usd"] - spent - reserved
    return {"month": month, "allowance_usd": cfg["monthly_allowance_usd"],
            "spent_usd": round(spent, 4), "reserved_usd": round(reserved, 4),
            "available_usd": round(left, 4),
            "approval_threshold_usd": cfg["approval_threshold_usd"],
            "max_job_usd": cfg["max_job_usd"]}
