"""Launch, track, cancel and approve jobs.

Budget enforcement happens in three layers:
  1. policy: the job's budget must fit the allowance, the per-job cap, and the estimate
  2. infrastructure: on Modal, the function's hard timeout is budget / price, so the
     provider kills the container, not the agent behaving well. Locally, the runner
     stops itself at its time cap.
  3. settlement: the actual cost is computed from GPU runtime and written to the ledger
"""
import base64
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from . import ledger
from .catalog import MODAL, local_device
from .quote import quote

TEMPLATES = Path(__file__).parent / "templates"
TERMINAL = {"completed", "failed", "cancelled", "stopped_at_budget", "denied"}


def launch(model_id, dataset, budget_usd=0.0, method="lora", examples=1000, seq_len=512, epochs=1,
           provider="auto", gpu=None, max_minutes=None, params_b=None, requested_by="agent"):
    """provider: 'auto' (cheapest launchable option), 'local' or 'modal'."""
    if method != "lora":
        return _refuse(model_id, budget_usd, requested_by,
                       "launch supports method='lora' only for now (estimate/quote also cover full and qlora)")
    if dataset.endswith(".jsonl"):
        if not Path(dataset).expanduser().exists():
            return _refuse(model_id, budget_usd, requested_by, f"dataset file not found: {dataset}")
        dataset = str(Path(dataset).expanduser().resolve())
    cfg, wal = ledger.config(), ledger.wallet()
    q = quote(model_id, method, examples, seq_len, epochs, params_b=params_b, budget_usd=budget_usd)
    launchable = [o for o in q["options"] if o["launchable"]]
    if provider != "auto":
        launchable = [o for o in launchable if o["provider"] == provider]
    if gpu:
        launchable = [o for o in launchable if o["gpu"] == gpu]
    if max_minutes:
        launchable = [o for o in launchable if o["est_minutes"] <= max_minutes] or launchable
    opt = launchable[0] if launchable else None  # already ranked cheapest-first

    if opt is None and provider in ("modal", "auto"):
        from .providers import known
        m = known("modal")
        if m.get("ready") is False:
            fits = any(o["provider"] == "modal" for o in q["options"])
            if provider == "modal" or fits:
                return _refuse(model_id, budget_usd, requested_by,
                               f"Modal isn't set up yet (blocked on {m.get('blocked_on')}). Call "
                               f"connect_provider('modal'): it opens the one page the user needs, then "
                               f"verifies. Or use provider='local' if this machine fits the job.", q)
    if opt is None:
        return _refuse(model_id, budget_usd, requested_by,
                       f"no launchable option fits this job (needs ~{q['estimate']['vram_needed_gb']} GB VRAM"
                       f"{', provider=' + provider if provider != 'auto' else ''}"
                       f"{', gpu=' + gpu if gpu else ''})", q)
    if budget_usd > cfg["max_job_usd"]:
        return _refuse(model_id, budget_usd, requested_by,
                       f"budget ${budget_usd} exceeds the per-job cap of ${cfg['max_job_usd']}", q)
    if opt["provider"] != "local" and budget_usd > wal["available_usd"]:
        return _refuse(model_id, budget_usd, requested_by,
                       f"budget ${budget_usd} exceeds the remaining allowance of ${wal['available_usd']}", q)
    if opt["cost_usd"] > budget_usd:
        return _refuse(model_id, budget_usd, requested_by,
                       f"estimated cost ${opt['cost_usd']:.2f} on {opt['provider']} {opt['gpu']} exceeds "
                       f"budget ${budget_usd}; raise the budget, use provider='local', or shrink the job", q)

    job_id = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    job = {
        "id": job_id, "created": time.time(), "provider": opt["provider"], "gpu": opt["gpu"],
        "budget_usd": budget_usd, "est_cost_usd": opt["cost_usd"], "est_minutes": opt["est_minutes"],
        "model": model_id, "dataset": dataset, "method": method, "examples": examples,
        "seq_len": seq_len, "epochs": epochs, "requested_by": requested_by,
        "data_leaving_device": _data_leaving_device(dataset, opt["provider"]),
        "state": "launching", "actual_cost_usd": None,
    }
    if opt["provider"] == "modal":
        modal_gpu, per_s = MODAL[opt["gpu"]]
        job.update(modal_gpu=modal_gpu, price_per_s=per_s,
                   hard_timeout_s=max(60, min(86400, math.floor(budget_usd / per_s))))
    else:
        job.update(price_per_s=0.0,
                   hard_timeout_s=int((max_minutes or max(10, opt["est_minutes"] * 2)) * 60))

    if job["price_per_s"] > 0 and budget_usd > cfg["approval_threshold_usd"]:
        job["state"] = "pending_approval"
        ledger.save_job(job)
        ledger.record("approval_requested", job=job_id, budget_usd=budget_usd, provider=job["provider"],
                      gpu=job["gpu"], requested_by=requested_by)
        return {"ok": True, "job_id": job_id, "state": "pending_approval", "job": _public(job),
                "message": f"Budget ${budget_usd} is above the ${cfg['approval_threshold_usd']} approval "
                           f"threshold. Ask the user to run: gpuroute approve {job_id}"}

    ledger.save_job(job)
    _start(job)
    return {"ok": True, "job_id": job_id, "state": job["state"], "job": _public(job),
            "message": _launch_message(job)}


def approve(job_id, approve=True):
    job = ledger.load_job(job_id)
    if job["state"] != "pending_approval":
        return {"ok": False, "error": f"job is {job['state']}, not pending approval"}
    if not approve:
        job["state"] = "denied"
        ledger.save_job(job)
        ledger.record("denied", job=job_id, by="human")
        return {"ok": True, "state": "denied"}
    if job["budget_usd"] > ledger.wallet()["available_usd"]:
        return {"ok": False, "error": "not enough allowance left to approve this job"}
    ledger.record("approved", job=job_id, budget_usd=job["budget_usd"], by="human")
    job["state"] = "launching"
    ledger.save_job(job)
    _start(job)
    return {"ok": True, "state": job["state"], "job": _public(job), "message": _launch_message(job)}


def _launch_message(job):
    if job["provider"] == "modal":
        return (f"Launched on Modal {job['gpu']}. Modal will hard-stop it after {job['hard_timeout_s']}s "
                f"(= the ${job['budget_usd']} budget). Poll job_status('{job['id']}').")
    cap = job["hard_timeout_s"]
    cap_txt = f"{cap}s" if cap < 120 else f"{cap // 60} min"
    return (f"Launched locally on {job['gpu']} at $0. It stops itself after {cap_txt}. "
            f"Poll job_status('{job['id']}').")


def _start(job):
    d = ledger.JOBS / job["id"]
    d.mkdir(parents=True, exist_ok=True)
    if job["provider"] == "modal":
        app_name = f"gpuroute-{job['id']}"
        src = (TEMPLATES / "modal_lora.py").read_text()
        src = (src.replace("__APP_NAME__", app_name).replace("__GPU__", job["modal_gpu"])
               .replace("__TIMEOUT_S__", str(job["hard_timeout_s"])))
        (d / "app.py").write_text(src)
        shutil.copy(TEMPLATES / "train_core.py", d / "train_core.py")
        cmd = [shutil.which("modal") or "modal", "run", "--detach", str(d / "app.py"),
               "--model-id", job["model"], "--dataset", job["dataset"], "--rows", str(job["examples"]),
               "--seq-len", str(job["seq_len"]), "--epochs", str(job["epochs"]), "--job-id", job["id"]]
        job["app_name"] = app_name
        job["artifact"] = f"modal volume gpuroute-artifacts:/{job['id']}"
    else:
        spec = {"device": local_device()["device"], "model": job["model"], "dataset": job["dataset"],
                "examples": job["examples"], "seq_len": job["seq_len"], "epochs": job["epochs"],
                "max_seconds": job["hard_timeout_s"], "out_dir": str(d / "adapter")}
        cmd = [sys.executable, "-m", "gpuroute.local_run", json.dumps(spec)]
        job["artifact"] = str(d / "adapter")
    log = open(d / "run.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    job.update(pid=proc.pid, started=time.time(), state="running")
    ledger.save_job(job)
    ledger.record("launched", job=job["id"], provider=job["provider"], gpu=job["gpu"],
                  budget_usd=job["budget_usd"], hard_timeout_s=job["hard_timeout_s"],
                  est_cost_usd=job["est_cost_usd"], data_leaving_device=job["data_leaving_device"],
                  requested_by=job["requested_by"])


def status(job_id):
    job = ledger.load_job(job_id)
    if job["state"] in TERMINAL or job["state"] == "pending_approval":
        return {**_public(job), **({"outputs": job["outputs"]} if job.get("outputs") else {})}
    if job.get("kind") == "script":
        return _script_status(job)
    log_path = ledger.JOBS / job_id / "run.log"
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    events = _events(text)
    first = lambda name: next((e for e in events if e["event"] == name), None)  # noqa: E731
    started, finished = first("started"), first("finished")
    stopped, error = first("stopped_at_budget"), first("error")
    progress = [e for e in events if e["event"] == "progress"]
    alive = _alive(job.get("pid"))
    if alive and job["provider"] == "local" and time.time() - job["started"] > job["hard_timeout_s"] + 60:
        try:  # the runner's own watchdog should have fired; enforce the cap from outside too
            os.killpg(job["pid"], signal.SIGKILL)
        except OSError:
            pass
        alive, stopped = False, {"t": time.time()}

    end =(finished or stopped or {}).get("t") or time.time()
    gpu_seconds = max(0.0, end - started["t"]) if started else 0.0
    job["gpu_seconds"] = round(gpu_seconds, 1)
    job["cost_so_far_usd"] = round(gpu_seconds * job["price_per_s"], 4)
    if started:
        job["gpu_name"] = started.get("gpu")
    if progress:
        job["progress"] = {k: progress[-1][k] for k in ("step", "steps", "loss", "tok_per_s")}
        job["loss_curve"] = [p["loss"] for p in progress]
    if not started:
        job["phase"] = "building image / starting container" if job["provider"] == "modal" else "loading"
    elif not progress:
        job["phase"] = "loading model and data"
    else:
        job.pop("phase", None)

    if finished:
        _settle(job, "completed", gpu_seconds)
    elif stopped:
        _settle(job, "stopped_at_budget", gpu_seconds)
    elif not alive:
        lowered = text.lower()
        if "payment method" in lowered:
            from .providers import _save
            _save("modal", ready=False, blocked_on="payment_method")
            _settle(job, "failed", gpu_seconds,
                    error="Modal needs a payment method before it will run GPU functions (usage is still "
                          "covered by the $30/mo free credits). Add one at https://modal.com/settings "
                          "-> Usage and Billing (or call connect_provider('modal')), or relaunch with provider='local'.")
        elif "timeout" in lowered or "timed out" in lowered:
            _settle(job, "stopped_at_budget", min(gpu_seconds, job["hard_timeout_s"]))
        else:
            _settle(job, "failed", gpu_seconds, error=(error or {}).get("error") or _tail(text))
    ledger.save_job(job)
    out = _public(job)
    if job["state"] == "running":
        out["log_tail"] = _tail(text, 4)
    return out


def cancel(job_id):
    job = ledger.load_job(job_id)
    if job["state"] in TERMINAL:
        return {"ok": False, "error": f"job already {job['state']}"}
    if job["state"] == "pending_approval":
        job["state"] = "cancelled"
        ledger.save_job(job)
        ledger.record("cancelled", job=job_id)
        return {"ok": True, "state": "cancelled"}
    if job["provider"].startswith("node:"):
        from . import pool
        try:
            pool.remote_cancel(job["node"], job_id)
        except Exception:
            pass
    if job["provider"] == "modal":
        subprocess.run([shutil.which("modal") or "modal", "app", "stop", job["app_name"]],
                       capture_output=True, timeout=60)
    if _alive(job.get("pid")):
        try:
            os.killpg(job["pid"], signal.SIGTERM)
        except OSError:
            pass
        time.sleep(1)
    status(job_id)
    job = ledger.load_job(job_id)
    if job["state"] not in TERMINAL:
        _settle(job, "cancelled", job.get("gpu_seconds", 0))
        ledger.save_job(job)
    return {"ok": True, **_public(job)}


def _refuse(model_id, budget_usd, requested_by, reason, q=None):
    ledger.record("launch_refused", model=model_id, budget_usd=budget_usd, reason=reason,
                  requested_by=requested_by)
    out = {"ok": False, "error": reason}
    if q:
        out["cheapest_options"] = q["options"][:5]
    return out


def _settle(job, state, gpu_seconds, **extra):
    job["state"] = state
    job["actual_cost_usd"] = round(gpu_seconds * job["price_per_s"], 4)
    job["finished"] = time.time()
    job.update(extra)
    job.pop("phase", None)
    ledger.record("settled", job=job["id"], state=state, actual_cost_usd=job["actual_cost_usd"],
                  budget_usd=job["budget_usd"], gpu_seconds=round(gpu_seconds, 1))


def _events(text):
    out = []
    for line in text.splitlines():
        i = line.find("GPUROUTE {")
        if i >= 0:
            try:
                out.append(json.loads(line[i + 9:]))
            except json.JSONDecodeError:
                pass
    return out


def _alive(pid):
    if not pid:
        return False
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)  # reap if it's our own child
        if done:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _tail(text, n=8):
    lines = [l for l in text.splitlines()
             if l.strip() and "GPUROUTE {" not in l and "━" not in l and "Downloading" not in l]
    return "\n".join(lines[-n:])


def _data_leaving_device(dataset, provider):
    is_local = dataset.endswith(".jsonl") and Path(dataset).exists()
    if provider == "local":
        return "none: job runs on this machine" + (f" using {dataset}" if is_local else "")
    if is_local:
        rows = sum(1 for l in open(dataset) if l.strip())
        return f"local file {dataset} ({rows} rows) will be uploaded to Modal"
    return f"none: Modal downloads the public dataset {dataset} itself"


def _public(job):
    keys = ["id", "state", "phase", "provider", "gpu", "gpu_name", "title", "deps", "model", "dataset", "examples",
            "budget_usd", "hard_timeout_s", "est_cost_usd", "est_minutes", "cost_so_far_usd",
            "actual_cost_usd", "gpu_seconds", "progress", "loss_curve", "artifact", "error",
            "data_leaving_device", "requested_by", "outputs"]
    return {k: job[k] for k in keys if job.get(k) is not None}


# ---------- generic workloads: any script, on a shared node or this machine ----------

MAX_UPLOAD = 50 * 1024 * 1024


def run_workload(script, deps=(), args=(), files=(), title=None, needs_gpu=True, min_vram_gb=None,
                 max_minutes=15, target="auto", requested_by="agent"):
    """script: path to a .py file or inline Python. files: local input files copied next to it.
    The script should write results into ./out (or $GPUROUTE_OUT); they come back to this machine."""
    from . import pool
    from .node import hardware

    code = Path(script).read_text() if script.endswith(".py") and Path(script).exists() else script
    if not deps:
        from .deps import infer
        deps = infer(code)
    title = title or (Path(script).name if script.endswith(".py") else "inline script")
    files = [str(Path(f).expanduser().resolve()) for f in files]
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        return {"ok": False, "error": f"input files not found: {missing}"}
    if sum(Path(f).stat().st_size for f in files) > MAX_UPLOAD:
        return {"ok": False, "error": "input files exceed 50 MB; pass URLs the script downloads instead"}

    # Pick where to run: idle shared nodes first (they'd otherwise sit unused, and it keeps
    # this machine free), then this machine.
    tried = []
    candidates = []
    if target in ("auto",) or target not in ("local",):
        names = [target] if target not in ("auto", "local") else list(pool.nodes())
        for info in map(pool.node_info, names):
            gpus = (info.get("hardware") or {}).get("gpus") or []
            vram = max((g.get("vram_gb") or 0 for g in gpus), default=0)
            if not info.get("accepting_jobs"):
                tried.append(f"{info['name']}: {info.get('reason')}")
            elif needs_gpu and not gpus:
                tried.append(f"{info['name']}: no GPU")
            elif min_vram_gb and vram < min_vram_gb:
                tried.append(f"{info['name']}: {vram} GB < {min_vram_gb} GB needed")
            else:
                candidates.append((-vram, info))
    candidates.sort(key=lambda c: c[0])
    if candidates:
        where = ("node", candidates[0][1])
    elif target in ("auto", "local"):
        hw = hardware()
        if needs_gpu and not hw["gpus"]:
            return {"ok": False, "error": "no GPU available locally and no shared node accepting jobs",
                    "tried": tried}
        where = ("local", None)
    else:
        return {"ok": False, "error": f"node {target} can't take this job", "tried": tried}

    job_id = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    d = ledger.JOBS / job_id
    d.mkdir(parents=True, exist_ok=True)
    job = {"id": job_id, "kind": "script", "created": time.time(), "title": title, "model": title,
           "deps": list(deps), "args": list(args), "files": [Path(f).name for f in files],
           "budget_usd": 0.0, "price_per_s": 0.0, "est_cost_usd": 0.0, "requested_by": requested_by,
           "hard_timeout_s": int(max_minutes * 60), "state": "running", "actual_cost_usd": None,
           "started": time.time(), "artifact": str(d / "out")}

    if where[0] == "node":
        info = where[1]
        payload = {"script": code, "deps": list(deps), "args": list(args), "title": title,
                   "max_seconds": job["hard_timeout_s"], "job_id": job_id, "requester": requested_by,
                   "files": {Path(f).name: base64.b64encode(Path(f).read_bytes()).decode() for f in files}}
        r = pool.submit(info["name"], payload)
        if not r.get("ok"):
            return {"ok": False, "error": r.get("error"), "tried": tried}
        gpu = r.get("gpu")
        job.update(provider=f"node:{info['name']}", node=info["name"], gpu=gpu,
                   hard_timeout_s=r["max_seconds"],
                   data_leaving_device=(f"script{' + ' + str(len(files)) + ' input file(s)' if files else ''} "
                                        f"sent to shared node '{info['name']}'"))
    else:
        proc = start_local(d, code, deps, args, files, job["hard_timeout_s"])
        job.update(provider="local", gpu=local_device()["name"] if local_device() else "CPU", pid=proc.pid,
                   data_leaving_device="none: runs on this machine")
    ledger.save_job(job)
    ledger.record("launched", job=job_id, kind="script", provider=job["provider"], title=title,
                  deps=list(deps), cap_s=job["hard_timeout_s"], data_leaving_device=job["data_leaving_device"],
                  requested_by=requested_by)
    return {"ok": True, "job_id": job_id, "provider": job["provider"], "gpu": job["gpu"],
            "skipped": tried, "job": _public(job),
            "message": f"Running '{title}' on {job['provider']} ({job['gpu']}), capped at "
                       f"{job['hard_timeout_s'] // 60} min. Poll job_status('{job_id}'); outputs land in "
                       f"{job['artifact']}."}


def start_local(d, code, deps, args, files, max_seconds):
    from .node import start_script
    for f in files:
        shutil.copy(f, Path(d) / Path(f).name)
    return start_script(d, code, deps, args, max_seconds)


def _script_status(job):
    from . import pool
    d = ledger.JOBS / job["id"]
    if job["provider"].startswith("node:"):
        try:
            r = pool.remote_status(job["node"], job["id"])
        except Exception as e:
            job["phase"] = f"node unreachable ({e.__class__.__name__}); retrying"
            return _public(job)
        text = r.get("log", "")
        (d / "run.log").write_text(text)
        alive = r.get("running", False)
        if not alive:
            (d / "out").mkdir(exist_ok=True)
            for o in r.get("outputs", []):
                if o["bytes"] < MAX_UPLOAD and not (d / "out" / o["name"]).exists():
                    (d / "out" / o["name"]).write_bytes(pool.fetch_output(job["node"], job["id"], o["name"]))
    else:
        text = (d / "run.log").read_text(errors="replace") if (d / "run.log").exists() else ""
        alive = _alive(job.get("pid"))
    events = _events(text)
    last = lambda name: next((e for e in reversed(events) if e["event"] == name), None)  # noqa: E731
    started = last("started")
    end_ev = last("finished") or last("exited") or last("stopped_at_budget")
    progress = [e for e in events if e["event"] == "progress"]
    if started:
        end = end_ev["t"] if end_ev else time.time()
        job["gpu_seconds"] = round(end - started["t"], 1)
        job["gpu_name"] = started.get("gpu")
    if progress:
        job["progress"] = {k: progress[-1].get(k) for k in ("step", "steps", "loss", "tok_per_s")}
    job["outputs"] = [p.name for p in sorted((d / "out").glob("*"))] if (d / "out").exists() else []
    if not alive and job["state"] == "running":
        if last("stopped_at_budget"):
            _settle(job, "stopped_at_budget", job.get("gpu_seconds", 0))
        elif last("finished"):
            _settle(job, "completed", job.get("gpu_seconds", 0))
        else:
            _settle(job, "failed", job.get("gpu_seconds", 0), error=_tail(text))
    ledger.save_job(job)
    out = _public(job)
    out["outputs"] = job["outputs"]
    if job["state"] == "running":
        out["log_tail"] = _tail(text, 6)
    return out
