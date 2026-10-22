"""gpuroute CLI: the human side. Agents use the MCP server; people use this to
see quotes, watch jobs, approve spending and read the audit log."""
import argparse
import json
import sys
import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from . import data, jobs, ledger
from .estimate import estimate
from .quote import quote

con = Console()
STATE_STYLE = {"running": "yellow", "completed": "green", "stopped_at_budget": "magenta",
               "failed": "red", "cancelled": "dim", "denied": "red", "pending_approval": "cyan"}


def money(x):
    return "$0" if not x else (f"${x:.4f}" if x < 0.01 else f"${x:.2f}")


def mins(m):
    return f"{m:.0f} min" if m < 120 else f"{m / 60:.1f} h"


def job_args(p, launch=False):
    p.add_argument("model", help="Hugging Face model id, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    if launch:
        p.add_argument("dataset", help="HF dataset id with a 'text' column, or a local .jsonl path")
        p.add_argument("--budget", type=float, required=True, help="max $ this job may cost")
        p.add_argument("--provider", default="auto", choices=["auto", "local", "modal"])
        p.add_argument("--gpu")
        p.add_argument("--max-minutes", type=float)
    else:
        p.add_argument("--method", default="lora", choices=["lora", "qlora", "full"])
        p.add_argument("--budget", type=float)
        p.add_argument("--params-b", type=float, help="model size in billions, for unknown models")
    p.add_argument("--examples", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--json", action="store_true", help="machine-readable output")


def cmd_estimate(a):
    e = estimate(a.model, a.method, a.examples, a.seq_len, a.epochs, params_b=a.params_b)
    if a.json:
        return print(json.dumps(e, indent=2))
    b = e["vram_breakdown_gb"]
    con.print(Panel(
        f"[bold]{e['model']}[/] · {e['method']} · {e['params_b']}B params · {e['tokens']:,} tokens\n"
        f"VRAM needed: [bold]{e['vram_needed_gb']} GB[/]  (weights+optimizer {b['weights_and_optimizer']} · "
        f"activations {b['activations']} · overhead {b['overhead']})\n[dim]{e['assumptions']}[/]",
        title="estimate"))
    t = Table("GPU", "VRAM", "fits", "est. time")
    for name, g in sorted(e["per_gpu"].items(), key=lambda kv: kv[1]["est_seconds"]):
        t.add_row(name, f"{g['vram_gb']:.0f} GB", "[green]yes[/]" if g["fits"] else "[red]no[/]",
                  mins(g["est_seconds"] / 60))
    con.print(t)


def cmd_quote(a):
    q = quote(a.model, a.method, a.examples, a.seq_len, a.epochs, params_b=a.params_b, budget_usd=a.budget)
    if a.json:
        return print(json.dumps(q, indent=2))
    e = q["estimate"]
    con.print(f"\n[bold]{e['model']}[/] · {e['method']} · {e['tokens']:,} tokens · needs "
              f"[bold]{e['vram_needed_gb']} GB[/] VRAM\n")
    t = Table("#", "where", "GPU", "time", "cost", "you pay", "launch", "notes", show_lines=False)
    for i, o in enumerate(q["options"][:15], 1):
        style = "bold green" if o is q["recommended_launch"] else None
        note = o.get("note") or o.get("needs") or (f"{o['available_now']} available now" if "available_now" in o else "")
        if a.budget is not None and not o.get("within_budget", True):
            note = "[red]over budget[/] " + note
        t.add_row(str(i), o["provider"], o["gpu"], mins(o["est_minutes"]), money(o["cost_usd"]),
                  money(o["out_of_pocket_usd"]), "[green]✓[/]" if o["launchable"] else ("[yellow]setup[/]" if (o.get("note") or "").startswith("setup") else "[dim]manual[/]"),
                  note, style=style)
    con.print(t)
    if q["recommended_launch"]:
        r = q["recommended_launch"]
        con.print(f"[bold green]→ recommended:[/] {r['provider']} {r['gpu']} · {money(r['cost_usd'])} · "
                  f"{mins(r['est_minutes'])}")
    for n in q["notes"]:
        con.print(f"[dim]{n}[/]")


def cmd_launch(a):
    r = jobs.launch(a.model, a.dataset, a.budget, examples=a.examples, seq_len=a.seq_len, epochs=a.epochs,
                    provider=a.provider, gpu=a.gpu, max_minutes=a.max_minutes, requested_by="human:cli")
    if a.json:
        return print(json.dumps(r, indent=2))
    if not r["ok"]:
        con.print(f"[red bold]refused:[/] {r['error']}")
        return 1
    con.print(f"[green]{r['message']}[/]")
    con.print(f"[dim]data leaving device: {r['job']['data_leaving_device']}[/]")
    if r["state"] == "running":
        con.print(f"watch it:  [bold]gpuroute status {r['job_id']} --watch[/]")


def render_job(j):
    style = STATE_STYLE.get(j["state"], "white")
    lines = [f"[{style} bold]{j['state']}[/]" + (f"  [dim]{j['phase']}[/]" if j.get("phase") else ""),
             f"{j['provider']} · {j.get('gpu_name') or j['gpu']} · {j['model']}",
             f"budget {money(j['budget_usd'])} · est {money(j.get('est_cost_usd'))} · "
             f"spent {money(j.get('actual_cost_usd', j.get('cost_so_far_usd')))} · "
             f"GPU time {j.get('gpu_seconds', 0):.0f}s / cap {j.get('hard_timeout_s', 0)}s"]
    if j.get("progress"):
        p = j["progress"]
        filled = int(24 * p["step"] / max(1, p["steps"]))
        lines.append(f"[{'█' * filled}{'░' * (24 - filled)}] step {p['step']}/{p['steps']} · "
                     f"loss {p['loss']} · {p['tok_per_s']} tok/s")
    if j.get("loss_curve") and len(j["loss_curve"]) > 1:
        lines.append("loss " + spark(j["loss_curve"]))
    if j.get("outputs"):
        lines.append(f"[green]outputs:[/] {', '.join(j['outputs'])}  [dim]→ {j['artifact']}[/]")
    elif j.get("artifact") and j["state"] == "completed":
        lines.append(f"[green]adapter:[/] {j['artifact']}")
    if j.get("error"):
        lines.append(f"[red]{j['error']}[/]")
    lines.append(f"[dim]data leaving device: {j.get('data_leaving_device')}[/]")
    return Panel("\n".join(lines), title=f"job {j['id']}", expand=False)


def spark(vals):
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(vals), max(vals)
    return "".join(blocks[int((v - lo) / (hi - lo or 1) * 7)] for v in vals[-48:]) + f"  {vals[0]} → {vals[-1]}"


def cmd_status(a):
    if a.json:
        return print(json.dumps(jobs.status(a.job_id), indent=2))
    if not a.watch:
        return con.print(render_job(jobs.status(a.job_id)))
    with Live(render_job(jobs.status(a.job_id)), console=con, refresh_per_second=2) as live:
        while True:
            j = jobs.status(a.job_id)
            live.update(render_job(j))
            if j["state"] not in ("running", "launching"):
                break
            time.sleep(2)


def cmd_jobs(a):
    t = Table("job", "state", "where", "GPU", "model", "budget", "spent")
    for j in ledger.all_jobs()[:a.limit]:
        j = jobs.status(j["id"]) if j["state"] == "running" else jobs._public(j)
        t.add_row(j["id"], f"[{STATE_STYLE.get(j['state'], 'white')}]{j['state']}[/]", j["provider"],
                  j.get("gpu_name") or j["gpu"], j["model"].split("/")[-1], money(j["budget_usd"]),
                  money(j.get("actual_cost_usd", j.get("cost_so_far_usd"))))
    con.print(t)


def cmd_cancel(a):
    con.print(jobs.cancel(a.job_id))


def cmd_approve(a, ok=True):
    r = jobs.approve(a.job_id, ok)
    if not r["ok"]:
        con.print(f"[red]{r['error']}[/]")
        return 1
    con.print(f"[green]{r.get('message', r['state'])}[/]")


def cmd_wallet(a):
    w = ledger.wallet()
    if a.json:
        return print(json.dumps(w, indent=2))
    used = w["spent_usd"] + w["reserved_usd"]
    filled = int(30 * used / max(w["allowance_usd"], 1e-9))
    con.print(Panel(
        f"[{'█' * min(30, filled)}{'░' * max(0, 30 - filled)}]  {money(used)} of {money(w['allowance_usd'])}\n"
        f"spent {money(w['spent_usd'])} · reserved by running jobs {money(w['reserved_usd'])} · "
        f"[bold]available {money(w['available_usd'])}[/]\n"
        f"approval needed above {money(w['approval_threshold_usd'])} per job · "
        f"per-job cap {money(w['max_job_usd'])}", title=f"agent allowance · {w['month']}", expand=False))


def cmd_policy(a):
    cfg = ledger.set_config(monthly_allowance_usd=a.allowance, approval_threshold_usd=a.approve_above,
                            max_job_usd=a.max_job)
    con.print(cfg)


def cmd_log(a):
    t = Table("time (UTC)", "event", "detail")
    for e in ledger.entries(a.limit):
        detail = {k: v for k, v in e.items() if k not in ("ts", "event")}
        t.add_row(e["ts"][5:19].replace("T", " "), e["event"], json.dumps(detail)[:110])
    con.print(t)


def cmd_free_models(a):
    models, total, _ = data.free_models(a.limit)
    t = Table("model", "context")
    for m in models:
        t.add_row(m["id"], f"{m['context']:,}")
    con.print(t)
    con.print(f"[dim]{total} models at $0/token right now · {data.SOURCES['openrouter']['attribution']}[/]")


def cmd_setup(a):
    from . import providers
    if a.provider:
        with con.status(f"connecting {a.provider}…"):
            r = providers.connect(a.provider, open_browser=not a.no_open)
        for step in r.get("done", []):
            con.print(f"[green]✓[/] {step}")
        if r.get("ready"):
            con.print(f"[bold green]{a.provider} is ready.[/]")
        elif r.get("human_action"):
            h = r["human_action"]
            con.print(f"[yellow bold]one step for you:[/] {h['instruction']}")
            if h.get("url"):
                con.print(f"  {h['url']}" + ("  [dim](opened in your browser)[/]" if h.get("opened_in_browser") else ""))
            con.print(f"[dim]then run: gpuroute setup {a.provider}[/]")
        elif r.get("error"):
            con.print(f"[red]{r['error']}[/]")
        return
    t = Table("provider", "ready", "detail")
    for name, st in providers.status().items():
        ready = {True: "[green]yes[/]", False: "[red]no[/]", None: "[dim]unchecked[/]"}[st.get("ready")]
        detail = st.get("device") or st.get("gpu") or ""
        if st.get("blocked_on"):
            detail = f"blocked on {st['blocked_on']}: gpuroute setup {name}"
        t.add_row(name, ready, detail or st.get("note") or "")
    con.print(t)


def cmd_share(a):
    from .node import serve, serve_pool
    if a.pool:
        return serve_pool(a.pool, a.name, a.max_minutes, not a.always, a.idle_below, a.price)
    serve(a.port, a.name, a.max_minutes, not a.always, a.idle_below, a.price)


def cmd_hub(a):
    """Deploy (or reuse) your internet pool and print the one-line share command for anyone."""
    import secrets as _s
    import subprocess
    from pathlib import Path
    from . import pool
    conf = pool.pool_conf()
    if not conf or a.redeploy:
        app = Path(__file__).parent / "templates" / "hub_app.py"
        with con.status("deploying your pool relay to Modal…"):
            r = subprocess.run([sys.executable, "-m", "modal", "deploy", str(app)], capture_output=True, text=True)
        import re
        m = re.search(r"https://\S+\.modal\.run", r.stdout + r.stderr)
        if not m:
            con.print(f"[red]deploy failed:[/] {(r.stdout + r.stderr)[-400:]}")
            return 1
        import modal
        secret = (conf or {}).get("secret") or _s.token_urlsafe(16)
        modal.Dict.from_name("gpuroute-hub", create_if_missing=True).put("secret", secret)
        conf = {"url": m.group(0), "secret": secret}
        pool.POOL.write_text(json.dumps(conf))
        ledger.record("pool_created", url=conf["url"])
    code = f"{conf['url']}#{conf['secret']}"
    con.print(f"\n[bold]Your compute pool is live:[/] {conf['url']}\n")
    con.print("Anyone can add a spare machine from anywhere (no LAN needed), with:\n")
    con.print(f'  uvx --from "{GIT}" gpuroute share --pool "{code}" --name <their-name>\n')
    online = [n for n in pool.hub_nodes() if n.get("online")]
    con.print(f"[dim]{len(online)} machine(s) in the pool right now: {', '.join(n['name'] for n in online) or 'none yet'}[/]")


def cmd_nodes(a):
    from . import pool
    if a.action == "add":
        if not (a.name and a.url and a.token):
            con.print("usage: gpuroute nodes add <name> <url> <token>")
            return 1
        info = pool.add_node(a.name, a.url, a.token)
        if info["online"]:
            gpu = (info["hardware"].get("gpus") or [{"name": "CPU only"}])[0]["name"]
            con.print(f"[green]✓ added {a.name}[/]: {gpu} · "
                      f"{'idle' if info.get('idle') else 'busy'} · accepting jobs: {info.get('accepting_jobs')}")
        else:
            con.print(f"[yellow]added {a.name}, but it's unreachable right now: {info['reason']}[/]")
        return
    if a.action == "rm":
        pool.remove_node(a.name)
        return con.print(f"removed {a.name}")
    cmd_map(a)


def cmd_join(a):
    """gpuroute join iana@172.20.4.131:7400#token"""
    import re
    m = re.fullmatch(r"([\w.-]+)@([\w.:-]+)#(\S+)", a.code)
    if not m:
        con.print("[red]expected something like[/] iana@172.20.4.131:7400#token  [dim](printed by gpuroute share)[/]")
        return 1
    a.action, a.name, a.url, a.token = "add", m.group(1), f"http://{m.group(2)}", m.group(3)
    return cmd_nodes(a)


GIT = "git+https://github.com/sissississi-013/gpuroute#subdirectory=gpuroute"


def mcp_command():
    """How an agent should launch the MCP server: this install if it's permanent, else via uvx."""
    import shutil
    from pathlib import Path
    exe = Path(sys.prefix) / "bin" / "gpuroute-mcp"
    if exe.exists() and ".cache/uv" not in str(exe) and "archive-v" not in str(exe):
        return [str(exe)]
    return [shutil.which("uvx") or "uvx", "--from", GIT, "gpuroute-mcp"]


def cmd_init(a):
    import shutil
    import subprocess
    from .catalog import local_device
    con.print("\n[bold]gpuroute init[/]\n")
    dev = local_device()
    con.print(f"[green]✓[/] this machine: {dev['name'] if dev else 'no GPU (jobs go to shared nodes or the cloud)'}")
    con.print(f"[green]✓[/] agent allowance: {money(ledger.wallet()['available_usd'])} this month "
              f"(change with gpuroute policy)")
    cmd = mcp_command()
    agents = []
    for name, add in (("claude", ["claude", "mcp", "add", "--scope", "user", "gpuroute", "--"]),
                      ("codex", ["codex", "mcp", "add", "gpuroute", "--"])):
        if not shutil.which(name):
            continue
        already = name == "claude" and "User config" in subprocess.run(
            ["claude", "mcp", "get", "gpuroute"], capture_output=True, text=True).stdout
        if already:
            con.print(f"[green]✓[/] {name}: gpuroute already connected")
            agents.append(name)
            continue
        ok = a.yes or con.input(f"  Connect gpuroute to {name}? [Y/n] ").strip().lower() in ("", "y", "yes")
        if ok:
            r = subprocess.run(add + cmd, capture_output=True, text=True)
            if r.returncode == 0:
                con.print(f"[green]✓[/] {name}: connected. Restart {name} to load the tools")
                agents.append(name)
            else:
                con.print(f"[yellow]![/] {name}: {(r.stderr or r.stdout).strip()[:160]}")
    con.print("\n[dim]Cursor or any other MCP client: add this server[/]")
    con.print(json.dumps({"mcpServers": {"gpuroute": {"command": cmd[0], "args": cmd[1:]}}}))
    con.print("\n[bold]Try asking your agent:[/]")
    con.print("  “What compute can you use right now?”")
    con.print("  “Embed my notes on a GPU, and don’t hog my laptop.”")
    con.print("  “Fine-tune Qwen2.5-0.5B on my data for under $1.”")
    con.print("\n[bold]Three commands:[/]  gpuroute run script.py · gpuroute share · gpuroute ui\n")


def cmd_home(a):
    from . import pool
    entries = pool.compute_map()
    usable = [e for e in entries if e["available"]]
    con.print(f"\n[bold]gpuroute[/]: compute for your agents\n")
    for e in entries:
        mark = "[green]●[/]" if e["available"] else ("[red]●[/]" if e.get("online") is False else "[dim]○[/]")
        con.print(f"  {mark} {e['name']:<10} [dim]{e['kind']:<16}[/] {e.get('gpu') or ''}")
    con.print(f"\n  {len(usable)} usable now.\n")
    con.print("  [bold]gpuroute run[/] script.py      run anything on the best idle GPU (deps auto-detected)")
    con.print("  [bold]gpuroute share[/]              share this machine's idle GPU with your team")
    con.print("  [bold]gpuroute ui[/]                 open the dashboard")
    con.print("  [bold]gpuroute init[/]               connect gpuroute to Claude Code / Codex / Cursor")
    con.print("  [dim]more: gpuroute --help[/]\n")


def cmd_map(a):
    from . import pool
    entries = pool.compute_map()
    if getattr(a, "json", False):
        return print(json.dumps(entries, indent=2))
    t = Table("kind", "name", "compute", "now", "cost", "usable", "notes")
    for e in entries:
        u = e.get("utilization") or {}
        now = ""
        if e.get("online") is False:
            now = "[red]offline[/]"
        elif u:
            now = f"cpu {u.get('cpu_load_pct')}% · gpu {u.get('gpu_util_pct') if u.get('gpu_util_pct') is not None else '?'}%"
        t.add_row(e["kind"], e["name"], str(e.get("gpu") or ""), now, e["cost"],
                  "[green]yes[/]" if e["available"] else "[dim]no[/]", e.get("note") or "")
    con.print(t)


def cmd_run(a):
    r = jobs.run_workload(a.script, a.dep or [], a.args or [], a.file or [], needs_gpu=not a.cpu_ok,
                          min_vram_gb=a.min_vram, max_minutes=a.max_minutes, target=a.on,
                          requested_by="human:cli", budget_usd=a.budget)
    if not r["ok"]:
        con.print(f"[red bold]not started:[/] {r['error']}")
        for t in r.get("tried", []):
            con.print(f"  [dim]skipped {t}[/]")
        return 1
    for t in r.get("skipped", []):
        con.print(f"[dim]skipped {t}[/]")
    con.print(f"[green]→ {r['provider']}[/] · {r['gpu']} · deps: {', '.join(r['job'].get('deps') or []) or 'none'}")
    if sys.stdout.isatty():
        a.job_id, a.watch, a.json = r["job_id"], True, False
        cmd_status(a)
        j = jobs.status(r["job_id"])
        if j.get("outputs"):
            con.print(f"results in [bold]{ledger.JOBS / r['job_id'] / 'out'}[/]")
    else:
        con.print(f"watch it:  gpuroute status {r['job_id']} --watch")


def cmd_orbit(a):
    from . import orbital
    if a.action == "book":
        r = orbital.book(a.offer, a.script, budget_usd=a.budget, max_minutes=a.max_minutes, requested_by="human:cli")
        if not r.get("ok"):
            con.print(f"[red]{r['error']}[/]")
            return 1
        con.print(f"[green]{r['message']}[/]")
        return
    o = orbital.offers(a.hours)
    for sat in dict.fromkeys(x["satellite"] for x in o["offers"]):
        pos = next(x for x in o["offers"] if x["satellite"] == sat and x["type"] == "position")
        con.print(f"\n[bold]{sat}[/]  {pos['alt_km']} km up · {pos['lat']}, {pos['lon']} · "
                  f"{'[yellow]sunlit[/]' if pos['sunlit_now'] else '[blue]in eclipse[/]'}  [dim]{pos['fact']}[/]")
        t = Table("offer", "type", "operator", "window", "", "price / capacity", "downlink")
        for x in o["offers"]:
            if x["satellite"] != sat or x["type"] == "position":
                continue
            if x["type"] == "compute":
                t.add_row(x["offer_id"], "compute", x["operator"], x["window"], x["hardware"],
                          f"${x['usd_per_gpu_min']}/GPU-min", x["next_downlink"])
            else:
                t.add_row(x["offer_id"], "relay", x["operator"], x["window"], f"{x['max_elevation_deg']}° max",
                          f"{x['capacity_gb']} GB @ ${x['usd_per_gb']}/GB", "")
        con.print(t)
    con.print(f"[dim]{o['note']}[/]")


def cmd_mcp_config(a):
    import shutil
    exe = shutil.which("gpuroute-mcp") or f"{sys.prefix}/bin/gpuroute-mcp"
    print(f"claude mcp add gpuroute -- {exe}")


def main():
    ap = argparse.ArgumentParser(prog="gpuroute", description="Size, price and launch GPU training jobs "
                                 "for AI agents, with a hard budget cap.")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("init", help="set up: check this machine and connect gpuroute to your agents").add_argument(
        "--yes", "-y", action="store_true")
    sub.add_parser("join", help="add a shared machine: gpuroute join name@ip:port#token").add_argument("code")
    job_args(sub.add_parser("estimate", help="VRAM + time per GPU"))
    job_args(sub.add_parser("quote", help="rank every way to run a job"))
    job_args(sub.add_parser("launch", help="start a job with a budget"), launch=True)
    p = sub.add_parser("status", help="job progress and cost")
    p.add_argument("job_id")
    p.add_argument("--watch", "-w", action="store_true")
    p.add_argument("--json", action="store_true")
    sub.add_parser("jobs", help="recent jobs").add_argument("--limit", type=int, default=15)
    sub.add_parser("cancel", help="stop a job").add_argument("job_id")
    sub.add_parser("approve", help="approve a job waiting on you").add_argument("job_id")
    sub.add_parser("deny", help="deny a job waiting on you").add_argument("job_id")
    sub.add_parser("wallet", help="agent allowance this month").add_argument("--json", action="store_true")
    p = sub.add_parser("policy", help="set the agent spending policy")
    p.add_argument("--allowance", type=float, help="monthly $ cap for agents")
    p.add_argument("--approve-above", type=float, help="jobs above this $ need your approval")
    p.add_argument("--max-job", type=float, help="no single job above this $")
    sub.add_parser("log", help="audit log of every spending decision").add_argument("--limit", type=int, default=25)
    sub.add_parser("free-models", help="AI models free on OpenRouter right now").add_argument(
        "--limit", type=int, default=15)
    sub.add_parser("mcp-config", help="print the command to add gpuroute to Claude Code")
    p = sub.add_parser("share", help="share this machine's idle compute with your agents/team")
    p.add_argument("--port", type=int, default=7400)
    p.add_argument("--name")
    p.add_argument("--max-minutes", type=int, default=30, help="cap per job")
    p.add_argument("--idle-below", type=int, default=60, help="accept jobs only while CPU/GPU busy %% is below this")
    p.add_argument("--always", action="store_true", help="accept jobs even while the machine is busy")
    p.add_argument("--price", type=float, default=0.0, help="$/hour to charge (0 = free for your team)")
    p.add_argument("--pool", help="join an internet pool: the code printed by `gpuroute hub` (url#secret)")
    sub.add_parser("hub", help="create your internet compute pool and get the share command").add_argument(
        "--redeploy", action="store_true")
    p = sub.add_parser("nodes", help="list shared nodes, or add/rm one")
    p.add_argument("action", nargs="?", default="list", choices=["list", "add", "rm"])
    p.add_argument("name", nargs="?")
    p.add_argument("url", nargs="?")
    p.add_argument("token", nargs="?")
    sub.add_parser("map", help="every compute resource your agents can reach, and how busy it is").add_argument(
        "--json", action="store_true")
    p = sub.add_parser("run", help="run any Python script on the best idle GPU (shared node or this machine)")
    p.add_argument("script", help="path to a .py file; it should write results to ./out")
    p.add_argument("args", nargs="*")
    p.add_argument("--dep", action="append", help="pip dependency (repeatable); default: inferred from imports")
    p.add_argument("--file", action="append", help="input file to send along (repeatable)")
    p.add_argument("--on", default="auto", help="auto, local, a node name, or modal:<GPU> (e.g. modal:H100)")
    p.add_argument("--budget", type=float, default=1.0, help="hard $ cap for cloud GPUs")
    p.add_argument("--min-vram", type=float)
    p.add_argument("--max-minutes", type=float, default=15)
    p.add_argument("--cpu-ok", action="store_true", help="job doesn't need a GPU")
    p = sub.add_parser("orbit", help="satellite compute marketplace: gpuroute orbit [book <offer> script.py]")
    p.add_argument("action", nargs="?", default="list", choices=["list", "book"])
    p.add_argument("offer", nargs="?")
    p.add_argument("script", nargs="?")
    p.add_argument("--hours", type=float, default=12)
    p.add_argument("--budget", type=float, default=1.0)
    p.add_argument("--max-minutes", type=float, default=10)
    p = sub.add_parser("setup", help="provider readiness, or connect one: gpuroute setup modal")
    p.add_argument("provider", nargs="?")
    p.add_argument("--no-open", action="store_true", help="don't open the browser")
    p = sub.add_parser("ui", help="open the local dashboard")
    p.add_argument("--port", type=int, default=7311)
    p.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    if not a.cmd:
        return cmd_home(a)
    fn = {"estimate": cmd_estimate, "quote": cmd_quote, "launch": cmd_launch, "status": cmd_status,
          "jobs": cmd_jobs, "cancel": cmd_cancel, "approve": cmd_approve,
          "deny": lambda a: cmd_approve(a, False), "wallet": cmd_wallet, "policy": cmd_policy,
          "log": cmd_log, "setup": cmd_setup, "share": cmd_share, "join": cmd_join, "init": cmd_init, "hub": cmd_hub, "orbit": cmd_orbit, "nodes": cmd_nodes, "map": cmd_map, "run": cmd_run, "free-models": cmd_free_models, "mcp-config": cmd_mcp_config,
          "ui": lambda a: __import__("gpuroute.ui").ui.serve(a.port, not a.no_open)}[a.cmd]
    sys.exit(fn(a) or 0)


if __name__ == "__main__":
    main()
