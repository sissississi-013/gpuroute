"""`gpuroute share`: turn any machine into compute your (or your team's) agents can use.

The owner stays in control. Jobs only run while the machine is idle (if the owner
asks), each job is capped in minutes, one job runs at a time, and every job is
printed in the owner's terminal with who sent it and what it runs. Jobs are
plain Python scripts that declare their own dependencies; `uv` installs them in
a throwaway environment, so the node needs nothing pre-installed.
"""
import base64
import json
import os
import platform
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOME = Path.home() / ".gpuroute"
NODE_CONF = HOME / "node.json"
NODE_JOBS = HOME / "node_jobs"


# ---------- what this machine has, and how busy it is ----------

def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def hardware():
    info = {"hostname": socket.gethostname(), "os": platform.system(), "arch": platform.machine(),
            "cpus": os.cpu_count(), "ram_gb": None, "gpus": []}
    if info["os"] == "Darwin":
        info["ram_gb"] = round(int(_run(["sysctl", "-n", "hw.memsize"]) or 0) / 1024 ** 3)
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if "Apple" in chip:
            cores = re.search(r'"gpu-core-count"\s*=\s*(\d+)', _run(["ioreg", "-rc", "AGXAccelerator", "-d", "1"]))
            info["gpus"].append({"name": f"{chip} GPU" + (f" ({cores.group(1)} cores)" if cores else ""),
                                 "vram_gb": round(info["ram_gb"] * 0.65, 1), "kind": "mps"})
    elif Path("/proc/meminfo").exists():
        kb = int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1))
        info["ram_gb"] = round(kb / 1024 ** 2)
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    for line in filter(None, smi.splitlines()):
        name, mem = [x.strip() for x in line.split(",")]
        info["gpus"].append({"name": name, "vram_gb": round(int(mem) / 1024, 1), "kind": "cuda"})
    return info


def utilization():
    """How busy the machine is right now. GPU % comes from nvidia-smi or, on a Mac,
    the GPU driver's own statistics (no sudo needed)."""
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    cpus = os.cpu_count() or 1
    gpu = None
    smi = _run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
    if smi:
        gpu = max(int(x) for x in smi.split())
    elif platform.system() == "Darwin":
        m = re.search(r'"Device Utilization %"=(\d+)', _run(["ioreg", "-rc", "IOAccelerator", "-d", "1"]))
        gpu = int(m.group(1)) if m else None
    return {"cpu_load_pct": round(100 * load / cpus) if load is not None else None, "gpu_util_pct": gpu}


# ---------- running a job (shared by nodes and local runs) ----------

def start_script(job_dir, script, deps=(), args=(), max_seconds=600):
    """Run a Python script in its own folder with its declared deps, capped in time.
    Outputs written to ./out are collectable. Returns the Popen."""
    job_dir = Path(job_dir)
    (job_dir / "out").mkdir(parents=True, exist_ok=True)
    (job_dir / "job.py").write_text(script)
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if Path(uv).exists():
        cmd = [uv, "run", "--no-project", "--quiet"]
        for d in deps:
            cmd += ["--with", d]
        cmd += ["python", "job.py", *map(str, args)]
    else:  # no uv: use this interpreter and hope the deps are present
        cmd = [sys.executable, "job.py", *map(str, args)]
    log = open(job_dir / "run.log", "w")
    log.write(f"GPUROUTE {json.dumps({'event': 'started', 't': time.time(), 'gpu': _gpu_label()})}\n")
    log.flush()
    proc = subprocess.Popen(cmd, cwd=job_dir, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, "PYTHONUNBUFFERED": "1", "GPUROUTE_OUT": str(job_dir / "out")})

    def cap():
        if proc.poll() is None:
            with open(job_dir / "run.log", "a") as f:
                f.write(f"GPUROUTE {json.dumps({'event': 'stopped_at_budget', 't': time.time()})}\n")
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
    t = threading.Timer(max_seconds, cap)
    t.daemon = True
    t.start()

    def finish():
        code = proc.wait()
        with open(job_dir / "run.log", "a") as f:
            ev = "finished" if code == 0 else "exited"
            f.write(f"GPUROUTE {json.dumps({'event': ev, 't': time.time(), 'code': code})}\n")
    threading.Thread(target=finish, daemon=True).start()
    return proc


def _gpu_label():
    hw = hardware()
    return hw["gpus"][0]["name"] if hw["gpus"] else f"CPU ({hw['cpus']} cores)"


def outputs(job_dir):
    out = Path(job_dir) / "out"
    return [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(out.glob("*"))] if out.exists() else []


# ---------- the node server ----------

class Node:
    def __init__(self, name, max_minutes, only_when_idle, idle_below_pct, price_per_hour):
        HOME.mkdir(parents=True, exist_ok=True)
        conf = json.loads(NODE_CONF.read_text()) if NODE_CONF.exists() else {}
        if not conf.get("token"):
            conf["token"] = secrets.token_urlsafe(12)
            NODE_CONF.write_text(json.dumps(conf))
        self.token = conf["token"]
        self.name = name or socket.gethostname().split(".")[0].lower()
        self.rules = {"max_minutes": max_minutes, "only_when_idle": only_when_idle,
                      "idle_below_pct": idle_below_pct, "max_concurrent": 1,
                      "price_per_hour_usd": price_per_hour}
        self.jobs = {}  # id -> {"proc", "dir", "meta"}
        self.hw = hardware()

    def busy_jobs(self):
        return [j for j in self.jobs.values() if j["proc"].poll() is None]

    def is_idle(self, util):
        vals = [v for v in (util["cpu_load_pct"], util["gpu_util_pct"]) if v is not None]
        return all(v < self.rules["idle_below_pct"] for v in vals)

    def info(self):
        util = utilization()
        running = self.busy_jobs()
        idle = self.is_idle(util) and not running
        accepting = not running and (idle or not self.rules["only_when_idle"])
        return {"name": self.name, "hardware": self.hw, "utilization": util, "idle": idle,
                "accepting_jobs": accepting, "running_jobs": len(running), "rules": self.rules,
                "reason": None if accepting else ("running a job" if running else "owner is using this machine")}

    def submit(self, body):
        info = self.info()
        if not info["accepting_jobs"]:
            return 409, {"ok": False, "error": f"node {self.name} not accepting jobs: {info['reason']}"}
        max_s = min(int(body.get("max_seconds", 600)), self.rules["max_minutes"] * 60)
        job_id = body.get("job_id") or uuid.uuid4().hex[:10]
        d = NODE_JOBS / job_id
        d.mkdir(parents=True, exist_ok=True)
        for fname, b64 in (body.get("files") or {}).items():
            (d / Path(fname).name).write_bytes(base64.b64decode(b64))
        proc = start_script(d, body["script"], body.get("deps", []), body.get("args", []), max_s)
        meta = {"id": job_id, "requester": body.get("requester", "unknown"), "deps": body.get("deps", []),
                "max_seconds": max_s, "started": time.time(), "title": body.get("title", "job"),
                "files": list((body.get("files") or {}).keys())}
        self.jobs[job_id] = {"proc": proc, "dir": d, "meta": meta}
        _say(f"▶ {meta['title']} from {meta['requester']} · deps {', '.join(meta['deps']) or 'none'} · "
             f"{len(meta['files'])} input file(s) · "
             f"capped at {max_s // 60} min · job {job_id}")
        return 200, {"ok": True, "job_id": job_id, "node": self.name, "max_seconds": max_s,
                     "gpu": _gpu_label()}

    def status(self, job_id):
        j = self.jobs.get(job_id)
        if not j:
            return 404, {"error": "no such job"}
        log = (j["dir"] / "run.log").read_text(errors="replace")
        code = j["proc"].poll()
        if code is not None and not j["meta"].get("announced"):
            j["meta"]["announced"] = True
            _say(f"■ job {job_id} {'done' if code == 0 else f'exited ({code})'} after "
                 f"{time.time() - j['meta']['started']:.0f}s")
        return 200, {"job_id": job_id, "running": code is None, "exit_code": code,
                     "log": log[-20000:], "outputs": outputs(j["dir"]), **j["meta"]}

    def cancel(self, job_id):
        j = self.jobs.get(job_id)
        if not j:
            return 404, {"error": "no such job"}
        if j["proc"].poll() is None:
            os.killpg(j["proc"].pid, signal.SIGTERM)
            _say(f"✕ job {job_id} cancelled by requester")
        return 200, {"ok": True}


def _say(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def serve(port=7400, name=None, max_minutes=30, only_when_idle=True, idle_below_pct=40, price_per_hour=0.0):
    node = Node(name, max_minutes, only_when_idle, idle_below_pct, price_per_hour)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, code, body):
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

        def authed(self):
            if self.headers.get("Authorization") == f"Bearer {node.token}":
                return True
            self.reply(401, {"error": "bad or missing token"})
            return False

        def do_GET(self):
            if not self.authed():
                return
            parts = self.path.strip("/").split("/")
            if parts == ["info"]:
                return self.reply(200, node.info())
            if len(parts) == 2 and parts[0] == "jobs":
                return self.reply(*node.status(parts[1]))
            if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "out":
                j = node.jobs.get(parts[1])
                f = j and (j["dir"] / "out" / Path(parts[3]).name)
                if not f or not f.exists():
                    return self.reply(404, {"error": "no such file"})
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                return self.wfile.write(f.read_bytes())
            self.reply(404, {"error": "not found"})

        def do_POST(self):
            if not self.authed():
                return
            parts = self.path.strip("/").split("/")
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if parts == ["jobs"]:
                return self.reply(*node.submit(body))
            if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel":
                return self.reply(*node.cancel(parts[1]))
            self.reply(404, {"error": "not found"})

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    gpu = node.hw["gpus"][0]["name"] if node.hw["gpus"] else "CPU only"
    print(f"\n  gpuroute node '{node.name}' is sharing {gpu}, {node.hw['ram_gb']} GB RAM\n"
          f"  rules: {'only when idle (<' + str(idle_below_pct) + '% busy), ' if only_when_idle else ''}"
          f"max {max_minutes} min per job, one job at a time\n\n"
          f"  Send this line to whoever runs the agent:\n\n"
          f"    gpuroute join {node.name}@{lan_ip()}:{port}#{node.token}\n\n"
          f"  Jobs will be listed here as they arrive. Ctrl-C stops sharing.\n", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        for j in node.busy_jobs():
            os.killpg(j["proc"].pid, signal.SIGTERM)
        print("\nstopped sharing.")
