"""Modal app for one gpuroute workload: any script, on the GPU the agent picked.

GPU, deps and the hard timeout (= budget / price) are baked in at launch time.
"""
import base64
import json
import os
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "__APP_NAME__"
GPU = "__GPU__"
TIMEOUT_S = __TIMEOUT_S__
DEPS = __DEPS__

image = modal.Image.debian_slim(python_version="3.12")
if DEPS:
    image = image.pip_install(*DEPS)
app = modal.App(APP_NAME, image=image)


@app.function(gpu=GPU, timeout=TIMEOUT_S)
def run(script: str, files: dict, args: list):
    work = Path("/tmp/job")
    (work / "out").mkdir(parents=True, exist_ok=True)
    (work / "job.py").write_text(script)
    for name, b64 in files.items():
        (work / name).write_bytes(base64.b64decode(b64))
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print("GPUROUTE " + json.dumps({"event": "started", "t": time.time(), "gpu": gpu}), flush=True)
    code = subprocess.run(["python", "job.py", *args], cwd=work,
                          env={**os.environ, "GPUROUTE_OUT": str(work / "out")}).returncode
    outs = {p.name: base64.b64encode(p.read_bytes()).decode()
            for p in (work / "out").glob("*") if p.stat().st_size < 50_000_000}
    print("GPUROUTE " + json.dumps({"event": "finished" if code == 0 else "exited", "t": time.time(),
                                    "code": code}), flush=True)
    return outs


@app.local_entrypoint()
def main(spec: str):
    s = json.loads(Path(spec).read_text())
    outs = run.remote(s["script"], s["files"], s["args"])
    out_dir = Path(s["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, b64 in outs.items():
        (out_dir / name).write_bytes(base64.b64decode(b64))
