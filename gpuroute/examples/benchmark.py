"""Measure how much matrix-multiply throughput this machine's GPU actually delivers.

gpuroute run examples/benchmark.py --dep torch
"""
import json
import os
import platform
import socket
import time

import torch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device != "cpu" else torch.float32
name = torch.cuda.get_device_name(0) if device == "cuda" else f"{platform.machine()} {device}"


def sync():
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


results = []
for n in (1024, 2048, 4096):
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(3):
        a @ b
    sync()
    reps = 20
    t0 = time.time()
    for _ in range(reps):
        a @ b
    sync()
    secs = (time.time() - t0) / reps
    tflops = 2 * n ** 3 / secs / 1e12
    results.append({"n": n, "ms": round(secs * 1000, 2), "tflops": round(tflops, 2)})
    print(f"{n}x{n} matmul: {secs * 1000:.1f} ms  ->  {tflops:.2f} TFLOPS", flush=True)

out = {"host": socket.gethostname(), "device": device, "gpu": name, "dtype": str(dtype),
       "best_tflops": max(r["tflops"] for r in results), "runs": results}
os.makedirs(os.environ.get("GPUROUTE_OUT", "out"), exist_ok=True)
with open(os.path.join(os.environ.get("GPUROUTE_OUT", "out"), "benchmark.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out))
