"""Provider readiness and self-serve setup.

The agent does every step it can itself (install the CLI, start the login flow,
probe that GPUs actually run). Only steps that are legally or practically the
human's (signing in, adding a payment method) are handed back, as one URL that
is opened for them, and then verified automatically on the next call.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
from pathlib import Path

from . import ledger
from .catalog import local_device

STATE = ledger.HOME / "providers.json"

PROBE = '''
import modal
app = modal.App("gpuroute-probe")

@app.function(gpu="T4", timeout=120)
def probe():
    import subprocess
    return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()

@app.local_entrypoint()
def main():
    print("GPUROUTE_PROBE_OK", probe.remote())
'''


def _load():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def _save(name, **info):
    s = _load()
    s[name] = {**s.get(name, {}), **info, "checked": time.time()}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2))
    return s[name]


def known(name):
    """Last known readiness, without probing: {'ready': bool|None, ...}."""
    return _load().get(name, {"ready": None})


def modal_bin():
    return shutil.which("modal") or (str(Path(sys.prefix) / "bin" / "modal")
                                     if (Path(sys.prefix) / "bin" / "modal").exists() else None)


def _modal_workspace():
    exe = modal_bin()
    if not exe:
        return None
    r = subprocess.run([exe, "profile", "current"], capture_output=True, text=True, timeout=30)
    return r.stdout.strip() or None


def status():
    """Readiness of every provider gpuroute can launch on, from cached checks."""
    dev = local_device()
    out = {"local": {"ready": bool(dev), "device": dev and dev["name"],
                     "note": "free and private" if dev else "no usable GPU on this machine"}}
    m = known("modal")
    out["modal"] = {"ready": m.get("ready"), "gpu": m.get("gpu"), "workspace": m.get("workspace"),
                    "blocked_on": m.get("blocked_on"),
                    "note": "never checked: call connect_provider('modal')" if m.get("ready") is None else None}
    return out


def connect(name, open_browser=True, probe=True):
    if name == "local":
        dev = local_device()
        return {"provider": "local", "ready": bool(dev), "device": dev,
                "done": ["detected this machine's accelerator"] if dev else [],
                "error": None if dev else "no MPS or CUDA device available"}
    if name != "modal":
        return {"provider": name, "ready": False,
                "error": f"no launch adapter for {name!r} yet; quote_job lists manual steps for it"}

    done = []
    # 1. CLI: the agent can install it.
    if not modal_bin():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "modal"], check=False, timeout=300)
        done.append("installed the modal client")
        if not modal_bin():
            return _blocked("modal", done, "install", "pip install modal failed; install it manually", None)

    # 2. Auth: the agent starts the flow; the human clicks "approve" in the browser.
    ws = _modal_workspace()
    if not ws:
        subprocess.Popen([modal_bin(), "token", "new"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        done.append("started Modal sign-in (a browser tab opened)")
        return _blocked("modal", done, "sign_in", "Sign in to Modal in the browser tab that just opened "
                        "(new accounts include $30/month free credits).", "https://modal.com/signup",
                        open_browser=False)
    done.append(f"authenticated to Modal workspace '{ws}'")

    if not probe:
        return {"provider": "modal", **known("modal"), "done": done}

    # 3. GPU access: probe with a 5-second T4 call (~$0.001 of free credits).
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "probe.py"
        path.write_text(PROBE)
        r = subprocess.run([modal_bin(), "run", str(path)], capture_output=True, text=True, timeout=400)
    text = r.stdout + r.stderr
    if "GPUROUTE_PROBE_OK" in text:
        gpu = text.split("GPUROUTE_PROBE_OK", 1)[1].strip().splitlines()[0]
        done.append(f"ran a test function on a cloud GPU ({gpu})")
        _save("modal", ready=True, gpu=gpu, workspace=ws, blocked_on=None)
        ledger.record("provider_ready", provider="modal", workspace=ws, gpu=gpu)
        return {"provider": "modal", "ready": True, "workspace": ws, "gpu": gpu, "done": done}
    if "payment method" in text.lower():
        return _blocked("modal", done, "payment_method",
                        "Add a payment method in Modal (Settings → Usage and Billing). GPU use is still "
                        "drawn from the $30/month free credits first, and gpuroute's allowance caps what "
                        "agents can spend.", f"https://modal.com/settings/{ws}/usage",
                        open_browser=open_browser, workspace=ws)
    return _blocked("modal", done, "unknown", "GPU probe failed: " + text.strip()[-400:], None, workspace=ws)


def _blocked(name, done, step, instruction, url, open_browser=True, **extra):
    _save(name, ready=False, blocked_on=step, **extra)
    opened = False
    if url and open_browser:
        opened = webbrowser.open(url)
    ledger.record("human_action_requested", provider=name, step=step, url=url)
    return {"provider": name, "ready": False, "done": done,
            "human_action": {"step": step, "instruction": instruction, "url": url, "opened_in_browser": opened},
            "next": f"After the user confirms, call connect_provider('{name}') again to verify."}
