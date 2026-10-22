"""The compute pool: every machine your agents may use, plus what it's doing right now."""
import json
from concurrent.futures import ThreadPoolExecutor

import httpx

from . import data, ledger
from .catalog import FREE_TIERS, MODAL, local_device

NODES = ledger.HOME / "nodes.json"


def nodes():
    return json.loads(NODES.read_text()) if NODES.exists() else {}


def add_node(name, url, token):
    n = nodes()
    n[name] = {"url": url.rstrip("/"), "token": token}
    NODES.parent.mkdir(parents=True, exist_ok=True)
    NODES.write_text(json.dumps(n, indent=2))
    ledger.record("node_added", node=name, url=url)
    return node_info(name)


def remove_node(name):
    n = nodes()
    n.pop(name, None)
    NODES.write_text(json.dumps(n, indent=2))
    ledger.record("node_removed", node=name)


def _req(name, method, path, **kw):
    n = nodes()[name]
    r = httpx.request(method, n["url"] + path, headers={"Authorization": f"Bearer {n['token']}"},
                      timeout=kw.pop("timeout", 4), **kw)
    return r


def node_info(name):
    try:
        r = _req(name, "GET", "/info", timeout=3)
        r.raise_for_status()
        return {"name": name, "online": True, **r.json(), "name": name}
    except Exception as e:
        return {"name": name, "online": False, "accepting_jobs": False,
                "reason": f"unreachable ({e.__class__.__name__})"}


def submit(name, payload):
    r = _req(name, "POST", "/jobs", json=payload, timeout=15)
    return r.json()


def remote_status(name, job_id):
    r = _req(name, "GET", f"/jobs/{job_id}", timeout=6)
    return r.json()


def remote_cancel(name, job_id):
    return _req(name, "POST", f"/jobs/{job_id}/cancel", timeout=6).json()


def fetch_output(name, job_id, filename):
    return _req(name, "GET", f"/jobs/{job_id}/out/{filename}", timeout=60).content


def compute_map():
    """Everything an agent could run on, cheapest-to-use first."""
    entries = []
    dev = local_device()
    if dev:
        from .node import utilization
        u = utilization()
        entries.append({"kind": "this machine", "name": "local", "gpu": dev["name"], "vram_gb": dev["vram"],
                        "cost": "$0", "available": True, "utilization": u,
                        "note": "runs where the agent runs; data stays here"})

    names = list(nodes())
    with ThreadPoolExecutor(max_workers=8) as ex:
        infos = list(ex.map(node_info, names))
    for info in infos:
        hw = info.get("hardware", {})
        gpu = (hw.get("gpus") or [{}])[0]
        price = (info.get("rules") or {}).get("price_per_hour_usd") or 0
        entries.append({"kind": "shared node", "name": info["name"], "online": info["online"],
                        "gpu": gpu.get("name") or (f"CPU ({hw.get('cpus')} cores)" if hw else None),
                        "vram_gb": gpu.get("vram_gb"), "ram_gb": hw.get("ram_gb"),
                        "cost": f"${price}/h" if price else "$0 (shared)",
                        "available": info.get("accepting_jobs", False), "idle": info.get("idle"),
                        "utilization": info.get("utilization"), "rules": info.get("rules"),
                        "note": info.get("reason")})

    from .providers import known
    m = known("modal")
    entries.append({"kind": "cloud", "name": "modal", "gpu": ", ".join(list(MODAL)[:4]) + ", …",
                    "cost": f"from ${MODAL['T4'][1] * 3600:.2f}/h, $30/mo free credits",
                    "available": m.get("ready") is True,
                    "note": None if m.get("ready") else f"setup: connect_provider('modal')"
                    + (f" (blocked on {m.get('blocked_on')})" if m.get("blocked_on") else "")})
    try:
        prices, updated, _ = data.vast_prices()
        idle_4090 = prices.get("RTX 4090", {})
        total = sum(p["available"] for p in prices.values())
        entries.append({"kind": "spare-GPU market", "name": "vast.ai",
                        "gpu": f"{total} GPUs listed now (e.g. {idle_4090.get('available', 0)} RTX 4090 "
                               f"from ${idle_4090.get('p10', 0):.2f}/h)",
                        "cost": "live prices", "available": False,
                        "note": f"quote only (no launch adapter yet). {data.SOURCES['vast']['attribution']}, "
                                f"as of {updated}"})
    except Exception:
        pass
    entries += world_compute_summary()
    for t in FREE_TIERS:
        entries.append({"kind": "free tier", "name": t["provider"], "gpu": f"{t['gpu']} x{t['count']}",
                        "cost": "$0", "available": False, "note": f"manual: {t['how']}"})
    return entries


def world_compute_summary():
    """Compute that exists but agents can't rent yet, from public data (shown on the Map tab)."""
    from pathlib import Path
    geo = Path(__file__).parent / "static" / "geo"
    out = []
    try:
        epoch = json.loads((geo / "epoch.json").read_text())
        top = sorted(epoch, key=lambda d: -d["mw"])[:3]
        out.append({"kind": "frontier AI datacenters", "name": "epoch",
                    "gpu": f"{len(epoch)} sites mapped, e.g. " + ", ".join(f"{d['name']} ({d['mw']:.0f} MW)" for d in top),
                    "cost": "private", "available": False,
                    "note": "not rentable by agents. Epoch AI Frontier Data Centers, CC BY 4.0"})
        sea = json.loads((geo / "undersea.json").read_text())
        out.append({"kind": "underwater", "name": "undersea",
                    "gpu": ", ".join(f"{d['name']} ({d['status']})" for d in sea),
                    "cost": "n/a", "available": False, "note": "not rentable; public reporting, locations approximate"})
        orbit = json.loads((geo / "orbit.json").read_text())
        out.append({"kind": "orbital", "name": "orbit",
                    "gpu": "; ".join(f"{o['name']}: {o['what']}" for o in orbit),
                    "cost": "n/a", "available": False, "note": "not rentable yet; live positions from CelesTrak"})
    except Exception:
        pass
    return out
