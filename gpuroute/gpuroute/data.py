"""Live public data sources, each cached and attributed.

Vast's feed terms allow display and citation with attribution, but not bulk
collection or redistribution. So we keep only the latest snapshot (overwritten
on each refresh, never archived), show it to the user with attribution, and
link back to the source.
"""
import json
import time
from pathlib import Path

import httpx

HOME = Path.home() / ".gpuroute"
CACHE = HOME / "cache"

SOURCES = {
    "vast": {
        "url": "https://storage.googleapis.com/vast-public-gpu-pricing/gpu-pricing-public.json",
        "ttl": 3600,
        "attribution": "GPU marketplace prices: Vast.ai public pricing feed (vast.ai/pricing), "
                       "displayed with attribution per its terms",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/models",
        "ttl": 3600,
        "attribution": "Model catalog: OpenRouter public models API (openrouter.ai)",
    },
}


def fetch(name):
    """Return (data, fetched_at). Serves the cache inside its TTL, or if the network fails."""
    src = SOURCES[name]
    path = CACHE / f"{name}.json"
    if path.exists() and time.time() - path.stat().st_mtime < src["ttl"]:
        return json.loads(path.read_text()), path.stat().st_mtime
    try:
        r = httpx.get(src["url"], timeout=20, headers={"User-Agent": "gpuroute/0.1"})
        r.raise_for_status()
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text)
        return r.json(), time.time()
    except Exception:
        if path.exists():
            return json.loads(path.read_text()), path.stat().st_mtime
        raise


def vast_prices():
    """Current Vast.ai marketplace price per GPU: {our_gpu: {...}}."""
    from .catalog import VAST_KEYS

    data, ts = fetch("vast")
    out = {}
    for key, gpu in VAST_KEYS.items():
        g = data["gpus"].get(key)
        if not g or not g.get("current"):
            continue
        cur = g["current"]
        out[gpu] = {"p10": cur.get("p10"), "median": cur.get("median"), "min": cur.get("min"),
                    "available": cur.get("available", 0)}
    return out, data.get("updated"), ts


def free_models(limit=10):
    """Models OpenRouter currently serves at $0/token."""
    data, ts = fetch("openrouter")
    free = [m for m in data["data"]
            if m["pricing"].get("prompt") == "0" and m["pricing"].get("completion") == "0"]
    free.sort(key=lambda m: -m.get("context_length", 0))
    return [{"id": m["id"], "name": m["name"], "context": m.get("context_length")}
            for m in free[:limit]], len(free), ts
