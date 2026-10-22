"""The map, as data an agent can query: where compute physically is, from public datasets.

Layers: ATLAS datacenters (6,131 with coordinates), Epoch AI frontier AI sites (power, owner),
TeleGeography undersea cables, underwater datacenters (public reporting), and orbital compute
(live positions, sunlight and ground-station passes from CelesTrak + SGP4).
"""
import json
import math
from functools import lru_cache
from pathlib import Path

GEO = Path(__file__).parent / "static" / "geo"
PLACES = {  # a few handy anchors so agents can say "near San Francisco"
    "san francisco": (37.77, -122.42), "new york": (40.71, -74.01), "london": (51.51, -0.13),
    "frankfurt": (50.11, 8.68), "singapore": (1.35, 103.82), "tokyo": (35.68, 139.69),
    "virginia": (39.04, -77.49), "dallas": (32.78, -96.80), "amsterdam": (52.37, 4.90),
    "sydney": (-33.87, 151.21), "sao paulo": (-23.55, -46.63), "mumbai": (19.08, 72.88),
    "oregon": (45.60, -121.18), "seattle": (47.61, -122.33), "paris": (48.86, 2.35),
}


@lru_cache
def _load(name):
    return json.loads((GEO / name).read_text())


def _km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def query(layer="all", near=None, lat=None, lon=None, radius_km=500, search=None, owner=None,
          country=None, limit=15):
    """layer: all | datacenters | frontier | undersea | cables | orbit."""
    if near and lat is None:
        key = near.strip().lower()
        if key not in PLACES:
            return {"error": f"unknown place {near!r}; pass lat/lon, or one of: {', '.join(PLACES)}"}
        lat, lon = PLACES[key]
    def keep(name_fields, plat, plon, cntry):
        if search and not any(search.lower() in (f or "").lower() for f in name_fields):
            return False
        if owner and not any(owner.lower() in (f or "").lower() for f in name_fields):
            return False
        if country and country.lower() not in (cntry or "").lower():
            return False
        if lat is not None and _km(lat, lon, plat, plon) > radius_km:
            return False
        return True
    def dist(plat, plon):
        return round(_km(lat, lon, plat, plon)) if lat is not None else None

    out = {"query": {k: v for k, v in dict(layer=layer, near=near, lat=lat, lon=lon, radius_km=radius_km if lat is not None else None,
                                            search=search, owner=owner, country=country).items() if v is not None}}
    if layer in ("all", "frontier"):
        rows = [d for d in _load("epoch.json") if keep([d["name"], d["owner"]], d["lat"], d["lon"], d["country"])]
        rows.sort(key=lambda d: (dist(d["lat"], d["lon"]) or 0) if lat is not None else -d["mw"])
        out["frontier_ai_sites"] = {"total": len(rows), "source": "Epoch AI Frontier Data Centers (CC BY 4.0)",
                                    "access": "private: not rentable by agents",
                                    "items": [{"name": d["name"], "owner": d["owner"], "power_mw": d["mw"],
                                               "h100_equivalents": round(d["h100e"]), "country": d["country"],
                                               "lat": d["lat"], "lon": d["lon"], "km_away": dist(d["lat"], d["lon"])}
                                              for d in rows[:limit]]}
    if layer in ("all", "datacenters"):
        rows = [r for r in _load("atlas.json") if keep([r[2], r[3], r[4]], r[1], r[0], r[5])]
        if lat is not None:
            rows.sort(key=lambda r: _km(lat, lon, r[1], r[0]))
        by_company = {}
        for r in rows:
            by_company[r[3] or "?"] = by_company.get(r[3] or "?", 0) + 1
        out["datacenters"] = {"total": len(rows), "source": "ATLAS Global Data Center Map",
                              "access": "colocation: rentable by contract, not by agent API",
                              "top_operators": sorted(by_company.items(), key=lambda kv: -kv[1])[:8],
                              "items": [{"name": r[2], "operator": r[3], "city": r[4], "country": r[5],
                                         "lat": r[1], "lon": r[0], "km_away": dist(r[1], r[0])} for r in rows[:limit]]}
    if layer in ("all", "undersea"):
        rows = [d for d in _load("undersea.json") if keep([d["name"]], d["lat"], d["lon"], "")]
        out["underwater_datacenters"] = {"total": len(rows), "source": "public reporting; locations approximate",
                                         "access": "not rentable", "items": rows[:limit]}
    if layer in ("all", "cables"):
        rows = [c for c in _load("cables.json") if not search or search.lower() in (c["n"] or "").lower()]
        if lat is not None:
            def near_cable(c):
                lines = c["g"] if c["g"] and not isinstance(c["g"][0][0], (int, float)) else [c["g"]]
                return min(_km(lat, lon, p[1], p[0]) for line in lines for p in line[::3] or line)
            rows = [c for c in rows if near_cable(c) <= radius_km]
        out["undersea_cables"] = {"total": len(rows), "source": "TeleGeography Submarine Cable Map",
                                  "names": [c["n"] for c in rows[:limit * 2]]}
    if layer in ("all", "orbit"):
        from . import orbital
        offers = orbital.offers(6)["offers"]
        sats = []
        for o in json.loads((GEO / "orbit.json").read_text()):
            pos = next(x for x in offers if x["type"] == "position" and x["satellite"] == o["name"])
            comp = [x for x in offers if x["type"] == "compute" and x["satellite"] == o["name"]][:2]
            relay = [x for x in offers if x["type"] == "relay" and x["satellite"] == o["name"]][:2]
            sats.append({"satellite": o["name"], "what": o["what"], "now": pos,
                         "next_compute_windows": [{"offer_id": c["offer_id"], "window": c["window"]} for c in comp],
                         "next_downlinks": [{"offer_id": r["offer_id"], "station": r["operator"], "window": r["window"]}
                                            for r in relay]})
        out["orbital_compute"] = {"source": "CelesTrak elements + SGP4 (live)",
                                  "access": "bookable via book_orbital (mock listings, emulated on ground GPUs)",
                                  "satellites": sats}
    return out
