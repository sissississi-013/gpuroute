"""Orbital compute marketplace (first slice).

Satellite operators list two things an agent can buy:
  - compute: GPU time aboard a satellite, available only while it is in sunlight (solar power)
  - relay: downlink capacity during a pass over a ground station, to get results back down

The orbital mechanics are real: positions come from CelesTrak elements propagated with SGP4,
sunlight from a cylindrical Earth-shadow model, passes from elevation above each station.
The operator listings and prices are MOCK. No operator sells this through an API yet, so a
booked job is *emulated*: it runs on ground hardware (Modal GPU if ready, else a shared node
or this machine) at the satellite's real sunlit window, and its downlink is scheduled on the
satellite's real next pass.
"""
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sgp4.api import Satrec, jday

from . import ledger

GEO = Path(__file__).parent / "static" / "geo"
BOOKINGS = ledger.HOME / "orbital_bookings.json"
R_EARTH = 6378.137

# MOCK operator listings over real satellites.
OPERATORS = [
    {"operator": "Starcloud (mock listing)", "sat": "Starcloud-1", "hardware": "NVIDIA H100 80GB",
     "emulate_on": "modal:H100", "usd_per_gpu_min": 0.12, "needs_sun": True,
     "fact": "Starcloud-1 carries an H100 and has trained and run small language models in orbit"},
    {"operator": "HPE Spaceborne (mock listing)", "sat": "ISS", "hardware": "Spaceborne Computer-2 (edge GPU)",
     "emulate_on": "modal:T4", "usd_per_gpu_min": 0.05, "needs_sun": False,
     "fact": "HPE Spaceborne Computer-2 runs edge workloads aboard the ISS"},
]
# Approximate public locations of commercial ground stations.
STATIONS = [
    {"name": "KSAT Svalbard", "lat": 78.23, "lon": 15.39, "mbps": 400, "usd_per_gb": 2.0},
    {"name": "AWS Ground Station, Oregon", "lat": 45.62, "lon": -121.13, "mbps": 300, "usd_per_gb": 3.0},
    {"name": "SSC South Point, Hawaii", "lat": 19.01, "lon": -155.66, "mbps": 150, "usd_per_gb": 2.5},
    {"name": "KSAT Punta Arenas", "lat": -52.94, "lon": -70.85, "mbps": 200, "usd_per_gb": 2.0},
    {"name": "KSAT Singapore", "lat": 1.40, "lon": 103.80, "mbps": 200, "usd_per_gb": 2.5},
]
MIN_ELEV = 10.0
STEP_S = 30


def _sats():
    return {o["name"]: Satrec.twoline2rv(o["tle"][1], o["tle"][2])
            for o in json.loads((GEO / "orbit.json").read_text())}


def _gmst(jd):
    t = (jd - 2451545.0) / 36525.0
    return math.radians((280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * t * t) % 360)


def _sun_dir(jd):
    n = jd - 2451545.0
    L = math.radians((280.460 + 0.9856474 * n) % 360)
    g = math.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)
    eps = math.radians(23.439 - 0.0000004 * n)
    return (math.cos(lam), math.cos(eps) * math.sin(lam), math.sin(eps) * math.sin(lam))


def _state(sat, when):
    jd, fr = jday(when.year, when.month, when.day, when.hour, when.minute, when.second + when.microsecond / 1e6)
    err, r, _ = sat.sgp4(jd, fr)
    if err:
        return None
    s = _sun_dir(jd + fr)
    dot = r[0] * s[0] + r[1] * s[1] + r[2] * s[2]
    perp = math.sqrt(max(0.0, (r[0] ** 2 + r[1] ** 2 + r[2] ** 2) - dot * dot))
    sunlit = dot > 0 or perp > R_EARTH
    th = _gmst(jd + fr)  # TEME -> Earth-fixed (good to a fraction of a degree)
    x = r[0] * math.cos(th) + r[1] * math.sin(th)
    y = -r[0] * math.sin(th) + r[1] * math.cos(th)
    return {"ecef": (x, y, r[2]), "sunlit": sunlit, "alt_km": math.sqrt(sum(c * c for c in r)) - R_EARTH,
            "lat": math.degrees(math.atan2(r[2], math.hypot(x, y))), "lon": math.degrees(math.atan2(y, x))}


def _elev(st, ecef):
    la, lo = math.radians(st["lat"]), math.radians(st["lon"])
    sx, sy, sz = (R_EARTH * math.cos(la) * math.cos(lo), R_EARTH * math.cos(la) * math.sin(lo), R_EARTH * math.sin(la))
    dx, dy, dz = ecef[0] - sx, ecef[1] - sy, ecef[2] - sz
    up = (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))
    rng = math.sqrt(dx * dx + dy * dy + dz * dz)
    return math.degrees(math.asin((dx * up[0] + dy * up[1] + dz * up[2]) / rng))


def timeline(sat_name, hours=12):
    """Sunlit windows and ground-station passes for one satellite over the next `hours`."""
    sat = _sats()[sat_name]
    now = datetime.now(timezone.utc).replace(microsecond=0)
    sun, passes = [], []
    cur_sun = None
    cur_pass = {}
    for i in range(int(hours * 3600 / STEP_S) + 1):
        t = now + timedelta(seconds=i * STEP_S)
        s = _state(sat, t)
        if s is None:
            continue
        if s["sunlit"] and cur_sun is None:
            cur_sun = t
        elif not s["sunlit"] and cur_sun is not None:
            sun.append((cur_sun, t))
            cur_sun = None
        for st in STATIONS:
            e = _elev(st, s["ecef"])
            p = cur_pass.get(st["name"])
            if e >= MIN_ELEV:
                if p is None:
                    cur_pass[st["name"]] = {"start": t, "max_elev": e}
                else:
                    p["max_elev"] = max(p["max_elev"], e)
            elif p is not None:
                passes.append({"station": st["name"], "start": p["start"], "end": t, "max_elev": p["max_elev"]})
                cur_pass.pop(st["name"])
    if cur_sun is not None:
        sun.append((cur_sun, now + timedelta(hours=hours)))
    passes.sort(key=lambda p: p["start"])
    return now, sun, passes, _state(sat, now)


def _iso(t):
    return t.strftime("%H:%M UTC")


def offers(hours=12):
    """Everything bookable: compute windows (sunlit, or always-on) and relay passes, with mock prices."""
    out = []
    for op in OPERATORS:
        now, sun, passes, cur = timeline(op["sat"], hours)
        windows = sun if op["needs_sun"] else [(now, now + timedelta(hours=hours))]
        for i, (a, b) in enumerate(windows[:4]):
            mins = (b - a).total_seconds() / 60
            if mins < 3:
                continue
            nxt = next((p for p in passes if p["start"] >= a), None)
            out.append({
                "offer_id": f"{op['sat'].lower()}-c{i}", "type": "compute", "operator": op["operator"],
                "satellite": op["sat"], "hardware": op["hardware"],
                "window": f"{_iso(a)}–{_iso(b)}" + (" (open now)" if a <= now else ""),
                "starts_in_min": max(0, round((a - now).total_seconds() / 60)), "minutes": round(mins),
                "power": "solar, sunlit" if op["needs_sun"] else "station power",
                "usd_per_gpu_min": op["usd_per_gpu_min"],
                "next_downlink": (f"{nxt['station']} at {_iso(nxt['start'])}" if nxt else "none in window"),
                "mock_listing": True})
        for j, p in enumerate(passes[:6]):
            st = next(s for s in STATIONS if s["name"] == p["station"])
            secs = (p["end"] - p["start"]).total_seconds()
            gb = st["mbps"] * secs / 8 / 1000
            out.append({
                "offer_id": f"{op['sat'].lower()}-r{j}", "type": "relay", "operator": p["station"] + " (mock listing)",
                "satellite": op["sat"], "window": f"{_iso(p['start'])}–{_iso(p['end'])}",
                "starts_in_min": round((p["start"] - now).total_seconds() / 60), "minutes": round(secs / 60, 1),
                "max_elevation_deg": round(p["max_elev"]), "capacity_gb": round(gb, 1),
                "usd_per_gb": st["usd_per_gb"], "mock_listing": True})
        out.append({"offer_id": f"{op['sat'].lower()}-now", "type": "position", "satellite": op["sat"],
                    "lat": round(cur["lat"], 1), "lon": round(cur["lon"], 1), "alt_km": round(cur["alt_km"]),
                    "sunlit_now": cur["sunlit"], "fact": op["fact"]})
    return {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "offers": out,
            "note": "Orbits, sunlight and passes are computed from real CelesTrak elements (SGP4). Operator "
                    "listings and prices are MOCK: no operator sells orbital compute via API yet, so booked "
                    "jobs are emulated on ground GPUs following the satellite's real timeline."}


def book(offer_id, script, deps=(), args=(), files=(), budget_usd=1.0, max_minutes=10, requested_by="agent"):
    from . import jobs
    offs = offers(12)["offers"]
    o = next((x for x in offs if x["offer_id"] == offer_id and x["type"] == "compute"), None)
    if not o:
        return {"ok": False, "error": f"no compute offer {offer_id!r}; call orbital_offers first"}
    op = next(x for x in OPERATORS if x["sat"] == o["satellite"])
    minutes = min(max_minutes, o["minutes"])
    cost = round(minutes * o["usd_per_gpu_min"], 2)
    if cost > budget_usd:
        minutes = int(budget_usd / o["usd_per_gpu_min"])
        cost = round(minutes * o["usd_per_gpu_min"], 2)
    if minutes < 1:
        return {"ok": False, "error": f"budget ${budget_usd} buys less than a minute at ${o['usd_per_gpu_min']}/min"}
    if o["starts_in_min"] > 20:
        return {"ok": False, "error": f"{o['satellite']} isn't sunlit for {o['starts_in_min']} min; pick a window "
                                      "that's open now or opens within 20 min", "window": o["window"]}

    # Emulate: run on ground hardware like the satellite's, inside its real window.
    from .providers import known
    target = op["emulate_on"] if known("modal").get("ready") else "auto"
    r = jobs.run_workload(script, list(deps), list(args), list(files), title=f"{Path(script).name if script.endswith('.py') else 'job'} @ {o['satellite']}",
                          max_minutes=minutes, target=target, requested_by=requested_by, budget_usd=budget_usd)
    if not r.get("ok"):
        return r
    booking = {"booking_id": r["job_id"], "offer": o, "minutes_reserved": minutes, "mock_cost_usd": cost,
               "emulated_on": f"{r.get('provider')} ({r.get('gpu')})",
               "downlink": o["next_downlink"], "booked": time.time()}
    allb = json.loads(BOOKINGS.read_text()) if BOOKINGS.exists() else {}
    allb[r["job_id"]] = booking
    BOOKINGS.write_text(json.dumps(allb, indent=2))
    ledger.record("orbital_booked", job=r["job_id"], satellite=o["satellite"], operator=o["operator"],
                  window=o["window"], minutes=minutes, mock_cost_usd=cost, emulated_on=booking["emulated_on"],
                  downlink=o["next_downlink"], requested_by=requested_by)
    return {"ok": True, "job_id": r["job_id"], **booking,
            "message": f"Booked {minutes} min of {o['hardware']} on {o['satellite']} ({o['window']}), "
                       f"mock price ${cost}. Emulated on {booking['emulated_on']}; results downlink via "
                       f"{o['next_downlink']}. Poll job_status('{r['job_id']}')."}
