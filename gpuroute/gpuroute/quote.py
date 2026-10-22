"""Rank every way to run a job: free tiers, Modal (launchable), Vast marketplace."""
from . import data
from .catalog import FREE_TIERS, MODAL, MODAL_FREE_CREDITS_PER_MONTH
from .estimate import estimate


def quote(model_id, method="lora", examples=1000, seq_len=512, epochs=1, batch=8,
          params_b=None, budget_usd=None):
    est = estimate(model_id, method, examples, seq_len, epochs, batch, params_b)
    options, notes = [], []

    for tier in FREE_TIERS:
        g = est["per_gpu"][tier["gpu"]]
        if g["fits"] and g["est_seconds"] <= tier["session_hours"] * 3600:
            options.append({"provider": tier["provider"], "gpu": f'{tier["gpu"]} x{tier["count"]}',
                            "est_minutes": round(g["est_seconds"] / 60, 1), "cost_usd": 0.0,
                            "out_of_pocket_usd": 0.0, "launchable": False,
                            "how": tier["how"], "needs": tier["needs"], "source": tier["source"]})

    local = est["per_gpu"].get("local")
    if local and local["fits"]:
        from .catalog import local_device
        options.append({"provider": "local", "gpu": local_device()["name"],
                        "est_minutes": round(local["est_seconds"] / 60, 1), "cost_usd": 0.0,
                        "out_of_pocket_usd": 0.0, "private": True,
                        "launchable": method == "lora",
                        "note": "runs on this machine: $0, and the data never leaves the device"})

    from .providers import known
    modal_state = known("modal")
    modal_spent = _modal_spent_this_month()
    for gpu, (_, per_s) in MODAL.items():
        g = est["per_gpu"][gpu]
        if not g["fits"]:
            continue
        cost = g["est_seconds"] * per_s
        credits_left = max(0.0, MODAL_FREE_CREDITS_PER_MONTH - modal_spent)
        options.append({"provider": "modal", "gpu": gpu, "est_minutes": round(g["est_seconds"] / 60, 1),
                        "cost_usd": round(cost, 4), "price_per_hour": round(per_s * 3600, 3),
                        "out_of_pocket_usd": round(max(0.0, cost - credits_left), 4),
                        "launchable": method == "lora" and modal_state.get("ready") is not False,
                        "source": "https://modal.com/pricing",
                        "note": (f"setup needed ({modal_state.get('blocked_on')}): connect_provider('modal')"
                                 if modal_state.get("ready") is False else
                                 f"covered by Modal free credits (${credits_left:.2f} left this month)"
                                 if cost <= credits_left else None)})

    try:
        prices, updated, _ = data.vast_prices()
        for gpu, p in prices.items():
            g = est["per_gpu"].get(gpu)
            if not g or not g["fits"] or not p["p10"] or p["available"] == 0:
                continue
            cost = g["est_seconds"] / 3600 * p["p10"]
            options.append({"provider": "vast", "gpu": gpu, "est_minutes": round(g["est_seconds"] / 60, 1),
                            "cost_usd": round(cost, 4), "out_of_pocket_usd": round(cost, 4),
                            "price_per_hour": p["p10"], "available_now": p["available"],
                            "launchable": False, "source": "https://vast.ai/pricing",
                            "how": "console.vast.ai -> search offers -> rent (launch adapter not built yet)"})
        notes.append(f"Vast.ai prices as of {updated}. {data.SOURCES['vast']['attribution']}.")
    except Exception as e:
        notes.append(f"Vast.ai live prices unavailable ({e.__class__.__name__}); Vast options omitted.")

    if not options:
        notes.append(f"Nothing fits: job needs ~{est['vram_needed_gb']} GB VRAM. "
                     "Try method='qlora', a shorter seq_len, a smaller batch, or a smaller model.")

    options.sort(key=lambda o: (o["out_of_pocket_usd"], not o["launchable"], o["cost_usd"], o["est_minutes"]))
    if budget_usd is not None:
        for o in options:
            o["within_budget"] = o["cost_usd"] <= budget_usd

    best = next((o for o in options if o["launchable"]
                 and (budget_usd is None or o["cost_usd"] <= budget_usd)), None)
    cheapest = options[0] if options else None
    return {"estimate": est, "options": options, "recommended_launch": best,
            "cheapest_overall": cheapest, "notes": notes}


def _modal_spent_this_month():
    from .ledger import all_jobs
    import time
    month = time.strftime("%Y-%m")
    return sum(j.get("actual_cost_usd") or 0 for j in all_jobs()
               if j["provider"] == "modal" and time.strftime("%Y-%m", time.localtime(j["created"])) == month)
