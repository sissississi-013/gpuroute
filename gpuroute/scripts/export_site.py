"""Export a public, read-only snapshot of the dashboard for static hosting (Vercel)."""
import json
import shutil
import sys
from pathlib import Path

from gpuroute import pool, ui
from gpuroute.quote import quote

out = Path(sys.argv[1] if len(sys.argv) > 1 else "site")
pkg = Path(ui.__file__).parent
shutil.rmtree(out, ignore_errors=True)
(out / "snapshot").mkdir(parents=True)
shutil.copy(pkg / "templates" / "ui.html", out / "index.html")
shutil.copytree(pkg / "static", out / "static")


def scrub(obj):
    text = json.dumps(obj).replace(str(Path.home()), "~")
    return json.loads(text)


state = ui.state()
state["log"] = [e for e in state["log"] if e["event"] not in ("node_added", "node_removed")]
(out / "snapshot" / "state.json").write_text(json.dumps(scrub(state)))
(out / "snapshot" / "map.json").write_text(json.dumps(scrub(pool.compute_map())))
(out / "snapshot" / "quote.json").write_text(json.dumps(quote("Qwen/Qwen2.5-1.5B-Instruct", "lora", 2000, 512, 1,
                                                              budget_usd=1.0)))
from gpuroute import orbital
(out / "snapshot" / "orbit.json").write_text(json.dumps(orbital.offers(6)))
(out / "vercel.json").write_text(json.dumps({"cleanUrls": True}))
print("exported to", out)
