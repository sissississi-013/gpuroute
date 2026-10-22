"""Local dashboard for the human overseeing agent spend: `gpuroute ui`.

Binds to 127.0.0.1 only. Approving is deliberately available here and in the
CLI, never through the MCP server.
"""
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import data, jobs, ledger, providers
from .quote import quote

PAGE = Path(__file__).parent / "templates" / "ui.html"


def state():
    js = []
    for j in ledger.all_jobs()[:20]:
        js.append(jobs.status(j["id"]) if j["state"] in ("running", "launching") else jobs._public(j))
    return {"wallet": ledger.wallet(), "jobs": js, "log": ledger.entries(40)[::-1],
            "providers": providers.status()}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json", code=200):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == "/":
                return self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
            if u.path.startswith("/static/"):
                rel = Path(u.path[len("/static/"):])
                if ".." not in rel.parts:
                    f = Path(__file__).parent / "static" / rel
                    if f.is_file():
                        ctype = "application/json" if f.suffix == ".json" else "image/jpeg"
                        return self.send(f.read_bytes(), ctype)
            if u.path == "/api/state":
                return self.send(state())
            if u.path == "/api/quote":
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                res = quote(q["model"], q.get("method", "lora"), int(q.get("examples", 1000)),
                            int(q.get("seq_len", 512)), int(q.get("epochs", 1)),
                            budget_usd=float(q["budget"]) if q.get("budget") else None,
                            params_b=float(q["params_b"]) if q.get("params_b") else None)
                return self.send(res)
            if u.path == "/api/orbit":
                from . import orbital
                return self.send(orbital.offers(6))
            if u.path == "/api/map":
                from . import pool
                return self.send(pool.compute_map())
            if u.path == "/api/free-models":
                models, total, _ = data.free_models(12)
                return self.send({"models": models, "total": total})
        except Exception as e:
            return self.send({"error": str(e)}, code=400)
        self.send({"error": "not found"}, code=404)

    def do_POST(self):
        # Only accept requests from the dashboard itself, not other sites in the browser.
        origin = self.headers.get("Origin", "")
        if origin and not origin.startswith(("http://127.0.0.1", "http://localhost")):
            return self.send({"error": "forbidden"}, code=403)
        u = urlparse(self.path)
        job_id = parse_qs(u.query).get("job", [""])[0]
        try:
            if u.path == "/api/approve":
                return self.send(jobs.approve(job_id, True))
            if u.path == "/api/deny":
                return self.send(jobs.approve(job_id, False))
            if u.path == "/api/cancel":
                return self.send(jobs.cancel(job_id))
            if u.path == "/api/connect":
                name = parse_qs(u.query).get("provider", ["modal"])[0]
                return self.send(providers.connect(name, open_browser=False))
        except Exception as e:
            return self.send({"error": str(e)}, code=400)
        self.send({"error": "not found"}, code=404)


def serve(port=7311, open_browser=True):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"gpuroute dashboard on {url}  (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
