"""gpuroute hub: an internet relay so any machine, anywhere, can share spare compute.

Nodes connect outbound over HTTPS (no LAN, no port forwarding). The agent's machine
submits jobs here; nodes pick them up on their next heartbeat and post results back.
State lives in a Modal Dict. Every request needs the pool secret.
"""
import time

import modal

app = modal.App("gpuroute-hub")
store = modal.Dict.from_name("gpuroute-hub", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]")


@app.function(image=image)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, Header, HTTPException, Request

    api = FastAPI()

    def check(auth):
        if not auth or auth != f"Bearer {store.get('secret')}":
            raise HTTPException(401, "bad pool secret")

    @api.post("/node/{name}/beat")
    async def beat(name: str, req: Request, authorization: str = Header(None)):
        check(authorization)
        info = await req.json()
        store[f"node:{name}"] = {**info, "seen": time.time()}
        job = store.pop(f"queue:{name}", None)
        cancels = [k.split(":", 1)[1] for k in store.keys() if k.startswith("cancel:")]
        return {"job": job, "cancel": cancels}

    @api.get("/nodes")
    def nodes(authorization: str = Header(None)):
        check(authorization)
        now = time.time()
        return [{**v, "online": now - v.get("seen", 0) < 25}
                for k, v in ((k, store[k]) for k in store.keys() if k.startswith("node:"))]

    @api.post("/node/{name}/submit")
    async def submit(name: str, req: Request, authorization: str = Header(None)):
        check(authorization)
        job = await req.json()
        if store.get(f"queue:{name}"):
            raise HTTPException(409, "node already has a queued job")
        store[f"queue:{name}"] = job
        store[f"job:{job['job_id']}"] = {"running": True, "queued": True, "log": "", "outputs": {}}
        return {"ok": True}

    @api.post("/job/{job_id}")
    async def update(job_id: str, req: Request, authorization: str = Header(None)):
        check(authorization)
        store[f"job:{job_id}"] = await req.json()
        if not store[f"job:{job_id}"].get("running"):
            store.pop(f"cancel:{job_id}", None)
        return {"ok": True}

    @api.get("/job/{job_id}")
    def status(job_id: str, authorization: str = Header(None)):
        check(authorization)
        return store.get(f"job:{job_id}") or {"error": "no such job"}

    @api.post("/job/{job_id}/cancel")
    def cancel(job_id: str, authorization: str = Header(None)):
        check(authorization)
        store[f"cancel:{job_id}"] = True
        return {"ok": True}

    return api
