"""Embed a folder of text on a GPU and build a tiny semantic search index.

gpuroute run examples/embed.py --dep sentence-transformers --file notes.txt
Each non-empty line of every input .txt/.md file is one passage.
Writes out/embeddings.npy, out/passages.json and out/demo_search.json.
"""
import glob
import json
import os
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
out_dir = os.environ.get("GPUROUTE_OUT", "out")
os.makedirs(out_dir, exist_ok=True)

passages = []
for path in sorted(glob.glob("*.txt") + glob.glob("*.md")):
    with open(path, errors="replace") as f:
        passages += [line.strip() for line in f if len(line.strip()) > 20]
if not passages:
    from datasets import load_dataset  # fall back to a public dataset
    passages = [r["text"][:500] for r in load_dataset("mlabonne/guanaco-llama2-1k", split="train[:500]")]
print(f"{len(passages)} passages on {device}", flush=True)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
t0 = time.time()
emb = model.encode(passages, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
secs = time.time() - t0
print(f"embedded {len(passages)} passages in {secs:.1f}s ({len(passages) / secs:.0f}/s)", flush=True)

np.save(os.path.join(out_dir, "embeddings.npy"), emb)
with open(os.path.join(out_dir, "passages.json"), "w") as f:
    json.dump(passages, f)

queries = ["how do I train a model", "tips for writing", "explain a scientific concept"]
q = model.encode(queries, normalize_embeddings=True)
demo = {qq: [passages[i][:160] for i in np.argsort(-emb @ qv)[:3]] for qq, qv in zip(queries, q)}
with open(os.path.join(out_dir, "demo_search.json"), "w") as f:
    json.dump({"device": device, "passages": len(passages), "seconds": round(secs, 2), "search": demo}, f, indent=2)
print(json.dumps(demo, indent=2)[:1500])
