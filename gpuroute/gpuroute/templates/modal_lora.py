"""Modal app for one gpuroute job.

The GPU type and the hard timeout are baked in at launch time. The timeout is
computed from the job's budget, so Modal itself kills the container before the
budget can be exceeded.
"""
from pathlib import Path

import modal

APP_NAME = "__APP_NAME__"
GPU = "__GPU__"
TIMEOUT_S = __TIMEOUT_S__

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "transformers", "peft", "datasets", "accelerate")
         .add_local_file(Path(__file__).parent / "train_core.py", "/root/train_core.py"))
artifacts = modal.Volume.from_name("gpuroute-artifacts", create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.function(gpu=GPU, timeout=TIMEOUT_S, volumes={"/artifacts": artifacts})
def train(model_id, dataset, rows, seq_len, epochs, lr, lora_r, job_id):
    import train_core

    train_core.run("cuda", model_id, dataset, rows, seq_len, epochs, lr, lora_r,
                   out_dir=f"/artifacts/{job_id}")
    artifacts.commit()


@app.local_entrypoint()
def main(model_id: str, dataset: str, rows: int = 1000, seq_len: int = 512, epochs: int = 1,
         lr: float = 2e-4, lora_r: int = 16, job_id: str = "test"):
    import train_core

    # Local files are read here and sent with the call; Hub dataset ids load remotely.
    data = train_core.load_texts(dataset, rows) if dataset.endswith(".jsonl") else dataset
    train.remote(model_id, data, rows, seq_len, epochs, lr, lora_r, job_id)
