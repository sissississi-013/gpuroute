"""LoRA fine-tuning loop shared by every provider (Modal GPU, local Mac/CUDA).

Progress is printed as `GPUROUTE {json}` lines so gpuroute can follow a job
from its log alone, without connecting to the machine running it.
"""
import json
import time


def emit(event, **data):
    print("GPUROUTE " + json.dumps({"event": event, "t": time.time(), **data}), flush=True)


def load_texts(dataset, rows):
    if isinstance(dataset, list):
        return dataset[:rows]
    if dataset.endswith(".jsonl"):
        with open(dataset) as f:
            return [json.loads(line).get("text", "") for line in f if line.strip()][:rows]
    from datasets import load_dataset
    ds = load_dataset(dataset, split=f"train[:{rows}]")
    col = "text" if "text" in ds.column_names else ds.column_names[0]
    return [str(x) for x in ds[col]]


def run(device, model_id, dataset, rows=1000, seq_len=512, epochs=1, lr=2e-4, lora_r=16,
        out_dir="adapter", batch=8, deadline=None):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    else:
        name, dtype = ("Apple GPU (MPS)" if device == "mps" else "CPU"), torch.float32
    emit("started", gpu=name)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, task_type="CAUSAL_LM",
                                             target_modules="all-linear"))
    emit("model_loaded", params=sum(p.numel() for p in model.parameters()),
         trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))

    texts = load_texts(dataset, rows)
    emit("data_loaded", rows=len(texts))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    steps = epochs * ((len(texts) + batch - 1) // batch)
    step, tokens, t0 = 0, 0, time.time()
    model.train()
    for _ in range(epochs):
        for i in range(0, len(texts), batch):
            if deadline and time.time() > deadline:
                emit("stopped_at_budget", step=step, steps=steps)
                model.save_pretrained(out_dir)
                return
            enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True,
                      max_length=seq_len).to(device)
            labels = enc.input_ids.masked_fill(enc.attention_mask == 0, -100)
            loss = model(**enc, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            tokens += int(enc.attention_mask.sum())
            if step % 5 == 0 or step == steps or step == 1:
                emit("progress", step=step, steps=steps, loss=round(loss.item(), 4),
                     tok_per_s=round(tokens / (time.time() - t0)))

    model.save_pretrained(out_dir)
    emit("finished", steps=step, tokens=tokens, seconds=round(time.time() - t0, 1), artifact=out_dir)
