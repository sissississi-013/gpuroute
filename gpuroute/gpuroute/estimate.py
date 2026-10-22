"""Size a training job before renting anything: VRAM needed and GPU-hours per GPU.

The math is deliberately simple and shown to the caller, so an agent (or a
human) can see why a GPU was ruled in or out.
"""
from .catalog import GPUS, local_device, model_shape

GB = 1024 ** 3
MFU = 0.30            # fraction of peak FLOPS a typical >=1B fine-tune achieves


def mfu(params_b):
    """Small models can't saturate a GPU; calibrated on a 135M-param run on an M2 (~12%)."""
    return min(MFU, 0.10 + 0.2 * params_b)
STARTUP_S = 180       # container start + model download
OVERHEAD_GB = 1.5     # CUDA context, allocator fragmentation
SAFETY = 1.15

# Bytes per parameter held in GPU memory, by method.
#   full:  bf16 weights (2) + bf16 grads (2) + fp32 Adam m,v (8) + fp32 master (4)
#   lora:  frozen bf16 weights (2); adapters are tiny
#   qlora: 4-bit weights (0.5) + quantization constants (~0.06)
WEIGHT_BYTES = {"full": 16.0, "lora": 2.0, "qlora": 0.56}
# Training FLOPs per parameter per token: forward 2 + backward 4 (full) or
# 2 (adapters skip weight grads), + 2 for gradient-checkpoint recompute.
FLOPS_PER_TOKEN = {"full": 8, "lora": 6, "qlora": 6}


def estimate(model_id, method="lora", examples=1000, seq_len=512, epochs=1, batch=8,
             params_b=None):
    if method not in WEIGHT_BYTES:
        raise ValueError("method must be full, lora or qlora")
    p_b, hidden, layers, vocab = model_shape(model_id, params_b)
    n = p_b * 1e9

    weights = n * WEIGHT_BYTES[method]
    # Activations with gradient checkpointing: one bf16 tensor per layer
    # boundary, plus one layer's full working set, plus fp32 logits.
    acts = batch * seq_len * hidden * (2 * layers + 34) + batch * seq_len * vocab * 4
    vram_gb = (weights + acts) / GB * SAFETY + OVERHEAD_GB

    tokens = examples * seq_len * epochs  # upper bound: assumes every row fills seq_len
    flops = FLOPS_PER_TOKEN[method] * n * tokens

    gpus = dict(GPUS)
    local = local_device()
    if local:
        gpus["local"] = local
    per_gpu = {}
    for name, g in gpus.items():
        # MPS trains in fp32, so frozen weights take 4 bytes per param instead of 2.
        need = vram_gb + (n * 2 / GB * SAFETY if name == "local" and method == "lora" else 0)
        fits = g["vram"] >= need
        startup = 60 if name == "local" else STARTUP_S
        secs = flops / (g["tflops"] * 1e12 * mfu(p_b)) + startup
        # fp32 cards without bf16 tensor cores run far below their fp16 peak.
        if not g["bf16"] and name != "local":
            secs = (secs - startup) * 2 + startup
        per_gpu[name] = {"fits": fits, "vram_gb": g["vram"], "est_seconds": round(secs)}

    return {
        "model": model_id, "method": method, "params_b": p_b,
        "examples": examples, "seq_len": seq_len, "epochs": epochs, "batch": batch,
        "tokens": tokens, "train_flops": flops,
        "vram_needed_gb": round(vram_gb, 1),
        "vram_breakdown_gb": {"weights_and_optimizer": round(weights / GB, 2),
                              "activations": round(acts / GB, 2), "overhead": OVERHEAD_GB},
        "per_gpu": per_gpu,
        "assumptions": f"MFU {mfu(p_b):.0%}, gradient checkpointing on, {STARTUP_S}s startup, "
                       f"{int((SAFETY - 1) * 100)}% VRAM safety margin, token count is an upper bound",
    }
