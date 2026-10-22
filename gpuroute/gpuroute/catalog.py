"""Static reference data: GPU specs, provider list prices, free tiers, model shapes.

Everything here is cited. Live prices come from data.py; this file is only for
facts that don't change hourly.
"""

# Dense fp16/bf16 tensor TFLOPS (no sparsity), from NVIDIA datasheets.
# bf16=False means the card lacks bf16 and trains in fp32/fp16.
GPUS = {
    "T4":        {"vram": 16, "tflops": 65,   "bf16": False},
    "L4":        {"vram": 24, "tflops": 121,  "bf16": True},
    "A10":       {"vram": 24, "tflops": 125,  "bf16": True},
    "L40S":      {"vram": 48, "tflops": 362,  "bf16": True},
    "A100-40GB": {"vram": 40, "tflops": 312,  "bf16": True},
    "A100-80GB": {"vram": 80, "tflops": 312,  "bf16": True},
    "H100":      {"vram": 80, "tflops": 989,  "bf16": True},
    "H200":      {"vram": 141, "tflops": 989, "bf16": True},
    "B200":      {"vram": 180, "tflops": 2250, "bf16": True},
    "RTX 3090":  {"vram": 24, "tflops": 71,   "bf16": True},
    "RTX 4090":  {"vram": 24, "tflops": 165,  "bf16": True},
    "RTX 5090":  {"vram": 32, "tflops": 209,  "bf16": True},
    "RTX A6000": {"vram": 48, "tflops": 155,  "bf16": True},
    "A40":       {"vram": 48, "tflops": 150,  "bf16": True},
    "V100":      {"vram": 32, "tflops": 125,  "bf16": False},
}

# Modal list prices, $/sec, from https://modal.com/pricing (fetched 2026-10-22).
# Key = our GPU name, value = (modal gpu string, $/sec).
MODAL = {
    "T4":        ("T4", 0.000164),
    "L4":        ("L4", 0.000222),
    "A10":       ("A10", 0.000306),
    "L40S":      ("L40S", 0.000542),
    "A100-40GB": ("A100-40GB", 0.000583),
    "A100-80GB": ("A100-80GB", 0.000694),
    "H100":      ("H100", 0.001097),
    "H200":      ("H200", 0.001261),
    "B200":      ("B200", 0.001736),
}
MODAL_FREE_CREDITS_PER_MONTH = 30.0  # Starter plan

# Vast.ai public pricing feed key -> our GPU name.
VAST_KEYS = {
    "tesla t4": "T4", "l4": "L4", "a10": "A10", "l40s": "L40S",
    "a100 pcie": "A100-40GB", "a100 sxm4": "A100-80GB", "h100 sxm": "H100",
    "h200": "H200", "b200": "B200", "rtx 3090": "RTX 3090", "rtx 4090": "RTX 4090",
    "rtx 5090": "RTX 5090", "rtx a6000": "RTX A6000", "a40": "A40", "tesla v100": "V100",
}

# Free tiers. No launch adapter yet for these: gpuroute hands back steps instead.
FREE_TIERS = [
    {"provider": "kaggle", "gpu": "T4", "count": 2, "quota_hours_week": 30, "session_hours": 12,
     "needs": "Kaggle account + phone verification",
     "how": "kaggle.com/code -> New Notebook -> Settings -> Accelerator: GPU T4 x2",
     "source": "https://www.kaggle.com/docs/notebooks"},
    {"provider": "colab", "gpu": "T4", "count": 1, "quota_hours_week": None, "session_hours": 4,
     "needs": "Google account; availability not guaranteed",
     "how": "colab.research.google.com -> Runtime -> Change runtime type -> T4 GPU",
     "source": "https://research.google.com/colaboratory/faq.html"},
]

# Shapes for common models: (params in billions, hidden size, layers, vocab).
MODELS = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": (0.135, 576, 30, 49152),
    "HuggingFaceTB/SmolLM2-360M-Instruct": (0.36, 960, 32, 49152),
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": (1.7, 2048, 24, 49152),
    "Qwen/Qwen2.5-0.5B-Instruct": (0.49, 896, 24, 151936),
    "Qwen/Qwen2.5-1.5B-Instruct": (1.54, 1536, 28, 151936),
    "Qwen/Qwen2.5-3B-Instruct": (3.09, 2048, 36, 151936),
    "Qwen/Qwen2.5-7B-Instruct": (7.6, 3584, 28, 152064),
    "Qwen/Qwen3-8B": (8.2, 4096, 36, 151936),
    "meta-llama/Llama-3.2-1B-Instruct": (1.24, 2048, 16, 128256),
    "meta-llama/Llama-3.2-3B-Instruct": (3.21, 3072, 28, 128256),
    "meta-llama/Llama-3.1-8B-Instruct": (8.0, 4096, 32, 128256),
    "Qwen/Qwen2.5-14B-Instruct": (14.7, 5120, 48, 152064),
    "Qwen/Qwen2.5-32B-Instruct": (32.8, 5120, 64, 152064),
    "meta-llama/Llama-3.3-70B-Instruct": (70.6, 8192, 80, 128256),
}


# Apple Silicon GPU fp32 TFLOPS (training on MPS runs in fp32).
APPLE_TFLOPS = {"M1": 2.6, "M2": 3.6, "M3": 4.1, "M4": 4.3}


def local_device():
    """This machine's accelerator, if it can train: {'name', 'device', 'vram', 'tflops', 'bf16'}.

    Cached on disk because importing torch takes a couple of seconds.
    """
    import json
    from pathlib import Path

    cache = Path.home() / ".gpuroute" / "cache" / "local_device.json"
    if cache.exists():
        return json.loads(cache.read_text())
    dev = _detect_local()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(dev))
    return dev


def _detect_local():
    import platform
    import subprocess

    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {"name": props.name, "device": "cuda", "vram": props.total_memory / 1024 ** 3,
                "tflops": 50, "bf16": torch.cuda.is_bf16_supported()}
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
        ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True).stdout) / 1024 ** 3
        base = next((k for k in APPLE_TFLOPS if k in chip), "M1")
        # Unified memory is shared with the OS; MPS gets roughly two thirds.
        return {"name": f"{chip} ({ram:.0f} GB unified)", "device": "mps", "vram": round(ram * 0.65, 1),
                "tflops": APPLE_TFLOPS[base], "bf16": False}
    return None


def model_shape(model_id, params_b=None):
    if model_id in MODELS:
        return MODELS[model_id]
    if params_b is None:
        raise ValueError(f"unknown model {model_id!r}; pass params_b (size in billions)")
    # Rough shape from size alone, following typical dense decoder proportions.
    hidden = int((params_b * 1e9 / 12 / 32) ** 0.5 // 128 * 128) or 512
    layers = max(12, int(params_b * 1e9 / (12 * hidden * hidden)))
    return (params_b, hidden, layers, 128000)
