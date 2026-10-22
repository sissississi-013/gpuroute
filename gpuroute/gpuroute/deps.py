"""Infer pip packages from a script's imports, so nobody has to list dependencies."""
import ast
import sys

# import name -> pip name, where they differ
PIP_NAMES = {
    "sentence_transformers": "sentence-transformers", "sklearn": "scikit-learn", "cv2": "opencv-python",
    "PIL": "pillow", "yaml": "pyyaml", "bs4": "beautifulsoup4", "faster_whisper": "faster-whisper",
    "mlx_whisper": "mlx-whisper", "mlx_lm": "mlx-lm", "huggingface_hub": "huggingface-hub",
    "llama_cpp": "llama-cpp-python", "dotenv": "python-dotenv", "skimage": "scikit-image",
    "diffusers": "diffusers", "soundfile": "soundfile", "docx": "python-docx", "fitz": "pymupdf",
}


def infer(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    third_party = sorted(m for m in mods if m not in sys.stdlib_module_names and m != "__future__")
    return [PIP_NAMES.get(m, m.replace("_", "-")) for m in third_party]
