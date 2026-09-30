#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI"
GGUF_REPO = "https://github.com/leejet/ComfyUI-GGUF.git"

WORKFLOW_URL = "https://comfy.yellorn.com/api/workflows/qwen-image-2-1-uncensored-comfyui.json"
WORKFLOW_NAME = "qwen-image-2-1-uncensored-comfyui.json"

HF_DIFF = "https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/resolve/main"
HF_ENC = "https://huggingface.co/pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-W4A8/resolve/main"

MODELS = [
    ("qwen-image-2.1-UC-Q4_K_M.gguf",
     f"{HF_DIFF}/qwen-image-2.1-UC-Q4_K_M.gguf",
     "diffusion_models"),
    ("qwen3vl_8b_w4a8_heretic.safetensors",
     f"{HF_ENC}/qwen3vl_8b_w4a8_heretic.safetensors",
     "text_encoders"),
    ("qwen_image_2.1_vae_bf16.safetensors",
     f"{HF_DIFF}/vae/qwen_image_2.1_vae_bf16.safetensors",
     "vae"),
]


def detect_root() -> Path:
    try:
        import google.colab  # noqa: F401
        return Path("/content")
    except ImportError:
        pass

    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return Path("/kaggle/working")

    return Path.cwd()


def run(cmd, cwd=None):
    cmd = [str(x) for x in cmd]
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_aria2():
    if shutil.which("aria2c"):
        return

    if Path("/content").exists() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "-qq", "aria2"])
        return

    raise RuntimeError("aria2c is not installed.")


def download_with_aria2(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)

    run([
        "aria2c",
        "--console-log-level=error",
        "-c",
        "-x", "16",
        "-s", "16",
        "-k", "1M",
        url,
        "-d", str(dest.parent),
        "-o", dest.name,
    ])


def download_workflow(comfy_dir: Path):
    save_dir = comfy_dir / "user" / "default" / "workflows"
    save_dir.mkdir(parents=True, exist_ok=True)

    target = save_dir / WORKFLOW_NAME

    with urllib.request.urlopen(WORKFLOW_URL) as r:
        data = json.load(r)

    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("✅ Workflow saved:", target)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fresh Qwen Image 2.1 Uncensored GGUF installer"
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--pin", default="")
    parser.add_argument("--no-workflow", action="store_true")
    args = parser.parse_args(argv)

    root = (args.root or detect_root()).resolve()
    comfy_dir = root / "ComfyUI"
    gguf_dir = comfy_dir / "custom_nodes" / "ComfyUI-GGUF"

    print("=" * 70)
    print("QWEN IMAGE 2.1 UNCENSORED GGUF - CLEAN INSTALL")
    print("=" * 70)
    print("Root:", root)

    # Exact behavior of the working Colab cell:
    # remove ComfyUI and clone everything fresh.
    if comfy_dir.exists():
        print("\nRemoving old ComfyUI:", comfy_dir)
        shutil.rmtree(comfy_dir)

    print("\nCloning ComfyUI...")
    run(["git", "clone", COMFY_REPO, str(comfy_dir)])

    if args.pin:
        run(["git", "fetch", "--all", "-q"], cwd=comfy_dir)
        run(["git", "reset", "--hard", args.pin], cwd=comfy_dir)

    run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        cwd=comfy_dir,
    )

    print("\nCloning leejet/ComfyUI-GGUF...")
    run(["git", "clone", GGUF_REPO, str(gguf_dir)])

    run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        cwd=gguf_dir,
    )

    loader_file = gguf_dir / "loader.py"
    loader_text = loader_file.read_text(encoding="utf-8", errors="ignore")

    if "qwen_image21" not in loader_text:
        raise RuntimeError(
            "This ComfyUI-GGUF checkout does not support qwen_image21."
        )

    print("✅ qwen_image21 support detected")

    if not args.no_workflow:
        download_workflow(comfy_dir)

    ensure_aria2()

    for filename, url, subfolder in MODELS:
        target = comfy_dir / "models" / subfolder / filename
        download_with_aria2(url, target)

    print("\n" + "=" * 70)
    print("✅ INSTALL COMPLETE")
    print("=" * 70)
    print("ComfyUI:", comfy_dir)
    print("GGUF   :", gguf_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
