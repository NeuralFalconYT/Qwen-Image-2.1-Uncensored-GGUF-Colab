#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen-Image-2.1-Uncensored-GGUF"

COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI"
GGUF_REPO = "https://github.com/leejet/ComfyUI-GGUF"  # GGUF loader nodes

HF_DIFF = "https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/resolve/main"
HF_ENC = "https://huggingface.co/pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-W4A8/resolve/main"

# (label, url, subfolder under ComfyUI/models, filename)
MODELS = [
    ("DIFFUSION", f"{HF_DIFF}/qwen-image-2.1-UC-Q4_K_M.gguf",
     "diffusion_models", "qwen-image-2.1-UC-Q4_K_M.gguf"),
    ("ENCODER", f"{HF_ENC}/qwen3vl_8b_w4a8_heretic.safetensors",
     "text_encoders", "qwen3vl_8b_w4a8_heretic.safetensors"),
    ("VAE", f"{HF_DIFF}/vae/qwen_image_2.1_vae_bf16.safetensors",
     "vae", "qwen_image_2.1_vae_bf16.safetensors"),
]

MIN_MODEL_BYTES = 10 * 1024 * 1024  # anything smaller is treated as a failed download
WIDTH = 68
LOG_FILE = Path("install.log")

# ============================================================
# TERMINAL STYLE
# ============================================================


class C:
    G, DG, CY = "\033[92m", "\033[32m", "\033[96m"
    YL, RD, MG = "\033[93m", "\033[91m", "\033[95m"
    DM, B, X = "\033[2m", "\033[1m", "\033[0m"


T0 = time.time()


def t():
    return f"{time.time() - T0:6.1f}s"


def typewriter(text, delay=0.003):
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def section(title):
    print(f"\n{C.CY}┌─[ {C.B}{title}{C.X}{C.CY} ]{'─' * max(2, WIDTH - len(title) - 6)}{C.X}")


def ok(msg):
    print(f"{C.CY}│{C.X} {C.G}[ OK ]{C.X} {msg} {C.DM}t+{t()}{C.X}")


def info(msg):
    print(f"{C.CY}│{C.X} {C.CY}[INFO]{C.X} {msg}")


def warn(msg):
    print(f"{C.CY}│{C.X} {C.YL}[WARN]{C.X} {msg}")


def fail(msg):
    print(f"{C.CY}│{C.X} {C.RD}[FAIL]{C.X} {msg}")


def busy(msg):
    print(f"{C.CY}│{C.X} {C.MG}[ >> ]{C.X} {msg} ...")


def banner():
    art = r"""
  ██████╗ ██╗    ██╗███████╗███╗   ██╗
 ██╔═══██╗██║    ██║██╔════╝████╗  ██║
 ██║   ██║██║ █╗ ██║█████╗  ██╔██╗ ██║
 ██║▄▄ ██║██║███╗██║██╔══╝  ██║╚██╗██║
 ╚██████╔╝╚███╔███╔╝███████╗██║ ╚████║
  ╚══▀▀═╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝
"""
    print(C.G + art + C.X)
    typewriter(f"{C.CY}  >> {MODEL_NAME.upper()}{C.X}")
    typewriter(f"{C.CY}  >> IMAGE 2.1 // UNCENSORED // GGUF Q4_K_M{C.X}")
    typewriter(f"{C.DM}  >> initializing neural pipeline ...{C.X}")
    print(C.DG + "─" * WIDTH + C.X)


# ============================================================
# ENVIRONMENT
# ============================================================


def detect_environment() -> str:
    # Check Kaggle first
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"

    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass

    return "local"


def in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def default_root(env):
    if env == "colab":
        return Path("/content")
    if env == "kaggle":
        return Path("/kaggle/working")
    return Path("./").resolve()


def gpu_info():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return out or None
    except Exception:
        return None


def free_gb(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1024**3


# ============================================================
# COMMAND HELPERS
# ============================================================


def run(cmd, cwd=None, label=None):
    """Run silently, append output to install.log, dump the tail only on failure."""
    cmd = [str(c) for c in cmd]
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                            capture_output=True, text=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n{result.stdout}{result.stderr}")
    if result.returncode != 0:
        fail(label or " ".join(cmd))
        print(C.RD + (result.stdout[-1500:] + result.stderr[-1500:]) + C.X)
        print(f"{C.DM}Full log: {LOG_FILE.resolve()}{C.X}")
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def git_clone(url, dest, label, retries=3):
    dest = Path(dest)
    if dest.exists():
        info(f"Already present: {label}")
        return
    busy(f"Cloning {label}")
    for attempt in range(1, retries + 1):
        try:
            run(["git", "clone", "--depth", "1", "-q", url, dest], label=f"clone {label}")
            ok(f"Cloned {label}")
            return
        except subprocess.CalledProcessError:
            if attempt == retries:
                raise
            warn(f"Retry {attempt}/{retries - 1}")
            shutil.rmtree(dest, ignore_errors=True)
            time.sleep(2)


def pip_requirements(cwd, label):
    req = Path(cwd) / "requirements.txt"
    if not req.exists():
        warn(f"No requirements.txt for {label}")
        return
    busy(f"Installing dependencies: {label}")
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        cwd=cwd, label=f"pip install ({label})")
    ok(f"Dependencies ready: {label}")


def ensure_aria2(env):
    if shutil.which("aria2c"):
        ok("aria2 online")
        return True
    if env in ("colab", "kaggle"):
        busy("Installing aria2")
        run(["apt-get", "update", "-qq"], label="apt update")
        run(["apt-get", "install", "-y", "-qq", "aria2"], label="apt install aria2")
        ok("aria2 installed")
        return True
    warn("aria2c not found locally; falling back to built-in downloader (slower)")
    warn("Install: winget install aria2.aria2  |  brew install aria2  |  apt install aria2")
    return False


# ============================================================
# DOWNLOADS
# ============================================================


def _download_aria2(url, target, token):
    cmd = [
        "aria2c", "--console-log-level=error", "--summary-interval=3",
        "--download-result=hide", "--file-allocation=none",
        "-c", "-x", "16", "-s", "16", "-k", "1M",
        "--max-tries=5", "--retry-wait=3",
        url, "-d", str(target.parent), "-o", target.name,
    ]
    if token:
        cmd[1:1] = ["--header", f"Authorization: Bearer {token}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("[#") or "DL:" in line:
            print(f"{C.CY}│{C.X}   {C.DM}{line}{C.X}")
    proc.wait()
    return proc.returncode == 0


def _download_urllib(url, target, token):
    import urllib.request
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    tmp = target.with_suffix(target.suffix + ".part")
    last = 0.0
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 3 and total:
                print(f"{C.CY}│{C.X}   {C.DM}{done / 1024**3:.2f}/{total / 1024**3:.2f} GB "
                      f"({done * 100 // total}%){C.X}")
                last = time.time()
    tmp.rename(target)
    return True


def download(label, url, dest_dir, filename, use_aria2, token):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename

    if target.exists() and target.stat().st_size >= MIN_MODEL_BYTES:
        ok(f"Cached: {filename}")
        return

    busy(f"Downloading {C.B}{filename}{C.X}")
    try:
        good = (_download_aria2 if use_aria2 else _download_urllib)(url, target, token)
    except Exception as e:  # noqa: BLE001
        fail(f"{filename}: {e}")
        good = False

    if not good or not target.exists() or target.stat().st_size < MIN_MODEL_BYTES:
        fail(f"Download failed: {filename}")
        if not token:
            warn("If this is a 401/403, set HF_TOKEN in your environment and re-run")
        raise RuntimeError(f"download failed: {filename}")

    ok(f"Acquired {filename} ({target.stat().st_size / 1024**3:.2f} GB)")


# ============================================================
# FINAL REPORT
# ============================================================


def final_report(env, comfy_dir, model_paths, elapsed, clear):
    if clear and in_notebook():
        from IPython.display import clear_output
        time.sleep(1.2)
        clear_output(wait=True)

    def row(text, color=C.CY):
        print(C.G + "║" + C.X + color + text.ljust(WIDTH - 2)[:WIDTH - 2] + C.X + C.G + "║" + C.X)

    print(C.G + "╔" + "═" * (WIDTH - 2) + "╗" + C.X)
    print(C.G + "║" + C.B + f"  {MODEL_NAME.upper()}  //  ONLINE".ljust(WIDTH - 2)[:WIDTH - 2]
          + C.X + C.G + "║" + C.X)
    print(C.G + "╠" + "═" * (WIDTH - 2) + "╣" + C.X)
    row(f"  ENVIRONMENT : {env.upper()}")
    row(f"  COMFYUI     : {comfy_dir}")
    row(f"  ELAPSED     : {elapsed:.1f}s")
    print(C.G + "╠" + "═" * (WIDTH - 2) + "╣" + C.X)
    for label, p in model_paths:
        exists = p.exists()
        dot = f"{C.G}●{C.X}" if exists else f"{C.RD}○{C.X}"
        size = p.stat().st_size / 1024**3 if exists else 0
        text = f" {label:<9} : {p.name} ({size:.2f} GB)"
        print(C.G + "║" + C.X + dot + C.CY + text.ljust(WIDTH - 3)[:WIDTH - 3] + C.X + C.G + "║" + C.X)
    print(C.G + "╚" + "═" * (WIDTH - 2) + "╝" + C.X)
    print(f"\n{C.DM}>> {MODEL_NAME} installed.{C.X}")
    print(f"{C.DM}>> launch: cd {comfy_dir} && python main.py --listen 0.0.0.0 --port 8188{C.X}")
    print(f"{C.DM}>> workflow nodes: UnetLoaderGGUF -> models/diffusion_models, "
          f"CLIPLoader -> text_encoders, VAELoader -> vae{C.X}\n")


# ============================================================
# MAIN
# ============================================================


def main(argv=None):
    ap = argparse.ArgumentParser(description=f"Install {MODEL_NAME} for ComfyUI")
    ap.add_argument("--root", type=Path, help="install root (default: auto by environment)")
    ap.add_argument("--fresh", action="store_true", help="delete existing ComfyUI first")
    ap.add_argument("--pin", default="", help="pin ComfyUI to this commit hash")
    ap.add_argument("--launch", action="store_true", help="start ComfyUI after install")
    ap.add_argument("--no-clear", action="store_true", help="keep logs, skip clear_output")
    ap.add_argument("--port", type=int, default=8188)
    args, _ = ap.parse_known_args(argv)  # tolerate notebook-injected args

    env = detect_environment()
    root = (args.root or default_root(env)).resolve()
    comfy_dir = root / "ComfyUI"
    nodes_dir = comfy_dir / "custom_nodes"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    LOG_FILE.write_text("", encoding="utf-8")

    banner()

    # ---------- system scan ----------
    section("SYSTEM SCAN")
    ok(f"Environment : {C.B}{env.upper()}{C.X}")
    ok(f"Root path   : {root}")
    gpu = gpu_info()
    ok(f"GPU         : {gpu}") if gpu else warn("No NVIDIA GPU detected")
    space = free_gb(root)
    (ok if space > 15 else warn)(f"Free disk   : {space:.1f} GB")
    ok("HF token    : set") if token else info("HF token    : not set (public repos only)")

    # ---------- core ----------
    section("CORE INSTALL")
    if args.fresh and comfy_dir.exists():
        busy("Purging old ComfyUI")
        shutil.rmtree(comfy_dir)
        ok("Old ComfyUI purged")

    if args.pin:
        # a pin needs full history, so clone without --depth
        if not comfy_dir.exists():
            busy("Cloning ComfyUI (full history for pin)")
            run(["git", "clone", "-q", COMFY_REPO, comfy_dir], label="clone ComfyUI")
        run(["git", "fetch", "--all", "-q"], cwd=comfy_dir)
        run(["git", "reset", "--hard", args.pin], cwd=comfy_dir)
        ok(f"Pinned to {args.pin[:8]}")
    else:
        git_clone(COMFY_REPO, comfy_dir, "ComfyUI")

    pip_requirements(comfy_dir, "ComfyUI")

    # ---------- gguf ----------
    section("GGUF MODULE")
    gguf_dir = nodes_dir / "ComfyUI-GGUF"
    git_clone(GGUF_REPO, gguf_dir, "ComfyUI-GGUF")
    pip_requirements(gguf_dir, "ComfyUI-GGUF")

    # ---------- downloads ----------
    section("DOWNLOAD ENGINE")
    use_aria2 = ensure_aria2(env)

    section("MODEL ACQUISITION")
    model_paths = []
    for label, url, sub, fname in MODELS:
        dest = comfy_dir / "models" / sub
        download(label, url, dest, fname, use_aria2, token)
        model_paths.append((label, dest / fname))

    # ---------- verify + report ----------
    missing = [p for _, p in model_paths if not p.exists()]
    if missing:
        fail(f"Missing files: {', '.join(p.name for p in missing)}")
        return 1

    final_report(env, comfy_dir, model_paths, time.time() - T0, clear=not args.no_clear)

    if args.launch:
        print(f"{C.CY}>> launching ComfyUI on port {args.port} ...{C.X}\n")
        os.chdir(comfy_dir)
        os.execv(sys.executable,
                 [sys.executable, "main.py", "--listen", "0.0.0.0", "--port", str(args.port)])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{C.YL}[ABORT] interrupted by user{C.X}")
        sys.exit(130)
    except Exception as e:  # noqa: BLE001
        print(f"\n{C.RD}[FATAL] {e}{C.X}")
        print(f"{C.DM}See {LOG_FILE.resolve()} for details{C.X}")
        sys.exit(1)
