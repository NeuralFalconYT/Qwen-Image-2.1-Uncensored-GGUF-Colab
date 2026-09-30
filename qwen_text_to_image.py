from __future__ import annotations

import gc
import os
import random
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import numpy as np
import torch
from PIL import Image


# ============================================================
# DEFAULT CONFIG
# ============================================================

UNET_NAME = "qwen-image-2.1-UC-Q4_K_M.gguf"
CLIP_NAME = "qwen3vl_8b_w4a8_heretic.safetensors"
VAE_NAME = "qwen_image_2.1_vae_bf16.safetensors"

DEFAULT_OUTPUT_DIR = "saved_images"

# Runtime globals
COMFY_PATH: Path | None = None

UNET = None
CLIP = None
VAE = None

text_encoder = None
sampler = None
vae_decoder = None

NODE_CLASS_MAPPINGS = None
model_management = None

DEVICE_MAP: dict[str, str] = {}

_LOADED = False


# ============================================================
# ENVIRONMENT
# ============================================================

def detect_environment() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"

    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass

    return "local"


def get_root_path() -> Path:
    env = detect_environment()

    if env == "colab":
        return Path("/content")

    if env == "kaggle":
        return Path("/kaggle/working")

    return Path(".").resolve()


def get_comfy_path(comfy_path: str | os.PathLike | None = None) -> Path:
    if comfy_path is not None:
        return Path(comfy_path).expanduser().resolve()

    env_path = os.environ.get("COMFYUI_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    return get_root_path() / "ComfyUI"


# ============================================================
# HELPERS
# ============================================================

def get_output(obj: Union[Sequence, Mapping, Any], index: int = 0) -> Any:
    """Return a ComfyUI node output item by index."""
    if hasattr(obj, "result"):
        return obj.result[index]

    if isinstance(obj, Mapping) and "result" in obj:
        return obj["result"][index]

    return obj[index]


def _align_dimension(value: int, multiple: int = 16) -> int:
    value = int(value)

    if value <= 0:
        raise ValueError("width and height must be positive integers")

    return max(multiple, int(round(value / multiple)) * multiple)


def _make_filename(prompt: str, seed: int) -> str:
    """first_20_prompt_words_<short-uuid>_<seed>.png"""
    words = re.findall(r"[A-Za-z0-9]+", prompt)[:20]

    if not words:
        words = ["qwen_image"]

    slug = "_".join(words).lower()
    slug = slug[:150].strip("_")

    short_uuid = uuid.uuid4().hex[:10]

    return f"{slug}_{short_uuid}_{seed}.png"


# ============================================================
# MEMORY
# ============================================================

def clean_memory() -> None:
    """Clean temporary Python/CUDA memory without unloading the main models."""
    gc.collect()

    if torch.cuda.is_available():
        for gpu_id in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(gpu_id):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass

    if model_management is not None:
        try:
            model_management.soft_empty_cache()
        except Exception:
            pass


def get_gpu_info() -> list[dict[str, Any]]:
    info = []

    for gpu_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(gpu_id)

        info.append(
            {
                "id": gpu_id,
                "name": torch.cuda.get_device_name(gpu_id),
                "total_gb": props.total_memory / 1024**3,
                "allocated_gb": torch.cuda.memory_allocated(gpu_id) / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved(gpu_id) / 1024**3,
            }
        )

    return info


def print_vram(prefix: str = "VRAM") -> None:
    print(f"\n{prefix}")

    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return

    for item in get_gpu_info():
        print(
            f"GPU {item['id']} | {item['name']} | "
            f"allocated={item['allocated_gb']:.2f} GB | "
            f"reserved={item['reserved_gb']:.2f} GB | "
            f"total={item['total_gb']:.2f} GB"
        )


# ============================================================
# DEVICE PLACEMENT
# ============================================================

def _normalize_device(device: str | int) -> str:
    """Accept 0, 1, "0", "cuda:1" ... and return "cuda:N"."""
    if isinstance(device, int):
        return f"cuda:{device}"

    device = str(device).strip().lower()

    if device.isdigit():
        return f"cuda:{device}"

    if device == "cuda":
        return "cuda:0"

    if not device.startswith("cuda:"):
        raise ValueError(
            f"Invalid device '{device}'. Use 'cuda:0' or 'cuda:1' "
            "(no 'default' / 'cpu')."
        )

    return device


def resolve_devices(
    unet_device: str | int | None = None,
    clip_device: str | int | None = None,
    vae_device: str | int | None = None,
) -> dict[str, str]:
    """
    Decide where every model goes.

      1 GPU  -> everything cuda:0
      2+ GPU -> UNET cuda:0, CLIP cuda:1, VAE cuda:1
    """
    gpu_count = torch.cuda.device_count()

    if gpu_count >= 2:
        defaults = {"unet": "cuda:0", "clip": "cuda:1", "vae": "cuda:1"}
    else:
        defaults = {"unet": "cuda:0", "clip": "cuda:0", "vae": "cuda:0"}

    chosen = {
        "unet": _normalize_device(unet_device) if unet_device is not None else defaults["unet"],
        "clip": _normalize_device(clip_device) if clip_device is not None else defaults["clip"],
        "vae": _normalize_device(vae_device) if vae_device is not None else defaults["vae"],
    }

    for role, dev in chosen.items():
        idx = int(dev.split(":")[1])
        if idx >= gpu_count:
            raise RuntimeError(
                f"{role} device is {dev} but only {gpu_count} GPU(s) detected."
            )

    return chosen


def _load_on_device(
    candidates: Sequence[str],
    device: str,
    kwargs: dict[str, Any],
    label: str,
    allow_plain_fallback: str | None = None,
    plain_kwargs: dict[str, Any] | None = None,
) -> Any:
    """
    Load a model with an explicit device using a MultiGPU loader node.

    candidates            : MultiGPU node names to try, in order.
    allow_plain_fallback  : core node name usable ONLY when the device is
                            cuda:0 on a single-GPU machine (core ComfyUI
                            already lands on cuda:0 there).
    """
    for node_name in candidates:
        if node_name not in NODE_CLASS_MAPPINGS:
            continue

        loader = NODE_CLASS_MAPPINGS[node_name]()
        info = loader.INPUT_TYPES()
        inputs = {**info.get("required", {}), **info.get("optional", {})}

        device_key = None
        for key in ("device", "compute_device"):
            if key in inputs:
                device_key = key
                break

        if device_key is None:
            raise RuntimeError(
                f"{node_name} has no device input; cannot place {label} on {device}."
            )

        options = inputs[device_key][0]
        if isinstance(options, (list, tuple)) and device not in options:
            raise RuntimeError(
                f"{node_name} does not offer '{device}'. "
                f"Available: {list(options)}"
            )

        call_kwargs = dict(kwargs)
        call_kwargs[device_key] = device

        fn = getattr(loader, loader.FUNCTION)
        return get_output(fn(**call_kwargs)), node_name

    # No MultiGPU node found.
    if (
        allow_plain_fallback
        and torch.cuda.device_count() == 1
        and device == "cuda:0"
        and allow_plain_fallback in NODE_CLASS_MAPPINGS
    ):
        loader = NODE_CLASS_MAPPINGS[allow_plain_fallback]()
        fn = getattr(loader, loader.FUNCTION)
        return get_output(fn(**(plain_kwargs or kwargs))), allow_plain_fallback

    raise RuntimeError(
        f"Cannot place {label} on {device}: none of {list(candidates)} found.\n"
        "Install ComfyUI-MultiGPU into ComfyUI/custom_nodes and restart:\n"
        "  git clone https://github.com/pollockjj/ComfyUI-MultiGPU "
        "ComfyUI/custom_nodes/ComfyUI-MultiGPU"
    )


def _describe_device(model_obj: Any) -> str:
    """Best-effort readout of where a ComfyUI model object will run."""
    for attr_path in (
        ("load_device",),
        ("patcher", "load_device"),
        ("first_stage_model", "device"),
        ("device",),
    ):
        try:
            obj = model_obj
            for attr in attr_path:
                obj = getattr(obj, attr)
            return str(obj)
        except Exception:
            continue
    return "unknown"


# ============================================================
# MODEL LOAD
# ============================================================

async def load_model(
    comfy_path: str | os.PathLike | None = None,
    unet_name: str = UNET_NAME,
    clip_name: str = CLIP_NAME,
    vae_name: str = VAE_NAME,
    unet_device: str | int | None = None,
    clip_device: str | int | None = None,
    vae_device: str | int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Load Qwen Image 2.1 once, with explicit GPU placement.

    Defaults:
        1 GPU  : UNET/CLIP/VAE all on cuda:0
        2 GPUs : UNET cuda:0 | CLIP cuda:1 | VAE cuda:1

    Manual override example:
        await qwen.load_model(unet_device="cuda:0",
                              clip_device="cuda:1",
                              vae_device="cuda:0")
    """
    global COMFY_PATH, DEVICE_MAP
    global UNET, CLIP, VAE
    global text_encoder, sampler, vae_decoder
    global NODE_CLASS_MAPPINGS, model_management
    global _LOADED

    if _LOADED:
        if verbose:
            print("✅ Qwen Image 2.1 is already loaded.")
            print("Devices:", DEVICE_MAP)
            print_vram("Current VRAM")
        return {
            "loaded": True,
            "already_loaded": True,
            "devices": dict(DEVICE_MAP),
            "gpus": get_gpu_info(),
        }

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected.")

    COMFY_PATH = get_comfy_path(comfy_path)

    if not COMFY_PATH.is_dir():
        raise FileNotFoundError(f"ComfyUI directory not found: {COMFY_PATH}")

    os.chdir(COMFY_PATH)

    comfy_str = str(COMFY_PATH)
    if comfy_str not in sys.path:
        sys.path.insert(0, comfy_str)

    # Make cuda:0 the current device so nothing lands elsewhere by accident.
    torch.cuda.set_device(0)

    devices = resolve_devices(unet_device, clip_device, vae_device)

    if verbose:
        print("=" * 60)
        print("GPU DETECTION")
        print("=" * 60)
        print("GPU count:", torch.cuda.device_count())

        for item in get_gpu_info():
            print(f"GPU {item['id']}: {item['name']} | {item['total_gb']:.2f} GB")

        print("\nDevice plan:")
        print(f"  UNET : {devices['unet']}")
        print(f"  CLIP : {devices['clip']}")
        print(f"  VAE  : {devices['vae']}")

    import nodes

    await nodes.init_extra_nodes(
        init_custom_nodes=True,
        init_api_nodes=False,
    )

    from nodes import (
        KSampler,
        NODE_CLASS_MAPPINGS as NODE_MAP,
        VAEDecode,
    )

    NODE_CLASS_MAPPINGS = NODE_MAP

    try:
        import comfy.model_management as _model_management
        model_management = _model_management
    except Exception:
        model_management = None

    for name in ("UnetLoaderGGUF", "TextEncodeQwenImage21"):
        if name not in NODE_CLASS_MAPPINGS:
            raise RuntimeError(f"Missing required ComfyUI node: {name}")

    text_encoder = NODE_CLASS_MAPPINGS["TextEncodeQwenImage21"]()
    sampler = KSampler()
    vae_decoder = VAEDecode()

    # --------------------------------------------------------
    # UNET
    # --------------------------------------------------------

    if verbose:
        print(f"\nLoading GGUF UNET on {devices['unet']}...")

    UNET, used = _load_on_device(
        candidates=("UnetLoaderGGUFMultiGPU",),
        device=devices["unet"],
        kwargs={"unet_name": unet_name},
        label="UNET",
        allow_plain_fallback="UnetLoaderGGUF",
        plain_kwargs={"unet_name": unet_name},
    )

    if verbose:
        print(f"✅ UNET loaded via {used} -> {devices['unet']}")

    clean_memory()

    # --------------------------------------------------------
    # CLIP
    # --------------------------------------------------------

    if verbose:
        print(f"\nLoading text encoder on {devices['clip']}...")

    if str(clip_name).lower().endswith(".gguf"):
        clip_candidates = ("CLIPLoaderGGUFMultiGPU",)
        clip_plain = "CLIPLoaderGGUF"
        clip_plain_kwargs = {"clip_name": clip_name, "type": "qwen_image"}
    else:
        clip_candidates = ("CLIPLoaderMultiGPU",)
        clip_plain = "CLIPLoader"
        # Core CLIPLoader needs device="default" -> only valid on 1 GPU (= cuda:0).
        clip_plain_kwargs = {
            "clip_name": clip_name,
            "type": "qwen_image",
            "device": "default",
        }

    CLIP, used = _load_on_device(
        candidates=clip_candidates,
        device=devices["clip"],
        kwargs={"clip_name": clip_name, "type": "qwen_image"},
        label="CLIP",
        allow_plain_fallback=clip_plain,
        plain_kwargs=clip_plain_kwargs,
    )

    if verbose:
        print(f"✅ CLIP loaded via {used} -> {devices['clip']}")

    clean_memory()

    # --------------------------------------------------------
    # VAE
    # --------------------------------------------------------

    if verbose:
        print(f"\nLoading VAE on {devices['vae']}...")

    VAE, used = _load_on_device(
        candidates=("VAELoaderMultiGPU",),
        device=devices["vae"],
        kwargs={"vae_name": vae_name},
        label="VAE",
        allow_plain_fallback="VAELoader",
        plain_kwargs={"vae_name": vae_name},
    )

    if verbose:
        print(f"✅ VAE loaded via {used} -> {devices['vae']}")

    DEVICE_MAP = dict(devices)
    _LOADED = True

    clean_memory()

    if verbose:
        print("\n" + "=" * 60)
        print("🔥 QWEN IMAGE 2.1 READY")
        print("=" * 60)
        print("Reported load devices:")
        print(f"  UNET : {_describe_device(UNET)}")
        print(f"  CLIP : {_describe_device(CLIP)}")
        print(f"  VAE  : {_describe_device(VAE)}")
        print_vram("After model load")

    return {
        "loaded": True,
        "already_loaded": False,
        "devices": dict(DEVICE_MAP),
        "gpus": get_gpu_info(),
    }


# ============================================================
# GENERATION
# ============================================================

def generate_image(
    prompt: str,
    negative_prompt: str = (
        "blurry, low quality, bad anatomy, deformed hands, "
        "extra fingers, bad proportions, lowres"
    ),
    width: int = 1024,
    height: int = 1024,
    steps: int = 28,
    cfg: float = 2.2,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    denoise: float = 1.0,
    seed: int | None = None,
    output_dir: str | os.PathLike = DEFAULT_OUTPUT_DIR,
    align_dimensions: bool = True,
    cleanup_after: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Generate and save one image.

    Returns:
        {"image": PIL.Image, "path": str, "seed": int, "width": int, "height": int}
    """
    if not _LOADED:
        raise RuntimeError(
            "Model is not loaded. Run: await qwen_text_to_image.load_model()"
        )

    if not prompt or not str(prompt).strip():
        raise ValueError("prompt cannot be empty")

    width = int(width)
    height = int(height)
    steps = int(steps)
    cfg = float(cfg)
    denoise = float(denoise)

    if align_dimensions:
        width = _align_dimension(width, 16)
        height = _align_dimension(height, 16)
    elif width % 16 != 0 or height % 16 != 0:
        raise ValueError(
            "width and height must be divisible by 16 when align_dimensions=False"
        )

    seed = random.randint(0, 9999999) if seed is None else int(seed)

    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = get_root_path() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("QWEN IMAGE 2.1 GENERATION")
        print("=" * 60)
        print("Size     :", f"{width}x{height}")
        print("Steps    :", steps)
        print("CFG      :", cfg)
        print("Sampler  :", sampler_name)
        print("Scheduler:", scheduler)
        print("Seed     :", seed)
        print("Devices  :", DEVICE_MAP)

    clean_memory()

    try:
        with torch.inference_mode():
            if verbose:
                print("\nEncoding prompt...")

            encoded = text_encoder.execute(
                clip=CLIP,
                prompt=str(prompt),
                negative_prompt=str(negative_prompt),
                vae=VAE,
                resolution=max(width, height),
                images=None,
            )

            positive = get_output(encoded, 0)
            negative = get_output(encoded, 1)

            # Encoder emits a default latent; discard it and build the exact size.
            latent = {
                "samples": torch.zeros(
                    (1, 64, height // 16, width // 16),
                    dtype=torch.float32,
                    device="cpu",
                )
            }

            if verbose:
                print(f"Sampling {width}x{height}...")

            sampled = sampler.sample(
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                model=UNET,
                positive=positive,
                negative=negative,
                latent_image=latent,
            )

            latent_result = get_output(sampled, 0)

            del encoded, positive, negative, latent, sampled
            clean_memory()

            if verbose:
                print("Decoding...")

            decoded = vae_decoder.decode(samples=latent_result, vae=VAE)
            images = get_output(decoded, 0)

            del decoded, latent_result

        image_np = images[0].detach().cpu().numpy()
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        output_path = output_dir / _make_filename(str(prompt), seed)
        pil_image.save(output_path, format="PNG")

        if verbose:
            print("\n✅ DONE")
            print("Saved :", output_path)

        return {
            "image": pil_image,
            "path": str(output_path),
            "seed": seed,
            "width": pil_image.width,
            "height": pil_image.height,
            "requested_width": width,
            "requested_height": height,
        }

    finally:
        # UNET / CLIP / VAE stay loaded for the next call.
        if cleanup_after:
            clean_memory()


# ============================================================
# UNLOAD
# ============================================================

def unload_model(verbose: bool = True) -> None:
    """Unload Qwen models and free GPU memory on all GPUs."""
    global UNET, CLIP, VAE
    global text_encoder, sampler, vae_decoder
    global DEVICE_MAP
    global _LOADED

    UNET = None
    CLIP = None
    VAE = None

    text_encoder = None
    sampler = None
    vae_decoder = None
    DEVICE_MAP = {}

    if model_management is not None:
        try:
            model_management.unload_all_models()
        except Exception:
            pass

        try:
            model_management.soft_empty_cache()
        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():
        for gpu_id in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(gpu_id):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass

    _LOADED = False

    if verbose:
        print("✅ Qwen Image 2.1 unloaded.")
        print_vram("After unload")


def is_loaded() -> bool:
    return _LOADED
