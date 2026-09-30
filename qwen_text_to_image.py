"""
qwen_text_to_image.py

Direct Python wrapper around ComfyUI + ComfyUI-GGUF for Qwen Image 2.1.

Notebook usage:
    import qwen_text_to_image as qwen

    await qwen.load_model()

    result = qwen.generate_image(
        prompt="cinematic anime hero in a neon city",
        negative_prompt="blurry, low quality",
        width=1280,
        height=720,
        steps=28,
        cfg=2.2,
    )

    print(result["path"])
    display(result["image"])

    qwen.unload_model()

Notes:
- Width/height are supplied directly. There is no built-in T4 resolution map.
- Dimensions are automatically aligned to multiples of 16 by default.
- Models stay loaded between generate_image() calls.
- Temporary tensors / CUDA cache are cleaned after each generation.
"""

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

_LOADED = False


# ============================================================
# ENVIRONMENT
# ============================================================

def detect_environment() -> str:
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass

    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"

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
    """
    Build:
      first_20_prompt_words_<short-uuid>.png
    """

    words = re.findall(r"[A-Za-z0-9]+", prompt)[:20]

    if not words:
        words = ["qwen_image"]

    slug = "_".join(words).lower()

    # Keep filenames reasonable even with long prompt words.
    slug = slug[:150].strip("_")

    short_uuid = uuid.uuid4().hex[:10]

    return f"{slug}_{short_uuid}_{seed}.png"


def _tensor_to_pil(images) -> Image.Image:
    tensor = images[0]

    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


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
# MODEL LOAD
# ============================================================

async def load_model(
    comfy_path: str | os.PathLike | None = None,
    unet_name: str = UNET_NAME,
    clip_name: str = CLIP_NAME,
    vae_name: str = VAE_NAME,
    prefer_multi_gpu: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Load Qwen Image 2.1 once.

    In Colab/Jupyter:
        await qwen.load_model()

    Single GPU:
        Uses normal ComfyUI loaders.

    Multiple GPUs:
        If compatible MultiGPU custom nodes are installed, the function
        tries to place the UNET on cuda:0 and auxiliary model on cuda:1.
        Otherwise it safely falls back to standard ComfyUI loading.
    """
    global COMFY_PATH
    global UNET, CLIP, VAE
    global text_encoder, sampler, vae_decoder
    global NODE_CLASS_MAPPINGS, model_management
    global _LOADED

    if _LOADED:
        if verbose:
            print("✅ Qwen Image 2.1 is already loaded.")
            print_vram("Current VRAM")
        return {
            "loaded": True,
            "already_loaded": True,
            "gpus": get_gpu_info(),
        }

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected.")

    COMFY_PATH = get_comfy_path(comfy_path)

    if not COMFY_PATH.is_dir():
        raise FileNotFoundError(
            f"ComfyUI directory not found: {COMFY_PATH}"
        )

    os.chdir(COMFY_PATH)

    comfy_str = str(COMFY_PATH)
    if comfy_str not in sys.path:
        sys.path.insert(0, comfy_str)

    if verbose:
        print("=" * 60)
        print("GPU DETECTION")
        print("=" * 60)

        gpu_count = torch.cuda.device_count()
        print("GPU count:", gpu_count)

        for item in get_gpu_info():
            print(
                f"GPU {item['id']}: {item['name']} | "
                f"{item['total_gb']:.2f} GB"
            )

    import nodes

    await nodes.init_extra_nodes(
        init_custom_nodes=True,
        init_api_nodes=False,
    )

    from nodes import (
        CLIPLoader,
        KSampler,
        NODE_CLASS_MAPPINGS as NODE_MAP,
        VAEDecode,
        VAELoader,
    )

    NODE_CLASS_MAPPINGS = NODE_MAP

    try:
        import comfy.model_management as _model_management
        model_management = _model_management
    except Exception:
        model_management = None

    required = [
        "UnetLoaderGGUF",
        "TextEncodeQwenImage21",
    ]

    for name in required:
        if name not in NODE_CLASS_MAPPINGS:
            raise RuntimeError(f"Missing required ComfyUI node: {name}")

    text_encoder = NODE_CLASS_MAPPINGS["TextEncodeQwenImage21"]()
    sampler = KSampler()
    vae_decoder = VAEDecode()

    clip_loader = CLIPLoader()
    vae_loader = VAELoader()

    gpu_count = torch.cuda.device_count()
    multi_gpu = bool(prefer_multi_gpu and gpu_count >= 2)

    # --------------------------------------------------------
    # UNET
    # --------------------------------------------------------

    if verbose:
        print("\nLoading GGUF model...")

    multi_unet_name = None

    if multi_gpu:
        for candidate in (
            "UnetLoaderGGUFMultiGPU",
            "UnetLoaderGGUFDisTorch2MultiGPU",
            "UnetLoaderGGUFDisTorchMultiGPU",
        ):
            if candidate in NODE_CLASS_MAPPINGS:
                multi_unet_name = candidate
                break

    if multi_unet_name:
        loader = NODE_CLASS_MAPPINGS[multi_unet_name]()

        input_info = loader.INPUT_TYPES()
        required_inputs = input_info.get("required", {})
        optional_inputs = input_info.get("optional", {})

        kwargs = {"unet_name": unet_name}

        if "device" in required_inputs or "device" in optional_inputs:
            kwargs["device"] = "cuda:0"

        UNET = get_output(loader.load_unet(**kwargs))

        if verbose:
            print(f"✅ UNET loaded with {multi_unet_name} on cuda:0")
    else:
        loader = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        UNET = get_output(
            loader.load_unet(
                unet_name=unet_name
            )
        )

        if verbose:
            print("✅ UNET loaded")

    # --------------------------------------------------------
    # CLIP
    # --------------------------------------------------------

    if verbose:
        print("\nLoading text encoder...")

    multi_clip_name = None

    if multi_gpu:
        for candidate in (
            "CLIPLoaderGGUFMultiGPU",
            "CLIPLoaderGGUFDisTorch2MultiGPU",
            "CLIPLoaderGGUFDisTorchMultiGPU",
        ):
            if candidate in NODE_CLASS_MAPPINGS:
                multi_clip_name = candidate
                break

    if multi_clip_name:
        loader = NODE_CLASS_MAPPINGS[multi_clip_name]()

        input_info = loader.INPUT_TYPES()
        required_inputs = input_info.get("required", {})
        optional_inputs = input_info.get("optional", {})

        kwargs = {
            "clip_name": clip_name,
            "type": "qwen_image",
        }

        if "device" in required_inputs or "device" in optional_inputs:
            kwargs["device"] = "cuda:1"

        CLIP = get_output(loader.load_clip(**kwargs))

        if verbose:
            print(f"✅ CLIP loaded with {multi_clip_name} on cuda:1")
    else:
        # On a single GPU, use default.
        # On multiple GPUs without a compatible explicit loader, use CPU
        # so cuda:0 remains available to the diffusion model.
        clip_device = "cpu" if multi_gpu else "default"

        CLIP = get_output(
            clip_loader.load_clip(
                clip_name=clip_name,
                type="qwen_image",
                device=clip_device,
            )
        )

        if verbose:
            print(f"✅ CLIP loaded ({clip_device})")

    # --------------------------------------------------------
    # VAE
    # --------------------------------------------------------

    if verbose:
        print("\nLoading VAE...")

    VAE = get_output(
        vae_loader.load_vae(
            vae_name=vae_name
        )
    )

    if verbose:
        print("✅ VAE loaded")

    _LOADED = True

    clean_memory()

    if verbose:
        print("\n" + "=" * 60)
        print("🔥 QWEN IMAGE 2.1 READY")
        print("=" * 60)
        print_vram("After model load")

    return {
        "loaded": True,
        "already_loaded": False,
        "multi_gpu": multi_gpu,
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

    Example:
        result = qwen.generate_image(
            prompt="cinematic anime hero",
            width=1280,
            height=720,
            steps=28,
            cfg=2.2,
        )

    Returns:
        {
            "image": PIL.Image.Image,
            "path": ".../saved_images/...",
            "seed": int,
            "width": int,
            "height": int,
        }
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
    else:
        if width % 16 != 0 or height % 16 != 0:
            raise ValueError(
                "width and height must be divisible by 16 "
                "when align_dimensions=False"
            )

    if seed is None:
        seed = random.randint(0, 9999999)
    else:
        seed = int(seed)

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

    clean_memory()

    encoded = None
    positive = None
    negative = None
    encoder_latent = None
    latent = None
    sampled = None
    latent_result = None
    decoded = None
    images = None
    image_tensor = None
    image_np = None
    pil_image = None

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

            # Qwen's text encoder also emits a default latent.
            # We discard it and create one using the exact requested size.
            encoder_latent = get_output(encoded, 2)

            latent = {
                "samples": torch.zeros(
                    (
                        1,
                        64,
                        height // 16,
                        width // 16,
                    ),
                    dtype=torch.float32,
                    device="cpu",
                )
            }

            del encoder_latent
            encoder_latent = None

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
            encoded = positive = negative = latent = sampled = None

            clean_memory()

            if verbose:
                print("Decoding...")

            decoded = vae_decoder.decode(
                samples=latent_result,
                vae=VAE,
            )

            images = get_output(decoded, 0)

            del decoded, latent_result
            decoded = latent_result = None

        image_tensor = images[0]
        image_np = image_tensor.detach().cpu().numpy()
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)

        pil_image = Image.fromarray(image_np)

        filename = _make_filename(str(prompt), seed)
        output_path = output_dir / filename

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
        # Remove only generation-time objects.
        # UNET / CLIP / VAE remain loaded for the next call.
        for name in (
            "encoded",
            "positive",
            "negative",
            "encoder_latent",
            "latent",
            "sampled",
            "latent_result",
            "decoded",
            "images",
            "image_tensor",
            "image_np",
        ):
            if name in locals():
                try:
                    del locals()[name]
                except Exception:
                    pass

        if cleanup_after:
            clean_memory()


# ============================================================
# UNLOAD
# ============================================================

def unload_model(verbose: bool = True) -> None:
    """Unload Qwen models and free GPU memory."""
    global UNET, CLIP, VAE
    global text_encoder, sampler, vae_decoder
    global _LOADED

    UNET = None
    CLIP = None
    VAE = None

    text_encoder = None
    sampler = None
    vae_decoder = None

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
