"""DIPAI image generation - text-to-image with a pretrained diffusion model.

This module is deliberately SEPARATE from the text pipeline (mirroring
dipai.vision for image understanding):
  * It loads its own pretrained diffusion pipeline (default: Stable Diffusion
    1.5, CreativeML OpenRAIL++-S) with the official diffusers API.
  * The DIPAI text checkpoint, tokenizer, and vocabulary are NEVER touched.
  * Nothing is loaded until load_image_gen() or the first generation request,
    so text-only chat/training works unchanged - even without diffusers or
    the model weights installed (every heavy import is lazy).

Run (image-generation CLI):
    python -m dipai.imagegen --prompt "a glass of iced milk tea, 50mm lens"
    # or through the web UI: python -m webui.app  -> "Image generation" tab

Interfaces:
    load_image_gen(config=None, *, force=False) -> ImageGenStatus
    generate_image(prompt, ...) -> PIL.Image.Image
    save_image(image, out_path, metadata=None) -> Path
    download_model(config=None, force=False) -> Path
    get_imagegen_status() -> ImageGenStatus
    unload_image_gen() -> None

Honesty notes (do not over-claim):
  * Quality depends on the model, prompt, settings, and hardware. Prompt
    wording alone does not guarantee photorealism.
  * On a 4 GB GPU the practical ceiling is SD 1.5 at <=768x768. SDXL and
    larger models will not fit.
  * This module INFERS with a pretrained model; it does not train one.

Privacy: prompts and images are processed locally. They are never uploaded,
never sent to any external service, and prompts are never written to logs
(only sizes/devices/status are logged). Hugging Face access happens solely
to download model WEIGHTS (and can be disabled with local_files_only: true
or allow_download: false).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from dipai.config import (
    ConfigError,
    ImageGenConfig,
    default_imagegen_config,
    load_imagegen_config,
)
from dipai.hardware import get_device, probe_hardware

log = logging.getLogger("dipai.imagegen")

DEFAULT_CONFIG_PATH = "configs/imagegen.yaml"


class ImageGenError(Exception):
    """Raised for invalid input, model load failures, or inference errors."""


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@dataclass
class ImageGenStatus:
    """Snapshot of the image-generation module state (shown in chat/web UI)."""

    loaded: bool = False
    loading: bool = False
    model_id: str = ""
    device: str = ""
    dtype: str = ""
    params: int = 0
    load_seconds: float = 0.0
    generations: int = 0
    last_image_path: str = ""
    load_error: str = ""
    last_error: str = ""
    notes: str = ""
    # download / disk
    weights_cached: bool = False
    free_disk_gb: float = 0.0

    def line(self) -> str:
        if self.loading:
            return f"imagegen: loading {self.model_id.rsplit('/', 1)[-1]}..."
        if not self.loaded:
            base = "imagegen: not loaded"
            if self.load_error:
                return f"{base} (error: {self.load_error})"
            return f"{base} (loads on first image request)"
        name = self.model_id.rsplit("/", 1)[-1]
        out = (f"imagegen: {name} | {self.params / 1e6:.0f}M | "
               f"{self.device.upper()} {self.dtype}")
        if self.notes:
            out += f" | {self.notes}"
        if self.last_error:
            out += f" | last error: {self.last_error}"
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loaded": self.loaded,
            "loading": self.loading,
            "model_id": self.model_id,
            "device": self.device,
            "dtype": self.dtype,
            "params_m": round(self.params / 1e6, 1),
            "load_seconds": round(self.load_seconds, 2),
            "generations": self.generations,
            "last_image_path": self.last_image_path,
            "load_error": self.load_error,
            "last_error": self.last_error,
            "notes": self.notes,
            "weights_cached": self.weights_cached,
            "free_disk_gb": round(self.free_disk_gb, 2),
        }


@dataclass
class ImageGenRequest:
    """One generation request; all fields optional except the prompt."""

    prompt: str
    negative_prompt: str = ""
    width: int = 0            # 0 -> use config default
    height: int = 0
    steps: int = 0            # 0 -> config default
    guidance_scale: float = 0.0   # 0.0 -> config default
    seed: int = -1            # -1 -> random each call

    def describe(self) -> str:
        return (f"{self.width}x{self.height} steps={self.steps} "
                f"cfg={self.guidance_scale} seed={self.seed}")


# ---------------------------------------------------------------------------
# Module-level singleton (one pipeline at a time - 4 GB VRAM budget)
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {
    "pipe": None,
    "status": ImageGenStatus(),
    "config": None,
    "device": "cpu",
    "dtype_name": "float32",
}


def get_imagegen_status() -> ImageGenStatus:
    """Return the current module status (refreshing disk info)."""
    st: ImageGenStatus = _STATE["status"]
    try:
        usage = shutil.disk_usage(Path.cwd())
        st.free_disk_gb = usage.free / 1024**3
    except Exception:  # noqa: BLE001
        pass
    return st


def is_loaded() -> bool:
    return _STATE["pipe"] is not None


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

def _estimate_vram_bytes(width: int, height: int, steps: int) -> int:
    """Rough peak-VRAM estimate for SD1.5 inference (fp16).

    Based on measured behaviour of SD1.5 on 4 GB GPUs: weights ~2.2 GB fp16,
    activations scale with pixels, plus ~0.4 GB CUDA/context overhead.
    This is an ESTIMATE for preflight messaging, not a guarantee.
    """
    weights = int(2.2 * 1024**3)
    pixels = max(64, width) * max(64, height)
    # ~0.9 GB at 512x512; scale roughly with pixel count.
    activations = int(0.9 * 1024**3 * pixels / (512 * 512))
    overhead = int(0.4 * 1024**3)
    # More steps -> slightly more peak from attention over longer schedules.
    step_factor = 1.0 + min(steps, 60) * 0.002
    return int((weights + activations + overhead) * step_factor)


def estimate_imagegen_vram_bytes(
    width: int = 512, height: int = 512, steps: int = 25
) -> Dict[str, int]:
    """Public estimator used by tests and the UI."""
    total = _estimate_vram_bytes(width, height, steps)
    return {
        "width": width,
        "height": height,
        "steps": steps,
        "total_bytes": total,
    }


def _check_disk_for_download(cfg: ImageGenConfig) -> float:
    """Ensure enough free disk for the weights cache; return free GB."""
    free_gb = shutil.disk_usage(Path.cwd()).free / 1024**3
    if free_gb < cfg.min_free_disk_gb:
        raise ImageGenError(
            f"not enough free disk to download model weights: "
            f"{free_gb:.1f} GB free < {cfg.min_free_disk_gb:.1f} GB required. "
            "Free up space or set imagegen.min_free_disk_gb lower if you "
            "know the cache already exists."
        )
    return free_gb


def _hf_cache_has_weights(model_id: str) -> bool:
    """Best-effort check whether the model is already in the HF cache."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            model_id,
            local_files_only=True,
            allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors",
                            "*.vae/*", "*.fp16/*"],
        )
        return True
    except Exception:  # noqa: BLE001 - not cached (or hub unreachable offline)
        return False


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _pick_dtype(cfg: ImageGenConfig, device: str):
    import torch

    if cfg.dtype == "float16":
        return torch.float16
    if cfg.dtype == "bfloat16":
        return torch.bfloat16
    if cfg.dtype == "float32":
        return torch.float32
    # auto
    if device == "cuda":
        try:
            if torch.cuda.is_bf16_supported():
                # SD pipelines are most battle-tested in fp16; prefer fp16
                # for compatibility unless the user forces bf16.
                return torch.float16
        except Exception:  # noqa: BLE001
            pass
        return torch.float16
    return torch.float32


# Files without which the pipeline cannot run (checked to decide whether a
# network download is needed at all - a load must never be blocked by the
# free-disk download policy when the cache is already complete).
_CRITICAL_WEIGHT_FILES = (
    "model_index.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
)


def _critical_weights_present(model_id: str) -> bool:
    """True when all critical pipeline files exist in the local HF cache."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            model_id,
            local_files_only=True,
            allow_patterns=list(_CRITICAL_WEIGHT_FILES),
        )
        return True
    except Exception:  # noqa: BLE001 - any miss means "not complete"
        return False


def download_model(
    config: Optional[ImageGenConfig] = None,
) -> Path:
    """Download (or resume) model weights WITHOUT loading the pipeline.

    Downloads stream to disk and need almost no RAM, so this works on
    machines that cannot yet afford to load the model. Raises ImageGenError
    with a clear message on insufficient disk or network failure.
    """
    cfg = config or default_imagegen_config()
    cfg.validate()
    st = get_imagegen_status()
    st.model_id = cfg.model_id
    if not cfg.allow_download:
        raise ImageGenError(
            "downloads are disabled (imagegen.allow_download: false). "
            "Set it to true or pre-populate the Hugging Face cache."
        )
    # When critical weights are already cached but free disk is below the
    # download policy, succeed with the local cache instead of refusing -
    # there is nothing essential left to fetch. A real miss still hits the
    # disk gate inside _download_weights and raises.
    if _critical_weights_present(cfg.model_id):
        try:
            path = _download_weights(cfg, st)
        except ImageGenError as exc:
            if "free disk" in str(exc).lower():
                from huggingface_hub import snapshot_download

                path = snapshot_download(
                    cfg.model_id,
                    local_files_only=True,
                    allow_patterns=list(_CRITICAL_WEIGHT_FILES),
                )
                log.info("using existing cache (download skipped: %s)", exc)
            else:
                raise
    else:
        path = _download_weights(cfg, st)
    st.weights_cached = True
    return path


def load_image_gen(
    config: Optional[ImageGenConfig] = None,
    *,
    force: bool = False,
    device: Optional[str] = None,
) -> ImageGenStatus:
    """Load the diffusion pipeline (lazy; safe to call when already loaded).

    Raises ImageGenError with a human-readable message on failure. Never
    fabricates a pipeline object: if load fails, status.loaded stays False.
    """
    with _LOCK:
        if _STATE["pipe"] is not None and not force:
            return _STATE["status"]
        if _STATE["status"].loading and not force:
            # Another thread is loading; poll briefly (single-user app).
            for _ in range(600):
                if _STATE["pipe"] is not None:
                    return _STATE["status"]
                time.sleep(0.1)
            raise ImageGenError("timed out waiting for image model load")

        st = ImageGenStatus(loading=True)
        _STATE["status"] = st
        t0 = time.perf_counter()
        try:
            cfg = config or default_imagegen_config()
            cfg.validate()
            _STATE["config"] = cfg
            st.model_id = cfg.model_id

            # Record free disk for status; the download path performs its own
            # space check only when weights are actually missing (a load of an
            # already-cached model must not be blocked by free-disk policy).
            try:
                st.free_disk_gb = shutil.disk_usage(Path.cwd()).free / 1024**3
            except Exception:  # noqa: BLE001
                pass

            dev = device or get_device(cfg.device)
            torch_dtype = _pick_dtype(cfg, dev)

            # RAM sanity: loading SD1.5 needs ~2 GB free for weights even
            # with low_cpu_mem_usage. Warn loudly on tiny machines but allow
            # the attempt (Windows paging may still succeed, slowly); a real
            # failure surfaces as a clear load error below.
            hw = probe_hardware(check_bnb=False)
            if hw.ram_available_gb < 1.5:
                log.warning(
                    "only %.2f GB system RAM free; loading Stable Diffusion "
                    "typically needs ~2 GB. The load may be slow (paging) or "
                    "fail - close other applications if it does.",
                    hw.ram_available_gb,
                )

            local_only = cfg.local_files_only or not cfg.allow_download
            if not local_only:
                # Download only when critical weights are missing; a load of
                # an already-cached model must not hit the network or the
                # free-disk download gate.
                if _critical_weights_present(cfg.model_id):
                    st.weights_cached = True
                    log.info("model weights already cached; skipping download")
                else:
                    _download_weights(cfg, st)

            try:
                pipe = _build_pipeline(cfg, dev, torch_dtype, local_only)
            except Exception as exc:  # noqa: BLE001
                if dev == "cuda" and cfg.cpu_fallback:
                    log.warning(
                        "CUDA image pipeline load failed (%s); retrying on CPU",
                        exc,
                    )
                    gc.collect()
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass
                    dev = "cpu"
                    torch_dtype = _pick_dtype(cfg, dev)
                    pipe = _build_pipeline(cfg, dev, torch_dtype, local_only=True)
                else:
                    raise

            # Memory-efficiency toggles (4 GB GPU).
            try:
                if cfg.enable_attention_slicing:
                    pipe.enable_attention_slicing()
                if cfg.enable_vae_slicing:
                    pipe.enable_vae_slicing()
                if cfg.enable_vae_tiling:
                    pipe.enable_vae_tiling()
            except Exception as exc:  # noqa: BLE001 - older/newer diffusers
                log.warning("optional memory optimisation skipped: %s", exc)

            _STATE["pipe"] = pipe
            _STATE["device"] = dev
            _STATE["dtype_name"] = str(torch_dtype).replace("torch.", "")

            n_params = sum(p.numel() for p in pipe.unet.parameters())
            try:
                n_params += sum(p.numel() for p in pipe.vae.parameters())
            except Exception:  # noqa: BLE001
                pass

            st.loaded = True
            st.loading = False
            st.device = dev
            st.dtype = _STATE["dtype_name"]
            st.params = int(n_params)
            st.load_seconds = time.perf_counter() - t0
            st.weights_cached = True
            notes = []
            if cfg.enable_attention_slicing:
                notes.append("attn-slice")
            if cfg.enable_vae_slicing:
                notes.append("vae-slice")
            if dev == "cpu":
                notes.append("CPU (slow)")
            st.notes = ", ".join(notes)
            log.info("imagegen ready: %s", st.line())
            return st
        except ImageGenError as exc:
            st.loading = False
            st.loaded = False
            st.load_error = str(exc)
            _STATE["pipe"] = None
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
            st.loading = False
            st.loaded = False
            st.load_error = f"{type(exc).__name__}: {exc}"
            _STATE["pipe"] = None
            raise ImageGenError(f"failed to load image model: {exc}") from exc


def _download_weights(cfg: ImageGenConfig, st: ImageGenStatus) -> Path:
    """Explicit weight download with progress logging. Returns cache path.

    Downloads only what the diffusers pipeline needs: the giant single-file
    checkpoints (v1-5-pruned*.safetensors/*.ckpt, ~4 GB each, for A1111
    WebUI) and redundant fp32 .bin copies are IGNORED to keep disk usage and
    download time sane. Already-cached files are skipped (resume).
    """
    from huggingface_hub import snapshot_download

    free_before = _check_disk_for_download(cfg)
    st.free_disk_gb = free_before
    log.info(
        "downloading image model %s (need ~2-4 GB, %.1f GB free) ...",
        cfg.model_id,
        free_before,
    )
    try:
        path = snapshot_download(
            cfg.model_id,
            resume_download=True,
            ignore_patterns=[
                # WebUI-only full checkpoints (never used by diffusers):
                "v1-5-pruned*", "*.ckpt",
                # Redundant legacy formats; safetensors variants suffice:
                "unet/diffusion_pytorch_model.bin",
                "unet/diffusion_pytorch_model.fp16.bin",
                "unet/diffusion_pytorch_model.non_ema.bin",
                "vae/diffusion_pytorch_model.bin",
                "vae/diffusion_pytorch_model.fp16.bin",
            ],
        )
    except Exception as exc:
        raise ImageGenError(
            f"model download failed for {cfg.model_id!r}: {exc}. "
            "Check your network connection, or pre-download the weights and "
            "set imagegen.local_files_only: true."
        ) from exc
    free_after = shutil.disk_usage(Path.cwd()).free / 1024**3
    st.free_disk_gb = free_after
    log.info("model cached at %s (%.1f GB disk used by download)",
             path, free_before - free_after)
    return Path(path)


def _build_pipeline(cfg: ImageGenConfig, device: str, torch_dtype, local_only: bool):
    """Construct the diffusers pipeline (heavy imports stay lazy)."""
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise ImageGenError(f"PyTorch is required for image generation: {exc}") from exc
    try:
        from diffusers import AutoPipelineForText2Image
    except Exception as exc:  # noqa: BLE001
        raise ImageGenError(
            f"diffusers is not installed or failed to import: {exc}. "
            "Install with: pip install 'diffusers>=0.31,<0.36'"
        ) from exc

    log.info("building image pipeline %s on %s (%s, local_only=%s)",
             cfg.model_id, device, torch_dtype, local_only)

    def _load(variant=None):
        kwargs: Dict[str, Any] = dict(
            torch_dtype=torch_dtype,
            safety_checker=None,          # local offline tool; no NSFW filter
            requires_safety_checker=False,
            low_cpu_mem_usage=cfg.low_cpu_mem_usage,
            local_files_only=local_only,
        )
        if variant is not None:
            kwargs["variant"] = variant
        return AutoPipelineForText2Image.from_pretrained(cfg.model_id, **kwargs)

    try:
        # fp16 variant first when running in fp16: halves the bytes read off
        # disk (and thus peak load RAM) vs. loading fp32 weights and casting.
        if torch_dtype == torch.float16:
            try:
                pipe = _load(variant="fp16")
            except Exception:  # noqa: BLE001 - variant files may not exist
                log.info("fp16 variant unavailable; loading default weights")
                pipe = _load()
        else:
            pipe = _load()
    except OSError as exc:
        raise ImageGenError(
            f"model weights for {cfg.model_id!r} not found locally and could "
            f"not be downloaded: {exc}. Run "
            "`python -m dipai.imagegen --download-only` or enable downloads."
        ) from exc
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def unload_image_gen() -> None:
    """Free the pipeline and VRAM. Safe to call when not loaded."""
    with _LOCK:
        pipe = _STATE.get("pipe")
        _STATE["pipe"] = None
        if pipe is not None:
            try:
                del pipe
            except Exception:  # noqa: BLE001
                pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        st = _STATE["status"]
        st.loaded = False
        st.notes = ""
        log.info("imagegen unloaded")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_image(
    prompt: str,
    *,
    config: Optional[ImageGenConfig] = None,
    negative_prompt: Optional[str] = None,
    width: int = 0,
    height: int = 0,
    steps: int = 0,
    guidance_scale: float = 0.0,
    seed: int = -1,
    progress_fn: Optional[Callable[[float], None]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Generate an image from a text prompt. Returns (PIL.Image, metadata).

    The pipeline is loaded on first use (lazy). Settings are clamped to the
    config's safe caps. `seed >= 0` makes the result reproducible on the
    same device/model/settings; `seed < 0` draws a fresh random seed.
    Raises ImageGenError for empty prompts, load failures, or OOM.
    """
    if not prompt or not str(prompt).strip():
        raise ImageGenError(
            "prompt is empty - describe the image you want "
            "(e.g. 'a glass of iced milk tea on a wooden table, 50mm lens')"
        )
    prompt = str(prompt).strip()
    if len(prompt) > 2000:
        raise ImageGenError(
            f"prompt is too long ({len(prompt)} chars; max 2000). "
            "Shorten the description."
        )

    cfg = config or (_STATE.get("config") or default_imagegen_config())
    cfg.validate()

    # Clamp into safe bounds.
    w, h = cfg.clamp_size(width or cfg.default_width, height or cfg.default_height)
    n_steps = cfg.clamp_steps(steps or cfg.default_steps)
    gscale = float(guidance_scale) if guidance_scale and guidance_scale > 0 \
        else float(cfg.default_guidance_scale)
    neg = negative_prompt if negative_prompt is not None else cfg.default_negative_prompt

    st = get_imagegen_status()
    # Preflight VRAM message (estimate only; real OOM still handled below).
    est = _estimate_vram_bytes(w, h, n_steps)
    if st.loaded and st.device == "cuda":
        hw = probe_hardware(check_bnb=False)
        if est > hw.vram_free_bytes * 1.1:
            log.warning(
                "estimated peak %.2f GB may exceed free VRAM %.2f GB; "
                "consider a smaller size or unloading other GPU users",
                est / 1024**3, hw.vram_free_gb,
            )

    if not is_loaded():
        load_image_gen(cfg)

    pipe = _STATE["pipe"]
    if pipe is None:
        raise ImageGenError("image pipeline failed to load (see status)")

    import torch

    # Resolve seed (store the actual seed used in metadata).
    actual_seed = int(seed) if seed is not None and int(seed) >= 0 \
        else int(torch.seed() % (2**31 - 1))
    generator = torch.Generator(device="cpu").manual_seed(actual_seed)

    meta: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": neg,
        "width": w,
        "height": h,
        "steps": n_steps,
        "guidance_scale": gscale,
        "seed": actual_seed,
        "model_id": cfg.model_id,
        "device": _STATE.get("device", "cpu"),
        "dtype": _STATE.get("dtype_name", "float32"),
        "generator": "dipai.imagegen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "estimated_vram_bytes": est,
    }

    def _cb(pipe_obj, step_index, timestep, callback_kwargs):  # diffusers >= 0.31
        if progress_fn is not None:
            try:
                total = callback_kwargs.get("num_inference_steps") or n_steps
                progress_fn(min(1.0, (step_index + 1) / max(1, total)))
            except Exception:  # noqa: BLE001 - progress must not kill gen
                pass
        return callback_kwargs

    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=neg or None,
                width=w,
                height=h,
                num_inference_steps=n_steps,
                guidance_scale=gscale,
                generator=generator,
                callback_on_step_end=_cb if progress_fn else None,
            )
        image = result.images[0]
    except torch.cuda.OutOfMemoryError as exc:
        st.last_error = "CUDA out of memory"
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        raise ImageGenError(
            f"CUDA out of memory generating {w}x{h}. Try a smaller size "
            "(e.g. 512x512), fewer steps, or close other GPU applications. "
            "Original: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        st.last_error = f"{type(exc).__name__}: {exc}"
        raise ImageGenError(f"image generation failed: {exc}") from exc

    elapsed = time.perf_counter() - t0
    meta["generation_seconds"] = round(elapsed, 2)
    st.generations += 1
    st.last_error = ""
    log.info(
        "generated %dx%d in %.1fs (steps=%d seed=%d)",
        w, h, elapsed, n_steps, actual_seed,
    )
    return image, meta


def save_image(
    image: Any,
    out_path: Optional[str | Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    output_dir: Optional[str | Path] = None,
) -> Path:
    """Save a PIL image (optionally with a JSON metadata sidecar).

    When `out_path` is None, a timestamped filename is created inside
    `output_dir` (or the config default). Returns the final PNG path.
    """
    from PIL import Image  # lazy

    if not isinstance(image, Image.Image):
        raise ImageGenError(f"expected a PIL.Image, got {type(image).__name__}")

    if out_path is None:
        base = Path(output_dir) if output_dir else Path(
            (_STATE.get("config") or default_imagegen_config()).output_dir
        )
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        seed_part = f"_seed{metadata['seed']}" if metadata and "seed" in metadata else ""
        out_path = base / f"dipai_{stamp}{seed_part}.png"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    image.save(out_path, format="PNG")
    if metadata:
        # Embed a compact subset as PNG text chunks (viewable in many tools).
        try:
            from PIL import PngImagePlugin

            pnginfo = PngImagePlugin.PngInfo()
            for key in ("prompt", "model_id", "seed"):
                if key in metadata:
                    pnginfo.add_text(key, str(metadata[key]))
            image.save(out_path, format="PNG", pnginfo=pnginfo)
        except Exception:  # noqa: BLE001 - sidecar still written below
            pass
        if (_STATE.get("config") or default_imagegen_config()).save_metadata:
            sidecar = out_path.with_suffix(".json")
            try:
                sidecar.write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("could not write metadata sidecar: %s", exc)

    _STATE["status"].last_image_path = str(out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m dipai.imagegen",
        description="DIPAI text-to-image generation (Stable Diffusion 1.5)",
    )
    ap.add_argument("--prompt", "-p", default=None,
                    help="text description of the image")
    ap.add_argument("--negative-prompt", "-n", default=None,
                    help="things to avoid (default from config)")
    ap.add_argument("--out", "-o", default=None,
                    help="output PNG path (default: timestamped in output_dir)")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--guidance", type=float, default=0.0, dest="guidance_scale")
    ap.add_argument("--seed", type=int, default=-1,
                    help=">=0 for reproducible output; -1 = random")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                    help="imagegen YAML config (default: configs/imagegen.yaml)")
    ap.add_argument("--device", default=None, choices=[None, "auto", "cuda", "cpu"])
    ap.add_argument("--download-only", action="store_true",
                    help="download model weights and exit (no generation)")
    ap.add_argument("--no-progress", action="store_true",
                    help="hide the step progress bar")
    ap.add_argument("--status", action="store_true",
                    help="print module status and exit")
    ap.add_argument("--list-outputs", action="store_true",
                    help="list images in the output directory and exit")
    return ap


def main(argv: Optional[list] = None) -> int:
    from dipai.utils import setup_logging

    ap = _build_arg_parser()
    args = ap.parse_args(argv)
    setup_logging()

    try:
        cfg = load_imagegen_config(args.config, required=False)
    except ConfigError as exc:
        print(f"[error] {exc}")
        return 2

    if args.device:
        cfg.device = args.device

    if args.status:
        st = get_imagegen_status()
        st.model_id = cfg.model_id
        st.weights_cached = _critical_weights_present(cfg.model_id)
        try:
            st.free_disk_gb = shutil.disk_usage(Path.cwd()).free / 1024**3
        except OSError:
            pass
        print(st.line())
        print(json.dumps(st.to_dict(), indent=2))
        return 0

    if args.list_outputs:
        out_dir = Path(cfg.output_dir)
        if not out_dir.exists():
            print(f"(no outputs yet: {out_dir} does not exist)")
            return 0
        files = sorted(out_dir.glob("*.png"))
        for f in files:
            print(f"{f}  ({f.stat().st_size / 1024:.0f} KB)")
        print(f"{len(files)} image(s) in {out_dir}")
        return 0

    if args.download_only:
        try:
            path = download_model(cfg)
            print(f"model weights cached at: {path}")
            st = get_imagegen_status()
            print(f"free disk after download: {st.free_disk_gb:.1f} GB")
            return 0
        except ImageGenError as exc:
            print(f"[error] {exc}")
            return 1

    if not args.prompt:
        ap.error("--prompt is required unless --download-only/--status/--list-outputs")

    try:
        def _progress(frac: float) -> None:
            bar = "#" * int(frac * 30)
            print(f"\r  [{bar:<30}] {frac * 100:5.1f}%", end="", flush=True)

        image, meta = generate_image(
            args.prompt,
            config=cfg,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            progress_fn=_progress if not args.no_progress else None,
        )
        print()  # newline after progress bar
        path = save_image(image, args.out, meta, output_dir=cfg.output_dir)
        print(f"saved: {path}")
        print(f"seed={meta['seed']} size={meta['width']}x{meta['height']} "
              f"steps={meta['steps']} in {meta.get('generation_seconds', '?')}s")
        return 0
    except ImageGenError as exc:
        print(f"\n[error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
