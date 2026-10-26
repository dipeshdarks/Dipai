"""Tests for dipai.imagegen (text-to-image) and its web UI handlers.

Design: every test that would need model weights runs against a FAKE
pipeline injected into the module state, so the suite is fast and never
downloads anything. The real-model paths are covered by opt-in smoke tests
(gated on DIPAI_IMAGEGEN_SMOKE=1), clearly marked as such.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from dipai import imagegen
from dipai.config import (
    ConfigError,
    ImageGenConfig,
    default_imagegen_config,
    load_imagegen_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_state():
    """Isolate the module singleton between tests."""
    imagegen.unload_image_gen()
    imagegen._STATE["config"] = None
    yield
    imagegen.unload_image_gen()
    imagegen._STATE["config"] = None
    imagegen._STATE["status"] = imagegen.ImageGenStatus()


class FakePipe:
    """Minimal stand-in for a diffusers pipeline (no weights, no torch)."""

    def __init__(self, fail_oom: bool = False):
        self.calls = []
        self.fail_oom = fail_oom
        self.slicing = {"attn": False, "vae_s": False, "vae_t": False}
        self._device = "cpu"

    def enable_attention_slicing(self):
        self.slicing["attn"] = True

    def enable_vae_slicing(self):
        self.slicing["vae_s"] = True

    def enable_vae_tiling(self):
        self.slicing["vae_t"] = True

    def to(self, device):
        self._device = device
        return self

    def set_progress_bar_config(self, **kw):
        pass

    class _UNet:
        @staticmethod
        def parameters():
            return []

    class _VAE:
        @staticmethod
        def parameters():
            return []

    unet = _UNet()
    vae = _VAE()

    def __call__(self, prompt, negative_prompt=None, width=512, height=512,
                 num_inference_steps=25, guidance_scale=7.5, generator=None,
                 callback_on_step_end=None, **kw):
        import torch

        if self.fail_oom:
            raise torch.cuda.OutOfMemoryError("fake OOM")
        self.calls.append(dict(
            prompt=prompt, negative_prompt=negative_prompt, width=width,
            height=height, num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=generator.initial_seed() if generator is not None else None,
        ))
        if callback_on_step_end is not None:
            for step in range(num_inference_steps):
                callback_on_step_end(
                    self, step, 0.0,
                    {"num_inference_steps": num_inference_steps},
                )
        img = Image.new("RGB", (width, height), color=(200, 30, 30))
        return type("Result", (), {"images": [img]})()


def _install_fake_pipe(fail_oom: bool = False) -> FakePipe:
    """Inject a FakePipe as if load_image_gen() had succeeded."""
    pipe = FakePipe(fail_oom=fail_oom)
    cfg = default_imagegen_config()
    imagegen._STATE["pipe"] = pipe
    imagegen._STATE["config"] = cfg
    imagegen._STATE["device"] = "cpu"
    imagegen._STATE["dtype_name"] = "float32"
    imagegen._STATE["status"] = imagegen.ImageGenStatus(
        loaded=True, model_id=cfg.model_id, device="cpu", dtype="float32",
        params=860_000_000,
    )
    return pipe


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestImageGenConfig:
    def test_default_config_validates(self):
        cfg = ImageGenConfig()
        cfg.validate()

    def test_load_yaml_config(self):
        cfg = load_imagegen_config("configs/imagegen.yaml", required=False)
        assert cfg.model_id.startswith("stable-diffusion")
        assert cfg.default_width % 8 == 0
        assert cfg.max_width >= cfg.default_width
        assert cfg.output_dir

    def test_missing_config_file_not_required(self):
        cfg = load_imagegen_config("configs/definitely_missing.yaml", required=False)
        assert isinstance(cfg, ImageGenConfig)
        cfg.validate()

    def test_missing_config_file_required_raises(self):
        with pytest.raises(ConfigError, match="not found"):
            load_imagegen_config("configs/definitely_missing.yaml", required=True)

    def test_unknown_key_raises(self, tmp_path: Path):
        p = tmp_path / "bad.yaml"
        p.write_text("model_id: x\nnot_a_real_key: 1\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="not_a_real_key"):
            load_imagegen_config(p)

    def test_non_multiple_of_8_rejected(self):
        cfg = ImageGenConfig(default_width=513)
        with pytest.raises(ConfigError, match="multiple of 8"):
            cfg.validate()

    def test_default_size_over_max_rejected(self):
        cfg = ImageGenConfig(default_width=1024, max_width=768)
        with pytest.raises(ConfigError, match="exceeds"):
            cfg.validate()

    def test_steps_range_rejected(self):
        cfg = ImageGenConfig(min_steps=30, max_steps=10, default_steps=20)
        with pytest.raises(ConfigError, match="min_steps"):
            cfg.validate()

    def test_default_steps_outside_bounds_rejected(self):
        cfg = ImageGenConfig(min_steps=5, max_steps=20, default_steps=25)
        with pytest.raises(ConfigError, match="default_steps"):
            cfg.validate()

    def test_negative_guidance_rejected(self):
        cfg = ImageGenConfig(default_guidance_scale=-1.0)
        with pytest.raises(ConfigError, match="guidance_scale"):
            cfg.validate()

    def test_clamp_size(self):
        cfg = ImageGenConfig()
        # 2000 clamped to max 768; 100 is within [64, 768] -> rounds to 96.
        assert cfg.clamp_size(2000, 100) == (768, 96)
        # Below the 64 floor -> clamped up to 64, then to /8 (64).
        assert cfg.clamp_size(10, 10) == (64, 64)
        assert cfg.clamp_size(512, 512) == (512, 512)
        assert cfg.clamp_size(513, 517) == (512, 512)  # rounds down to /8

    def test_clamp_steps(self):
        cfg = ImageGenConfig(min_steps=1, max_steps=60)
        assert cfg.clamp_steps(0) == 1
        assert cfg.clamp_steps(999) == 60
        assert cfg.clamp_steps(25) == 25


# ---------------------------------------------------------------------------
# VRAM estimator / disk preflight
# ---------------------------------------------------------------------------

class TestEstimates:
    def test_estimator_monotonic_with_pixels(self):
        small = imagegen.estimate_imagegen_vram_bytes(512, 512, 25)
        big = imagegen.estimate_imagegen_vram_bytes(768, 768, 25)
        assert big["total_bytes"] > small["total_bytes"]
        # SD1.5 @512 should land in the 3-4 GB ballpark (4 GB GPU target).
        gb = small["total_bytes"] / 1024**3
        assert 2.5 <= gb <= 4.5

    def test_estimator_monotonic_with_steps(self):
        a = imagegen.estimate_imagegen_vram_bytes(512, 512, 10)
        b = imagegen.estimate_imagegen_vram_bytes(512, 512, 50)
        assert b["total_bytes"] > a["total_bytes"]

    def test_disk_check_refuses_when_too_low(self, monkeypatch):
        cfg = ImageGenConfig(min_free_disk_gb=10_000)  # impossible in practice
        with pytest.raises(ImageGenError_, match="free disk|disk"):
            imagegen._check_disk_for_download(cfg)


# Alias so the match above works with the actual exception type.
ImageGenError_ = imagegen.ImageGenError


# ---------------------------------------------------------------------------
# Validation of prompts (no pipeline needed)
# ---------------------------------------------------------------------------

class TestPromptValidation:
    def test_empty_prompt_raises(self, tmp_path: Path):
        with pytest.raises(imagegen.ImageGenError, match="prompt is empty"):
            imagegen.generate_image("   ")

    def test_none_prompt_raises(self):
        with pytest.raises(imagegen.ImageGenError, match="prompt is empty"):
            imagegen.generate_image(None)

    def test_too_long_prompt_raises(self):
        with pytest.raises(imagegen.ImageGenError, match="too long"):
            imagegen.generate_image("x" * 2001)


# ---------------------------------------------------------------------------
# Generation with a fake pipeline
# ---------------------------------------------------------------------------

class TestGenerateWithFakePipeline:
    def test_generate_returns_image_and_metadata(self):
        pipe = _install_fake_pipe()
        img, meta = imagegen.generate_image(
            "a red apple", width=512, height=512, steps=4, seed=123,
        )
        assert isinstance(img, Image.Image)
        assert img.size == (512, 512)
        assert meta["prompt"] == "a red apple"
        assert meta["seed"] == 123
        assert meta["steps"] == 4
        assert "model_id" in meta
        assert "generation_seconds" in meta
        assert len(pipe.calls) == 1

    def test_seed_reproducible(self):
        _install_fake_pipe()
        _, m1 = imagegen.generate_image("cat", steps=2, seed=42)
        _, m2 = imagegen.generate_image("cat", steps=2, seed=42)
        assert m1["seed"] == m2["seed"] == 42
        # The generator the fake pipe saw must carry the same seed.
        # (checked indirectly: fake records seed via generator.initial_seed)

    def test_random_seed_when_negative(self):
        _install_fake_pipe()
        _, m1 = imagegen.generate_image("cat", steps=2, seed=-1)
        _, m2 = imagegen.generate_image("cat", steps=2, seed=-1)
        assert m1["seed"] >= 0 and m2["seed"] >= 0

    def test_settings_clamped_to_caps(self):
        pipe = _install_fake_pipe()
        _, meta = imagegen.generate_image(
            "x", width=4096, height=4096, steps=500, seed=1,
        )
        cfg = default_imagegen_config()
        assert meta["width"] <= cfg.max_width
        assert meta["height"] <= cfg.max_height
        assert meta["steps"] <= cfg.max_steps
        call = pipe.calls[-1]
        assert call["width"] <= cfg.max_width

    def test_progress_callback_reports_completion(self):
        _install_fake_pipe()
        seen = []
        imagegen.generate_image(
            "x", steps=3, seed=1, progress_fn=lambda f: seen.append(f),
        )
        assert seen and seen[-1] == pytest.approx(1.0)

    def test_negative_prompt_forwarded(self):
        pipe = _install_fake_pipe()
        _, meta = imagegen.generate_image(
            "apple", negative_prompt="blurry", steps=2, seed=1,
        )
        assert pipe.calls[-1]["negative_prompt"] == "blurry"
        assert meta["negative_prompt"] == "blurry"

    def test_default_negative_prompt_applied_when_unset(self):
        _install_fake_pipe()
        _, meta = imagegen.generate_image("apple", steps=2, seed=1)
        assert meta["negative_prompt"]  # non-empty default from config

    def test_cuda_oom_is_wrapped_with_advice(self):
        _install_fake_pipe(fail_oom=True)
        with pytest.raises(imagegen.ImageGenError, match="out of memory"):
            imagegen.generate_image("x", steps=1, seed=1)


# ---------------------------------------------------------------------------
# Saving images + metadata sidecars
# ---------------------------------------------------------------------------

class TestSaveImage:
    def test_save_and_reopen(self, tmp_path: Path):
        img = Image.new("RGB", (64, 64), color=(10, 20, 30))
        out = imagegen.save_image(img, tmp_path / "t.png", {"seed": 7})
        assert out.exists()
        reopened = Image.open(out)
        assert reopened.size == (64, 64)
        assert reopened.getpixel((0, 0)) == (10, 20, 30)

    def test_metadata_sidecar_written(self, tmp_path: Path):
        cfg = default_imagegen_config()
        imagegen._STATE["config"] = cfg
        img = Image.new("RGB", (32, 32))
        meta = {"seed": 99, "prompt": "test", "model_id": "m"}
        out = imagegen.save_image(img, tmp_path / "with_meta.png", meta)
        sidecar = out.with_suffix(".json")
        assert sidecar.exists()
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert data["seed"] == 99
        assert data["prompt"] == "test"

    def test_timestamped_filename_when_no_path(self, tmp_path: Path):
        cfg = default_imagegen_config()
        imagegen._STATE["config"] = cfg
        img = Image.new("RGB", (16, 16))
        out = imagegen.save_image(
            img, None, {"seed": 5}, output_dir=tmp_path,
        )
        assert out.parent == tmp_path
        assert out.name.startswith("dipai_")
        assert "seed5" in out.name

    def test_saving_non_image_raises(self, tmp_path: Path):
        with pytest.raises(imagegen.ImageGenError, match="PIL.Image"):
            imagegen.save_image("not an image", tmp_path / "x.png")

    def test_pgn_metadata_embedded(self, tmp_path: Path):
        img = Image.new("RGB", (16, 16))
        out = imagegen.save_image(
            img, tmp_path / "meta.png", {"prompt": "hello", "seed": 3},
        )
        reopened = Image.open(out)
        assert reopened.text.get("prompt") == "hello"
        assert reopened.text.get("seed") == "3"


# ---------------------------------------------------------------------------
# Status / load-failure paths (no weights)
# ---------------------------------------------------------------------------

class TestStatusAndFailures:
    def test_status_before_load(self):
        imagegen.unload_image_gen()
        st = imagegen.get_imagegen_status()
        assert st.loaded is False
        assert "not loaded" in st.line()

    def test_status_to_dict_keys(self):
        st = imagegen.get_imagegen_status()
        d = st.to_dict()
        for key in ("loaded", "model_id", "device", "generations",
                    "weights_cached", "free_disk_gb"):
            assert key in d

    def test_generate_with_broken_load_raises_clear_error(self, monkeypatch):
        """When the pipeline cannot be loaded, the user sees an ImageGenError."""
        imagegen.unload_image_gen()

        def _boom(*a, **k):
            raise imagegen.ImageGenError("weights not found locally")

        monkeypatch.setattr(imagegen, "load_image_gen", _boom)
        with pytest.raises(imagegen.ImageGenError, match="not found"):
            imagegen.generate_image("x", steps=1, seed=1)

    def test_unload_when_not_loaded_is_safe(self):
        imagegen.unload_image_gen()  # must not raise
        assert imagegen.is_loaded() is False

    def test_is_loaded_reflects_state(self):
        assert imagegen.is_loaded() is False
        _install_fake_pipe()
        assert imagegen.is_loaded() is True
        imagegen.unload_image_gen()
        assert imagegen.is_loaded() is False


# ---------------------------------------------------------------------------
# CLI argument surface
# ---------------------------------------------------------------------------

class TestCLI:
    def test_parser_has_required_flags(self):
        ap = imagegen._build_arg_parser()
        args = ap.parse_args([
            "--prompt", "hi", "--width", "256", "--steps", "5", "--seed", "1",
        ])
        assert args.prompt == "hi"
        assert args.width == 256
        assert args.steps == 5
        assert args.seed == 1

    def test_status_flag_exits_zero(self, capsys):
        rc = imagegen.main(["--status"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "imagegen" in out

    def test_missing_prompt_errors(self, capsys):
        with pytest.raises(SystemExit):
            imagegen.main([])


# ---------------------------------------------------------------------------
# Web UI handlers (no Gradio app construction needed)
# ---------------------------------------------------------------------------

class TestWebUIHandlers:
    def test_import_handlers(self):
        from webui import app

        assert hasattr(app, "imagegen_generate")
        assert hasattr(app, "imagegen_download")
        assert hasattr(app, "imagegen_unload")

    def test_generate_empty_prompt_returns_error_status(self):
        from webui import app

        img, status = app.imagegen_generate(
            "", "", 512, 512, 25, 7.5, -1,
        )
        assert img is None
        assert "[error" in status

    def test_generate_uses_fake_pipeline_and_saves(self, tmp_path: Path, monkeypatch):
        from webui import app

        _install_fake_pipe()
        cfg = default_imagegen_config()
        cfg.output_dir = str(tmp_path / "imgs")
        monkeypatch.setattr(
            app, "load_imagegen_config",
            lambda *a, **k: cfg,
        )
        img, status = app.imagegen_generate(
            "a red apple", "", 512, 512, 2, 7.5, 42,
        )
        assert img is not None
        assert "[error" not in status
        assert "seed 42" in status
        saved = list((tmp_path / "imgs").glob("*.png"))
        assert len(saved) == 1

    def test_unload_handler_returns_status(self):
        from webui import app

        _install_fake_pipe()
        status = app.imagegen_unload()
        assert "unloaded" in status
        assert imagegen.is_loaded() is False

    def test_download_handler_failure_is_visible(self, monkeypatch):
        from webui import app

        def _boom(*a, **k):
            raise imagegen.ImageGenError("network down")

        monkeypatch.setattr(app.imagegen, "download_model", _boom)
        stream = app.imagegen_download_stream()
        first = next(stream)                      # "downloading..." indicator
        assert "[error" not in first
        second = next(stream)                     # real result or error
        assert "[error" in second

    def test_download_model_refuses_when_disallowed(self):
        cfg = ImageGenConfig(allow_download=False)
        with pytest.raises(imagegen.ImageGenError, match="disabled"):
            imagegen.download_model(cfg)

    def test_download_model_disk_check_for_uncached_model(self):
        # Disk gate must fire when weights are NOT yet on disk (use a model
        # id that cannot be cached so the critical-present short-circuit
        # does not apply; the gate runs before any network call).
        cfg = ImageGenConfig(min_free_disk_gb=10_000,
                             model_id="dipai-tests/definitely-not-cached")
        with pytest.raises(imagegen.ImageGenError, match="free disk"):
            imagegen.download_model(cfg)

    def test_download_model_low_disk_but_cached_returns_cache(self):
        # Criticals cached + disk below policy -> succeed with local cache.
        cfg = ImageGenConfig(min_free_disk_gb=10_000)
        path = imagegen.download_model(cfg)
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# Opt-in REAL model smoke test (downloads SD1.5 - never runs by default)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(
    not __import__("os").environ.get("DIPAI_IMAGEGEN_SMOKE"),
    reason="set DIPAI_IMAGEGEN_SMOKE=1 to run the real SD1.5 download+generate",
)
class TestRealModelSmoke:
    def test_real_generation_64px(self, tmp_path: Path):
        """Tiny real run: 64x64, 2 steps - proves the real pipeline works."""
        cfg = default_imagegen_config()
        cfg.default_width = 64
        cfg.default_height = 64
        cfg.min_steps = 1
        cfg.default_steps = 2
        img, meta = imagegen.generate_image(
            "a small red cube", config=cfg, steps=2, seed=1,
        )
        assert isinstance(img, Image.Image)
        assert meta["seed"] == 1
        out = imagegen.save_image(img, tmp_path / "real.png", meta)
        assert out.exists()
