# DIPAI Multimodal Upgrade — Implementation Plan

Phase 1 deliverable (2). Grounded in `docs/UPGRADE_AUDIT.md`; every step names
the real files it touches. No existing feature is replaced; nothing under
`checkpoints/` or `data/` is modified or deleted at any point.

## Ground rules for every phase

* Inspect before editing; preserve all existing code/config/data/weights.
* No fabricated capabilities: a feature may only be called working after its
  real test or real run has been executed and reported.
* Downloads: official Hugging Face org only, `trust_remote_code=False`,
  disk/RAM preflight printed first, resumable, never from untrusted sources.
* Incremental changes; the app must stay runnable after each phase; every
  phase ends with its focused tests actually run and reported.

## Phase 2 — Text generation upgrade

Target: `dipai/model_registry.py`, `dipai/inference.py`, `dipai/generate.py`,
`dipai/chat.py`, `tests/`.

1. **Registry**: add built-in `qwen` entry (`backend="hf-full"`,
   path `Qwen/Qwen2.5-0.5B-Instruct`, hub id). `exists()` learns to recognise
   a hub id whose snapshot is already in the local HF cache.
2. **Hub-id loading**: `from_model`/`from_adapter` accept hub ids as well as
   local dirs (already-cached snapshots load offline; an uncached id fails
   with an explicit "not downloaded — run `huggingface-cli download …`"
   style error).
3. **System prompt**: optional configurable system prompt on
   `DIPAIEngine.generate()`; injected as a `system` message on the HF path and
   prepended on the scratch path; exposed as `--system-prompt` (chat CLI) and
   an advanced-settings textbox (UI).
4. **Context limits**: HF path truncates the rendered chat to
   `max_position_embeddings − max_new_tokens` (log a warning when it fires);
   scratch path already slices to `block_size`.
5. **Stop strings / repetition**: generation stops (and output is trimmed) at
   user-role echo markers (`\nUser:`, `### Human`, etc.) in addition to EOS;
   existing repetition penalty + max-new cap remain.
6. **Errors**: map CUDA "out of memory" to a clear `EngineError` with concrete
   advice (smaller context/quantize/CPU); fix `ready` for hf backends.
7. **Quantization choice**: `quantize="auto"|"4bit"|"none"` on the HF loaders
   (auto keeps today's behaviour); expose `--quantize` on chat/webui.
8. **Fixed test set**: run `dipai.evaluate` on the *same* fixed dataset slice
   for `qwen` before/after (no fine-tuning is claimed by this upgrade) and
   record the real numbers in the final report.

Tests: system-prompt injection, truncation, stop trimming, OOM wrapping,
registry hub-id resolution, mocked hub loading — all offline.

## Phase 3 — Image generation (real, pretrained)

Target: `dipai/imagegen.py` (new), `configs/imagegen.yaml` (new),
`requirements.txt`, tests.

1. **Dependencies**: `pip install diffusers` (official PyPI; pinned range in
   requirements.txt); verify import against installed transformers/torch.
2. **Disk preflight**: purge pip cache (2827 MB, re-downloadable) → verify
   ≥ 4 GB free → download `stable-diffusion-v1-5/stable-diffusion-v1-5`
   **fp16 variant only** (≈2.15 GB: `unet`, `text_encoder`, `vae` fp16 +
   configs/scheduler/tokenizer) into the HF cache with resume support
   (`snapshot_download`, allow-list of needed files only).
3. **Pipeline module** `dipai/imagegen.py`:
   * `load_pipeline()` — fp16, `low_cpu_mem_usage`, attention slicing + VAE
     slicing, safety checker disabled (documented), device from
     `dipai.hardware.get_device`, OOM→clear-error mapping.
   * `generate(prompt, negative_prompt, width, height, steps, cfg, seed)`
     → PIL image + full metadata (prompt, negative, seed, size, steps, cfg,
     model id, scheduler, device, elapsed seconds).
   * Save to `outputs/images/<timestamp>_<seed>.png` + `.json` sidecar;
     reproducible seed; validation of empty prompts / bad sizes / step bounds.
   * `--download` CLI entry with printed disk+RAM+VRAM estimate that refuses
     to start when free space < estimate + margin (`--force` to override).
4. **Scope honesty**: text-to-image only (no img2img/inpaint in v1);
   realism statement = model+prompt+settings dependent, no guarantees.

Tests: invalid prompt/params, missing-model error text, save-and-reopen with
metadata round-trip, seed reproducibility and a **real** 8-step 256²
generation when the model is present (skipped otherwise, reported truthfully).

## Phase 4 — Unified interface

Target: `webui/app.py` (+ tests). Reuse the existing Gradio app/theme.

1. New **Image generation** tab: prompt + negative prompt, size dropdown
   (512², 512×768, 768×512), steps, guidance, seed (−1 = random), Generate
   button, progress/status states (loading → ready → generating →
   completed/failed), image preview, auto-save with visible save path +
   metadata panel, download control.
2. **Model selection**: dropdown of registry entries with availability
   (exists/cached) shown; switching reloads the text engine without touching
   the image pipeline.
3. **Download status**: image-model state shown in the tab (present /
   missing + a Download button that runs the Phase 3 downloader in a
   background thread with live status text).
4. **Live resource line**: GPU free/total VRAM + RAM available, refreshed per
   interaction (`dipai.hardware` + psutil), replacing the static-only header
   values; VRAM gate before image generation (refuse with advice when free
   VRAM < estimate — no simultaneous text+image workloads on 4 GB).
5. **Unload image model** button (frees VRAM for text use).
6. Keep dark theme/ accessibility of the existing app; no rebuild.

Tests: handler-level tests with a mocked pipeline (empty prompt, error
states, status transitions), registry dropdown population, VRAM-gate logic.

## Phase 5 — Training & data

* Text: existing licensed pipeline (`dipai/data/*`, `configs/data.yaml`,
  Modes A–D) is reused unchanged; no new training runs are claimed.
* Image: **no diffusion training on this GPU** (4 GB — out of scope by the
  mission's own rule). New doc `docs/IMAGE_TRAINING.md` documents the
  future LoRA route (dataset rights, preprocessing, VRAM, limits) and is
  clearly marked PLANNED — not implemented.

## Phase 6 — Hardware & deployment

* All device choices runtime-detected (`dipai/hardware.py`); no hard-coded
  CUDA indices or dtype assumptions.
* Preflight tables (disk/VRAM/RAM estimates) printed before every large
  download or load; sequential text/image use enforced by the VRAM gate.
* Existing checkpoints never deleted; cloud route documented in the existing
  `docs/CLOUD_GPU_TRAINING.md` + a note in the new docs for heavier jobs.

## Phase 7 — Testing & verification

Full suite (existing 200 + new Phase 2/3/4 tests) run and reported verbatim.
Real end-to-end evidence: one real text generation from `qwen` and one real
SD1.5 image written to `outputs/images/` with its metadata sidecar. UI
integration checks run through the Gradio handlers (module-level, as the
existing tests do). Nothing is marked passed that was not executed.

## Deliverables map

| # | Deliverable | Where |
|---|---|---|
| 1 | Audit + hardware report | `docs/UPGRADE_AUDIT.md` |
| 2 | Implementation plan | `docs/UPGRADE_PLAN.md` (this file) |
| 3 | Updated source + config | Phases 2–4 file targets |
| 4 | Windows setup instructions | `docs/WINDOWS_SETUP.md` |
| 5–6 | Text + real text-to-image workflows | README + docs |
| 7 | Optional fine-tuning scripts | existing `dipai/finetune.py` (unchanged, referenced) |
| 8 | Automated tests | `tests/test_textgen_upgrade.py`, `tests/test_imagegen.py`, `tests/test_imageui.py` |
| 9 | README with copy-paste PowerShell | README update |
| 10 | Final report (implemented/tested/untested/blocked) | `docs/UPGRADE_REPORT.md` |
