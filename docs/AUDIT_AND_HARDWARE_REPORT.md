# DIPAI — Repository Audit & Hardware Report

Date: 2026-10-26
Auditor: automated codebase inspection (no files were modified during audit)

---

## 1. Hardware (measured at audit time)

| Component | Measured value |
|---|---|
| GPU | NVIDIA GeForce RTX 2050, compute capability 8.6 |
| VRAM | 4.00 GB total, 3.22 GB free (desktop compositor uses ~340 MB) |
| CUDA driver | 581.80 / driver CUDA 13.0 |
| PyTorch CUDA | torch 2.14.1+cu126, `torch.cuda.is_available() = True` |
| CPU | 8 logical threads |
| System RAM | 5.86 GB total, **0.81 GB available at probe time** (other apps resident) |
| Disk (C:) | ~31 GB free of 476 GB |
| Python | 3.10.1 (venv at `DIPAI/.venv`) |

Warnings emitted by `python -m dipai.hardware`:
- Available RAM is low (0.81 GB) — close other applications before loading large models.
- Total RAM (5.9 GB) is below the project's 8 GB design target.

Implication for image generation: Stable Diffusion 1.5 at 512×512 with fp16 + attention
slicing peaks around 3.0–3.6 GB VRAM — fits the 4 GB GPU. SDXL (≥6.5 GB) does not fit.
Weight loading needs several GB of free RAM; the machine must not run memory-heavy
apps concurrently.

## 2. Repository layout (DIPAI/)

Python 3.10 project, pure PyTorch + Hugging Face stack. No JavaScript frontend.

| Area | Location | State |
|---|---|---|
| Scratch LM (Mode A) | `dipai/model.py`, `train.py`, `tokenizer.py`, `dataset.py` | Implemented; checkpoints exist |
| HF LoRA/QLoRA (Mode B/C/D) | `dipai/finetune.py`, `configs/finetune.yaml` | Implemented |
| Inference engine | `dipai/inference.py` (scratch + hf + hf-full backends) | Implemented; streaming, sampling controls |
| Generation sampling | `dipai/generate.py` (temp/top-k/top-p/rep-pen) | Implemented |
| Terminal chat | `dipai/chat.py` | Implemented |
| Web UI | `webui/app.py` (Gradio 5.50, Chat + Image understanding tabs) | Implemented |
| Image **understanding** | `dipai/vision.py` (SmolVLM-256M, lazy load) | Implemented |
| Image **generation** | — | **Missing (this upgrade)** |
| Model registry | `dipai/model_registry.py` | Implemented |
| Hardware probe | `dipai/hardware.py` | Implemented |
| Data pipeline | `dipai/data/*` (registry, collect, clean, prepare) | Implemented |
| Tests | `tests/` (24 files, README claims 200 passed) | Implemented |
| Configs | `configs/*.yaml` | Implemented (no imagegen config yet) |

## 3. Existing checkpoints (preserved — none deleted)

| Directory | Type | Notes |
|---|---|---|
| `checkpoints/scratch` | Scratch ~21 M GPT | ckpt_00005000–6000 |
| `checkpoints/50m` | Scratch ~50 M GPT | ckpt_00005500–6000 |
| `checkpoints/sft50m`, `sft50m_v2` | Full-param SFT of 50 M | ckpt_00000200–1055 |
| `checkpoints/smoke`, `smoke4m` | Tiny smoke models | test only |
| `checkpoints/finetune/adapter` | **Qwen2.5-0.5B-Instruct LoRA adapter** | instruction-tuned; best text model on disk |
| `checkpoints/finetune/checkpoint-200/250` | Trainer intermediate | resumable |

## 4. Installed stack (venv)

torch 2.14.1+cu126 · transformers 4.57.6 · accelerate 1.15.0 · peft 0.21.2 ·
bitsandbytes 0.50.2 · gradio 5.50.0 · Pillow 11.3.0 · sentencepiece · tokenizers ·
datasets · safetensors · pytest 9.1.1 · numpy 2.2.6

**Not installed:** `diffusers`, `huggingface_hub` (pulled in transitively as a
transformers dep but the image-generation pipeline package is absent).

## 5. Gaps vs. the multimodal upgrade mission

| Requirement | Status |
|---|---|
| High-quality text generation | Partially met: Qwen2.5-0.5B-Instruct LoRA adapter exists; scratch models are continuations only. Needs system-prompt plumbing + defaulting to the instruct model. |
| Photorealistic image generation | **Absent** — no diffusion pipeline, no image outputs, no metadata store |
| Text-to-image UI mode | **Absent** — web UI has Chat + Image understanding only |
| Unified polished interface | Exists for text/vision; needs Image Generation tab |
| Realism controls (negative prompt, seed, size, steps, guidance) | **Absent** |
| Explicit model download with progress/disk checks | **Absent** for image models |
| Tests for image generation | **Absent** |

## 6. Constraints that shaped the plan

1. **4 GB VRAM** → Stable Diffusion 1.5 (fp16, attention slicing, 512×512) is the
   realistic default. SDXL is excluded.
2. **~6 GB RAM, often <1 GB free** → pipelines must use `low_cpu_mem_usage`,
   lazy load, and refuse to start with a clear error when RAM is insufficient.
3. **~31 GB free disk** → SD 1.5 weights ≈ 4–5 GB cache; download is feasible but
   must be explicit and size-checked first.
4. **Windows + torch cu126** → no hard-coded CUDA capability strings; runtime probe only.
5. **Preserve everything** → no checkpoint/data deletion; image module is additive
   (`dipai/imagegen.py`, new tab, new config).

## 7. Plan summary (executed in this upgrade)

1. Add `ImageGenConfig` + `configs/imagegen.yaml` (model id, size caps, steps, guidance,
   negative prompt defaults, output dir, download behaviour).
2. Install `diffusers` (+ ensure `huggingface_hub`); pin in `requirements.txt`.
3. Implement `dipai/imagegen.py`: lazy SD 1.5 pipeline, fp16 + attention/vae slicing,
   seed control, negative prompts, PNG metadata sidecar, progress callbacks, VRAM/RAM
   preflight, explicit download with disk-space check, CLI (`python -m dipai.imagegen`).
4. Add system-prompt support to `ConversationSession` / `DIPAIEngine` (both backends)
   and expose it in the web UI advanced settings; prefer the instruct adapter as the
   web UI default when present.
5. Add an **Image Generation** tab to `webui/app.py` (prompt, Generate, preview,
   save, advanced accordion, download status, progress/errors).
6. Tests: mocked pipeline unit tests (config, seed, metadata, save/reopen, invalid
   prompts, missing model, CPU path, VRAM estimate) + webui handler tests; run the
   full suite and report real numbers.
7. Update README + final report (implemented / tested / untested / blocked).
