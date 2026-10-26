# DIPAI Repository Audit & Hardware Report

Phase 1 deliverable. All numbers below were measured on this machine on
2026-10-22; nothing is estimated unless explicitly labelled as an estimate.

## 1. Repository audit

### What the project is

DIPAI (repo root: `C:\Users\Lenovo\Downloads\Dipai\DIPAI`, git branch `main`,
remote `https://github.com/dipeshdarks/Dipai`, last commit `ec47289`) is a
local-first Python 3.10 application for training and running language models.
It is pure Python + PyTorch; there is no JavaScript build step.

### Languages, entry points, dependencies

| Area | Finding |
|---|---|
| Language | Python (3.10.1), no compiled extensions of its own |
| Text chat CLI | `python -m dipai.chat` (`dipai/chat.py`) |
| Web UI | `python -m webui.app` or `python webui/app.py` (`webui/app.py`, Gradio 5.50.0) |
| Training | `python -m dipai.train` (scratch, Mode A/B), `python -m dipai.finetune` (Modes B/C/D) |
| Evaluation | `python -m dipai.evaluate` (perplexity + generation reports) |
| Data pipeline | `python -m dipai.data.collect` → `clean` → `prepare` (6-source licensed registry) |
| Tests | `pytest` (200 passed / 1 skipped as of `ec47289`) |
| Core deps | torch 2.14.1+cu126, transformers 4.57.6, peft 0.21.2, accelerate 1.15.0, bitsandbytes 0.50.2, gradio 5.50.0, Pillow 11.3.0 |
| Missing dep | **`diffusers` not installed** (required for text-to-image) |
| Not required | `controlnet_aux`, `invisible_watermark` — absent, not needed for baseline SD1.5 |

### Model architecture and existing text capabilities

* **DipaiGPT** (`dipai/model.py`) — a GPT-style decoder trained from scratch
  in-repo; checkpoints under `checkpoints/` (7 dirs, 4.9 GB total, never touched
  by this upgrade).
* **HF backends** (`dipai/inference.py`) — three loaders already exist and are
  wired to the model registry (`dipai/model_registry.py`):
  `from_checkpoint` (scratch), `from_adapter` (PEFT/LoRA, 4-bit on CUDA),
  `from_model` (full-parameter HF dir, 4-bit on CUDA), dispatched by
  `from_named`.
* Generation (`dipai/generate.py`) already provides temperature, top-k, top-p,
  repetition penalty, max-new-tokens, streaming callbacks, and a stop event;
  the HF path uses `TextIteratorStreamer`, chat templates, and an EOS stop.
* The web UI streams responses, owns visible history, has a settings
  accordion (temp/top-k/top-p/max-new/rep-pen), STOP/NEW buttons, and an
  on-demand **image-understanding** tab (SmolVLM, separate from text).

### What exists vs. what the new upgrade requires

| Requirement | Status in repo today |
|---|---|
| Chat UI, streaming, settings, history | Present |
| Configurable system prompt | **Missing** (no system role anywhere) |
| Context-limit enforcement on HF path | **Missing** (scratch path slices to `block_size`; HF path never truncates) |
| Stop-on-echo / stop-strings | **Missing** (EOS only; repetition penalty is the only loop guard) |
| Registered pretrained instruct model | **Missing** (registry has only `scratch`; Qwen is cached on disk but unregistered and unloadable by name — `from_model` requires a local dir) |
| OOM/insufficient-memory error mapping | **Missing** (raw CUDA exceptions surface in UI) |
| Text-to-image generation | **Absent entirely** (`diffusers` not installed, no module, no UI tab) |
| Image generation UI | **Absent** (UI has Chat + Image understanding only) |
| Live GPU/RAM meter in UI | **Missing** (header shows values captured once at load) |
| Model-download status in UI | **Missing** |

### Preserved by rule

`checkpoints/` (4.9 GB), `data/` (incl. untracked `data/clean/`), all configs,
`.venv/`, HF cache, and every existing test/feature stay untouched and
unrenamed. No architecture replacement, no duplicate implementations: the
image-generation path is a new module (`dipai/imagegen.py`) + a new UI tab that
reuse existing patterns (registry, config loader, logging, status line).

## 2. Hardware report (measured)

| Resource | Measured value | Consequence |
|---|---|---|
| GPU | NVIDIA GeForce RTX 2050, 4095.6 MB total, **3302.2 MB free** | SD1.5 fp16 @ 512×512 fits (est. peak 2.5–3.0 GB with attention slicing); SDXL does not |
| CUDA | 12.6 via torch 2.14.1+cu126, `cuda.is_available()==True`, tiny GEMM OK | No hard-coded CUDA settings; device resolved at runtime (`dipai/hardware.py`) |
| System RAM | 5.86 GB total, **0.53 GB available** | Tightest constraint: pipeline load will use the page file; load components with fp16 + `low_cpu_mem_usage`, CPU offload avoided |
| Disk (C:, only volume) | **2.52 GB free** / 474.7 GB | SD1.5 fp16 ≈ 2.15 GB does not fit safely *before* reclamation |
| Reclaimable cache | pip cache **2827 MB** (safe to purge, re-downloadable) | Purge raises free space to ≈ 5.3 GB |
| Temp dir | 8662 MB (incl. 3.3 GB app folder `eepplplt`) | **Not touched** — not identified as ours |
| HF cache | 2.83 GB: Qwen2.5-0.5B-Instruct **1906.6 MB** (already downloaded), SmolVLM-256M 987.8 MB | Qwen text model needs **zero download** |
| Network | `huggingface.co` reachable (API responded), GitHub push works | Downloads possible; still pre-flight-checked every time |

## 3. Model selection evidence (text + image)

### Text — `Qwen/Qwen2.5-0.5B-Instruct`

* Already in the local HF cache (1.9 GB) → no download.
* 0.5B params ≈ 1.0 GB fp16 on GPU → fits beside a 2.1 GB diffusion pipeline
  only with care; otherwise sequential use (existing pattern: lazy loads).
* Apache-2.0 licence; instruct-tuned → satisfies conversation, instruction
  following, summarization, coding, reasoning, creative writing.
* Needs a code change to be loadable *by name* (hub-id support in
  `from_model` + a registry entry).

### Image — `stable-diffusion-v1-5/stable-diffusion-v1-5`

Chosen after checking licence, memory, dependencies, and hardware, as required:

| Option | Disk (fp16) | VRAM @512² | Licence | Verdict |
|---|---|---|---|---|
| **SD 1.5** | ≈2.15 GB | 2.5–3.0 GB (fits 3.3 GB free w/ slicing) | CreativeML OpenRAIL-M | **Selected** |
| SDXL base/Turbo | ≈5–7 GB | >7 GB | OpenRAIL++ / Community | Rejected: disk and VRAM both insufficient |
| Segmind TinySD | ≈0.7 GB | ≈1.5 GB | OpenRAIL-M | Fallback if SD1.5 fails preflight |
| FLUX.1-schnell | ≈24 GB | >12 GB | Apache-2.0 | Rejected: far beyond hardware |

Quality honesty (per mission): realism depends on model + prompt + settings +
resources. Prompt wording alone guarantees nothing; claims in the UI/docs will
be limited to what the generated images actually show.

## 4. Risk register

1. **RAM 0.53 GB available** — loading SD1.5 weights needs ~1.7–2.1 GB peak;
   Windows will page. Mitigation: fp16 load, `low_cpu_mem_usage`, load directly
   toward GPU, sequential text/image use, OOM handled with a clear message.
2. **Disk 2.52 GB** — must purge pip cache first; `--download` runs a disk
   preflight and refuses to start if free space < estimate + 0.5 GB margin.
3. **Safety checker VRAM** — enabling SD1.5's NSFW checker would add ~0.5–0.9 GB
   and risk OOM on 4 GB. Decision: disabled with `requires_safety_checker=False`,
   documented honestly rather than hidden.
4. **Torch 2.14 + diffusers compatibility** — verified by import + a real
   generation smoke test before anything is reported as working.
