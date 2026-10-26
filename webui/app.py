"""DIPAI local web chat UI (Gradio).

Run:
    python webui/app.py --ckpt checkpoints/scratch
    # or:  python -m webui.app (from the project root)

Visual direction: dark interface, black/charcoal surfaces, white typography,
electric red (#FF2D2D) accent - minimal, professional, desktop-responsive.

This UI drives the REAL local model through dipai.inference.DIPAIEngine.
There are no simulated responses anywhere in this file: every assistant
message comes from model generation (or is an explicit error state).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as a script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from dipai import imagegen, vision
from dipai.config import load_imagegen_config, load_vision_config
from dipai.generate import GenerationError, GenerationSettings
from dipai.imagegen import ImageGenError
from dipai.inference import ConversationSession, DIPAIEngine, EngineError
from dipai.model_registry import default_registry
from dipai.utils import setup_logging
from dipai.vision import VisionError

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

ELECTRIC_RED = "#FF2D2D"
CHARCOAL = "#141414"
BLACK = "#0A0A0A"
SURFACE = "#1C1C1C"

CUSTOM_CSS = f"""
:root {{
    --dipai-red: {ELECTRIC_RED};
    --dipai-black: {BLACK};
    --dipai-charcoal: {CHARCOAL};
    --dipai-surface: {SURFACE};
}}
.gradio-container {{
    background: {BLACK} !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
}}
#dipai-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    background: {CHARCOAL};
    border-bottom: 2px solid {ELECTRIC_RED};
    border-radius: 8px 8px 0 0 !important;
}}
#dipai-logo {{
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 4px;
    color: #FFFFFF;
    font-family: 'Segoe UI', system-ui, sans-serif;
}}
#dipai-logo span {{ color: {ELECTRIC_RED}; }}
#dipai-status {{
    font-size: 12px;
    color: #9A9A9A;
    font-family: 'Consolas', monospace;
    text-align: right;
}}
#dipai-status b {{ color: #FFFFFF; }}
#dipai-chatbot {{
    background: {BLACK} !important;
    min-height: 480px;
}}
#dipai-chatbot .message {{
    background: {SURFACE} !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
}}
#dipai-chatbot .bot, #dipai-chatbot [data-testid="bot"] {{
    background: {CHARCOAL} !important;
    border-left: 3px solid {ELECTRIC_RED} !important;
}}
#dipai-chatbot .user, #dipai-chatbot [data-testid="user"] {{
    background: #232323 !important;
    border-right: 3px solid #555 !important;
}}
#dipai-send {{ background: {ELECTRIC_RED} !important; border: none !important; color: white !important; font-weight: 700 !important; }}
#dipai-send:hover {{ background: #E02020 !important; }}
#dipai-stop {{ background: #2A2A2A !important; color: #FF6B6B !important; border: 1px solid #444 !important; }}
#dipai-new {{ background: transparent !important; color: #FFFFFF !important; border: 1px solid {ELECTRIC_RED} !important; }}
#dipai-new:hover {{ background: rgba(255,45,45,0.12) !important; }}
footer {{ display: none !important; }}
"""


def build_theme():
    try:
        return gr.themes.Base(
            primary_hue="red",
            secondary_hue="gray",
            neutral_hue="gray",
        ).set(
            body_background_fill=BLACK,
            body_background_fill_dark=BLACK,
            block_background_fill=SURFACE,
            block_background_fill_dark=SURFACE,
            block_title_text_color="#FFFFFF",
            block_label_text_color="#CFCFCF",
            input_background_fill="#101010",
            input_background_fill_dark="#101010",
            input_text_color="#FFFFFF",
            input_text_color_dark="#FFFFFF",
            button_primary_background_fill=ELECTRIC_RED,
            button_primary_background_fill_dark=ELECTRIC_RED,
            button_primary_text_color="#FFFFFF",
            color_accent=ELECTRIC_RED,
            color_accent_soft="#3A1010",
            background_fill_primary=BLACK,
            background_fill_primary_dark=BLACK,
            text_color="#FFFFFF",
            text_color_dark="#FFFFFF",
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Vision (image understanding) handlers - module level so tests can drive
# them without building the Gradio app. The vision model is loaded ON DEMAND
# on the first question; the text-only chat tab never triggers it.
# ---------------------------------------------------------------------------

_VISION_CONFIG_PATH = "configs/vision.yaml"


def _vision_status_line() -> str:
    return f"`{vision.get_vision_status().line()}`"


def vision_ask(image_path: Optional[str], question: str):
    """Validate input, run real VLM inference, and return (answer, status).

    Errors are returned as visible "[error: ...]" text - never fabricated
    answers, never an unhandled crash that would take down the text chat.
    """
    try:
        if not image_path:
            return "[error: upload an image first]", _vision_status_line()
        if not question or not str(question).strip():
            return ("[error: enter a question about the image "
                    "(e.g. 'What is in this picture?')]"), _vision_status_line()
        cfg = load_vision_config(_VISION_CONFIG_PATH, required=False)
        vision.load_vision_model(cfg)          # no-op when already loaded
        answer = vision.answer_about_image(image_path, str(question).strip())
        return answer, _vision_status_line()
    except VisionError as exc:
        return f"[error: {exc}]", _vision_status_line()
    except Exception as exc:  # noqa: BLE001 - surfaced in UI, app stays alive
        return f"[error: vision request failed: {exc}]", _vision_status_line()


def vision_ask_stream(image_path: Optional[str], question: str):
    """Generator wrapper: shows a loading indicator before the answer."""
    yield "", _vision_status_line() + " | processing image..."
    answer, status = vision_ask(image_path, question)
    yield answer, status


def vision_clear():
    """Clear image, question, answer; keep the model status visible."""
    return None, "", "", _vision_status_line()


# ---------------------------------------------------------------------------
# Image GENERATION handlers - module level so tests can drive them without
# building the Gradio app. The diffusion pipeline loads ON DEMAND on the
# first request; the text chat never pays the memory cost of SD 1.5.
# ---------------------------------------------------------------------------

_IMAGEGEN_CONFIG_PATH = "configs/imagegen.yaml"


def _imagegen_status_line() -> str:
    return f"`{imagegen.get_imagegen_status().line()}`"


def imagegen_generate(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    progress: Optional[Any] = None,
):
    """Run real text-to-image generation; return (image, status).

    Errors are returned as a visible "[error: ...]" status - never a
    fabricated image, never an unhandled crash that would take down the UI.
    """
    try:
        if not prompt or not str(prompt).strip():
            return None, _imagegen_status_line() + " | [error: enter a prompt]"
        cfg = load_imagegen_config(_IMAGEGEN_CONFIG_PATH, required=False)

        def _frac(f: float) -> None:
            if progress is not None:
                try:
                    progress(f)
                except Exception:  # noqa: BLE001
                    pass

        image, meta = imagegen.generate_image(
            str(prompt).strip(),
            config=cfg,
            negative_prompt=str(negative_prompt).strip() if negative_prompt else None,
            width=int(width),
            height=int(height),
            steps=int(steps),
            guidance_scale=float(guidance_scale),
            seed=int(seed),
            progress_fn=_frac if progress is not None else None,
        )
        # Save automatically; metadata travels with the .json sidecar on disk.
        path = imagegen.save_image(image, None, meta, output_dir=cfg.output_dir)
        meta["saved_to"] = str(path)
        status = _imagegen_status_line() + f" | saved: {path.name} | " \
                                          f"seed {meta['seed']} | " \
                                          f"{meta.get('generation_seconds', '?')}s"
        return image, status
    except ImageGenError as exc:
        return None, _imagegen_status_line() + f" | [error: {exc}]"
    except Exception as exc:  # noqa: BLE001 - surfaced in UI, app stays alive
        return None, _imagegen_status_line() + f" | [error: image request failed: {exc}]"


def imagegen_download():
    """Explicitly download/verify model weights (status line returned).

    Downloads only fetch weights to disk - the heavy pipeline load still
    happens lazily on the first Generate, so this stays usable on
    memory-constrained machines.
    """
    try:
        cfg = load_imagegen_config(_IMAGEGEN_CONFIG_PATH, required=False)
        path = imagegen.download_model(cfg)
        return _imagegen_status_line() + f" | weights cached: {Path(path).name}"
    except ImageGenError as exc:
        return _imagegen_status_line() + f" | [error: {exc}]"
    except Exception as exc:  # noqa: BLE001
        return _imagegen_status_line() + f" | [error: download failed: {exc}]"


def imagegen_download_stream():
    """Generator wrapper: shows a loading indicator around the download."""
    yield _imagegen_status_line() + " | downloading model weights..."
    yield imagegen_download()


def imagegen_unload():
    """Free image-model VRAM back to the system."""
    try:
        imagegen.unload_image_gen()
    except Exception:  # noqa: BLE001
        pass
    return _imagegen_status_line() + " | unloaded (VRAM freed)"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app(engine: DIPAIEngine) -> gr.Blocks:
    """Build the full chat application bound to `engine`."""
    status = engine.status()

    with gr.Blocks(title="DIPAI", theme=build_theme(), css=CUSTOM_CSS) as demo:
        # Per-session stop event lives here (one conversation at a time).
        stop_state = gr.State(lambda: threading.Event())
        busy_state = gr.State(False)

        gr.HTML(
            f"""
            <div id="dipai-header">
              <div id="dipai-logo">DIP<span>AI</span></div>
              <div id="dipai-status">
                <b>{status.model_params / 1e6:.2f}M</b> params
                &nbsp;|&nbsp; step <b>{status.trained_to_step}</b>
                &nbsp;|&nbsp; <b>{status.device.upper()}</b>
                {'(' + status.gpu_name + ')' if status.gpu_name else ''}
                &nbsp;|&nbsp; ctx <b>{status.block_size}</b>
                <br>local model &mdash; continues text, not a trained assistant
              </div>
            </div>
            """
        )

        with gr.Tabs():
            # ---- text chat (works without ever loading the vision model) ---
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(
                    type="messages",
                    elem_id="dipai-chatbot",
                    height=520,
                    placeholder="<div style='color:#666;padding:20px'>Start a conversation below.</div>",
                    show_label=False,
                )

                with gr.Row():
                    prompt_box = gr.Textbox(
                        placeholder="Type a prompt... (Shift+Enter for new line)",
                        lines=3,
                        max_lines=8,
                        show_label=False,
                        container=False,
                        scale=8,
                        elem_id="dipai-prompt",
                    )
                    with gr.Column(scale=1, min_width=110):
                        send_btn = gr.Button("SEND", variant="primary", elem_id="dipai-send")
                        stop_btn = gr.Button("STOP", elem_id="dipai-stop")

                with gr.Row():
                    new_btn = gr.Button("＋ NEW CONVERSATION", elem_id="dipai-new", scale=1)
                    status_box = gr.Markdown(
                        f"`{status.line()}`", elem_id="dipai-statusline"
                    )

            # ---- image understanding (separate pretrained VLM) -------------
            with gr.Tab("Image understanding"):
                gr.Markdown(
                    "Upload a **JPEG / PNG / WebP** image and ask a question. "
                    "Answers come from a separate local vision-language model "
                    "(loads on the first question; the text model is untouched)."
                )
                with gr.Row():
                    image_in = gr.Image(
                        type="filepath",
                        sources=["upload"],
                        label="Upload image",
                        height=340,
                        elem_id="dipai-image",
                    )
                    with gr.Column(scale=1):
                        question_box = gr.Textbox(
                            placeholder="e.g. What is in this picture?",
                            label="Question",
                            lines=4,
                            max_lines=8,
                            elem_id="dipai-question",
                        )
                        with gr.Row():
                            ask_btn = gr.Button(
                                "ASK", variant="primary", elem_id="dipai-ask"
                            )
                            clear_img_btn = gr.Button(
                                "CLEAR IMAGE", elem_id="dipai-clear-image"
                            )
                        vision_status_box = gr.Markdown(
                            _vision_status_line(), elem_id="dipai-vision-status"
                        )
                answer_box = gr.Markdown("", elem_id="dipai-image-answer")

            # ---- image GENERATION (separate diffusion pipeline) -----------
            with gr.Tab("Image generation"):
                gr.Markdown(
                    "**Text-to-image** with a local Stable Diffusion pipeline "
                    "(separate from the text model; loads on first Generate). "
                    "Quality depends on the model, prompt, and settings — "
                    "prompt wording alone does not guarantee realism."
                )
                gen_prompt_box = gr.Textbox(
                    placeholder="e.g. Photorealistic product photo of a glass of "
                                "iced milk tea, soft window light, 50mm lens...",
                    label="Prompt",
                    lines=3,
                    max_lines=8,
                    elem_id="dipai-gen-prompt",
                )
                with gr.Row():
                    gen_btn = gr.Button(
                        "GENERATE", variant="primary", elem_id="dipai-generate"
                    )
                    dl_btn = gr.Button("DOWNLOAD MODEL", elem_id="dipai-dl-model")
                    unload_btn = gr.Button("UNLOAD MODEL", elem_id="dipai-unload-model")

                gen_status_box = gr.Markdown(
                    _imagegen_status_line(), elem_id="dipai-gen-status"
                )
                gen_image_out = gr.Image(
                    label="Generated image",
                    type="pil",
                    height=512,
                    elem_id="dipai-gen-image",
                )

                with gr.Accordion("Advanced image settings", open=False):
                    gen_neg = gr.Textbox(
                        label="Negative prompt (what to avoid)",
                        lines=2,
                        value="",
                        placeholder="blurry, low quality, deformed, watermark...",
                    )
                    with gr.Row():
                        gen_width = gr.Slider(256, 1024, value=512, step=64,
                                              label="Width")
                        gen_height = gr.Slider(256, 1024, value=512, step=64,
                                               label="Height")
                    with gr.Row():
                        gen_steps = gr.Slider(1, 60, value=25, step=1,
                                              label="Inference steps")
                        gen_cfg = gr.Slider(1.0, 15.0, value=7.5, step=0.5,
                                            label="Guidance scale (CFG)")
                    gen_seed = gr.Number(
                        value=-1,
                        label="Seed (-1 = random each run; >=0 = reproducible)",
                        precision=0,
                    )

        with gr.Accordion("Generation settings", open=False):
            system_prompt_box = gr.Textbox(
                label="System prompt (instruction for the assistant)",
                lines=2,
                value="",
                placeholder="e.g. You are a concise, helpful assistant.",
            )
            with gr.Row():
                temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.05,
                                        label="Temperature (0 = greedy)")
                top_k = gr.Slider(0, 200, value=40, step=1, label="Top-k (0 = off)")
            with gr.Row():
                top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top-p (1 = off)")
                max_new = gr.Slider(16, 1024, value=200, step=16, label="Max new tokens")
            with gr.Row():
                rep_pen = gr.Slider(1.0, 2.0, value=1.0, step=0.05,
                                    label="Repetition penalty (1 = off)")

        # ---- streaming generation ----------------------------------------
        def respond(
            message: str,
            history: List[Dict],
            stop_event: threading.Event,
            temperature_v: float,
            top_k_v: int,
            top_p_v: float,
            max_new_v: int,
            rep_pen_v: float,
            system_prompt_v: str = "",
        ):
            if not message or not message.strip():
                yield history + [], "", "idle"
                return

            history = list(history or [])
            history.append({"role": "user", "content": message.strip()})
            yield history + [], "", "generating"

            settings = GenerationSettings(
                temperature=float(temperature_v),
                top_k=int(top_k_v),
                top_p=float(top_p_v),
                max_new_tokens=int(max_new_v),
                repetition_penalty=float(rep_pen_v),
            )
            try:
                settings.validate()
            except GenerationError as exc:
                history.append({"role": "assistant", "content": f"[error: {exc}]"})
                yield history, "", "error"
                return

            # Build a session so the engine flattens history + system prompt
            # exactly the way the model expects.
            session = ConversationSession()
            session.system_prompt = (system_prompt_v or "").strip()
            for turn in history[:-1]:  # history already includes the new user msg
                if turn.get("role") in ("user", "assistant") and turn.get("content"):
                    session.add(turn["role"], str(turn["content"]))

            stop_event.clear()
            buf = {"text": ""}
            err: Dict[str, Optional[str]] = {"exc": None}

            def on_text(t: str) -> None:
                buf["text"] = t

            def worker() -> None:
                try:
                    resp = engine.generate(
                        message.strip(),
                        settings=settings,
                        session=session,
                        stream_fn=on_text,
                        stop_event=stop_event.is_set,
                    )
                    buf["text"] = resp
                except Exception as exc:  # noqa: BLE001 - surfaced in UI
                    err["exc"] = str(exc)

            th = threading.Thread(target=worker, daemon=True)
            th.start()

            last_shown = ""
            status_text = "generating"
            while th.is_alive():
                if stop_event.is_set():
                    status_text = "stopped"
                if buf["text"] and buf["text"] != last_shown:
                    last_shown = buf["text"]
                    yield history + [{"role": "assistant", "content": last_shown}], "", status_text
                time.sleep(0.12)

            th.join(timeout=5)
            if err["exc"]:
                history.append({"role": "assistant", "content": f"[error: {err['exc']}]"})
                yield history, "", "error"
                return
            final = buf["text"] or "(no output)"
            # Replace streaming preview with the final text (identical, plus
            # ensures the message is present even for instant generation).
            yield history + [{"role": "assistant", "content": final}], "", "done"

        def do_new(history: List[Dict]):
            return [], "new conversation"

        def do_stop(stop_event: threading.Event):
            stop_event.set()
            return "stopping..."

        # Wire events. The generator's outputs: chatbot, textbox, status.
        chat_inputs = [prompt_box, chatbot, stop_state, temperature, top_k,
                       top_p, max_new, rep_pen, system_prompt_box]
        send_event = send_btn.click(
            respond,
            inputs=chat_inputs,
            outputs=[chatbot, prompt_box, status_box],
        )
        prompt_box.submit(
            respond,
            inputs=chat_inputs,
            outputs=[chatbot, prompt_box, status_box],
        )
        stop_btn.click(do_stop, inputs=[stop_state], outputs=[status_box])
        new_btn.click(do_new, inputs=[chatbot], outputs=[chatbot, status_box])

        # ---- image understanding wiring ---------------------------------
        ask_event = {
            "fn": vision_ask_stream,
            "inputs": [image_in, question_box],
            "outputs": [answer_box, vision_status_box],
        }
        ask_btn.click(**ask_event)
        question_box.submit(**ask_event)
        clear_img_btn.click(
            vision_clear,
            inputs=[],
            outputs=[image_in, question_box, answer_box, vision_status_box],
        )

        # ---- image generation wiring ------------------------------------
        gen_event = {
            "fn": imagegen_generate,
            "inputs": [gen_prompt_box, gen_neg, gen_width, gen_height,
                       gen_steps, gen_cfg, gen_seed],
            "outputs": [gen_image_out, gen_status_box],
        }
        gen_btn.click(**gen_event, api_name="generate_image")
        gen_prompt_box.submit(**gen_event)
        dl_btn.click(
            imagegen_download_stream,
            inputs=[],
            outputs=[gen_status_box],
        )
        unload_btn.click(
            imagegen_unload,
            inputs=[],
            outputs=[gen_status_box],
        )

    return demo


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main() -> int:
    global _VISION_CONFIG_PATH, _IMAGEGEN_CONFIG_PATH

    ap = argparse.ArgumentParser(description="DIPAI web chat")
    ap.add_argument("--ckpt", "-k", default="checkpoints/scratch",
                    help="scratch checkpoint directory")
    ap.add_argument("--adapter", "-a", default=None,
                    help="PEFT adapter directory from dipai.finetune (Mode B); "
                         "overrides --ckpt")
    ap.add_argument("--model", "-m", default=None,
                    help="full-parameter HF model directory (Mode C) or a "
                         "name from the model registry; overrides --adapter")
    ap.add_argument("--list-models", action="store_true",
                    help="list registered models (name, backend, path) and exit")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--host", default="127.0.0.1", help="bind address")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a public gradio link")
    ap.add_argument("--no-stream", action="store_true", help="reserved (streaming is always on)")
    ap.add_argument("--vision-config", default="configs/vision.yaml",
                    help="vision YAML config (default: configs/vision.yaml; "
                         "missing file = built-in defaults)")
    ap.add_argument("--imagegen-config", default="configs/imagegen.yaml",
                    help="image-generation YAML config "
                         "(default: configs/imagegen.yaml)")
    args = ap.parse_args()
    _VISION_CONFIG_PATH = args.vision_config
    _IMAGEGEN_CONFIG_PATH = args.imagegen_config

    log = setup_logging()

    registry = default_registry()
    if args.list_models:
        for name in registry.names():
            m = registry.get(name)
            exists = "ok" if m.exists() else "--"
            print(f"[{exists}] {name:16s} {m.backend:8s} {m.path}")
        return 0

    try:
        if args.model:
            # registry name (dispatches on its backend) or a raw HF model dir
            engine = DIPAIEngine.from_named(args.model, device=args.device)
        elif args.adapter:
            engine = DIPAIEngine.from_adapter(args.adapter, device=args.device)
        else:
            # Default: prefer the instruction-tuned LoRA adapter when it
            # exists (better chat quality than the scratch continuation LM);
            # fall back to the scratch checkpoint otherwise.
            adapter_dir = Path("checkpoints/finetune/adapter")
            if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
                log.info("defaulting to instruct adapter at %s "
                         "(pass --ckpt/--model to override)", adapter_dir)
                engine = DIPAIEngine.from_adapter(adapter_dir, device=args.device)
            else:
                engine = DIPAIEngine.from_checkpoint(args.ckpt, device=args.device)
    except EngineError as exc:
        log.error("cannot start web UI: %s", exc)
        return 2

    # Optional eager vision load. A failure here NEVER blocks the text UI -
    # the vision tab simply reports the error on first use.
    try:
        vcfg = load_vision_config(args.vision_config, required=False)
        if vcfg.load_on_startup:
            vision.load_vision_model(vcfg)
    except VisionError as exc:
        log.warning("vision model not loaded (text chat unaffected): %s", exc)

    app = create_app(engine)
    app.queue(max_size=8)
    url = f"http://{args.host}:{args.port}"
    log.info("DIPAI web UI starting at %s (model: %s)", url, engine.status().checkpoint)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=False,
        show_error=True,
        favicon_path=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
