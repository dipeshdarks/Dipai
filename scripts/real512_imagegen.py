"""Realistic-size SD1.5 test: 512x512, 20 steps, example prompt + seed check.

Run:  .venv\\Scripts\\python.exe scripts\\real512_imagegen.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dipai import imagegen
from dipai.config import load_imagegen_config

PROMPT = (
    "Photorealistic commercial product photograph of a transparent glass "
    "filled with iced milk tea, visible tapioca pearls, natural condensation "
    "droplets, soft window light from the left, physically accurate reflections, "
    "subtle shadows on a real tabletop, 50mm lens, realistic exposure, detailed "
    "beverage textures, premium editorial photography."
)


def main() -> int:
    cfg = load_imagegen_config("configs/imagegen.yaml")

    t0 = time.time()
    st = imagegen.load_image_gen(cfg)
    print(f"LOAD OK {time.time() - t0:.0f}s: {st.line()}", flush=True)

    t1 = time.time()
    try:
        img, meta = imagegen.generate_image(
            PROMPT, config=cfg, width=512, height=512, steps=20, seed=1234,
        )
        print(f"GEN OK {time.time() - t1:.1f}s: size={img.size} "
              f"seed={meta['seed']} steps={meta['steps']}", flush=True)
        out = imagegen.save_image(img, None, meta)
        print(f"SAVED: {out}", flush=True)
        side = Path(out).with_suffix(".json")
        print(f"SIDECAR exists: {side.exists()}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"GEN FAILED {time.time() - t1:.1f}s: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return 2

    # Reproducibility: same seed + settings twice at 4 steps must match.
    try:
        a_img, _ = imagegen.generate_image(
            PROMPT, config=cfg, width=512, height=512, steps=4, seed=777,
        )
        b_img, _ = imagegen.generate_image(
            PROMPT, config=cfg, width=512, height=512, steps=4, seed=777,
        )
        same = bool((np.asarray(a_img) == np.asarray(b_img)).all())
        print(f"SEED REPRODUCIBLE: {same}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"SEED CHECK FAILED: {type(exc).__name__}: {exc}", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
