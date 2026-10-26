"""Real SD1.5 smoke test: load pipeline, generate 64x64, save with metadata.

Run:  .venv\\Scripts\\python.exe scripts\\smoke_imagegen.py
Exits 0 on success, 1 on load failure, 2 on generation failure.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dipai import imagegen
from dipai.config import load_imagegen_config


def main() -> int:
    cfg = load_imagegen_config("configs/imagegen.yaml")
    print(f"model: {cfg.model_id}")

    t0 = time.time()
    try:
        st = imagegen.load_image_gen(cfg)
        print(f"LOAD OK in {time.time() - t0:.1f}s: {st.line()}")
    except Exception as exc:  # noqa: BLE001
        print(f"LOAD FAILED after {time.time() - t0:.1f}s: "
              f"{type(exc).__name__}: {exc}")
        return 1

    t1 = time.time()
    try:
        img, meta = imagegen.generate_image(
            "a plain red cube on a white table",
            config=cfg,
            width=64,
            height=64,
            steps=2,
            seed=42,
        )
        size = img.size
        print(f"GEN OK in {time.time() - t1:.1f}s: size={size} "
              f"seed={meta['seed']} steps={meta['steps']}")
    except Exception as exc:  # noqa: BLE001
        print(f"GEN FAILED after {time.time() - t1:.1f}s: "
              f"{type(exc).__name__}: {exc}")
        return 2

    path = imagegen.save_image(img, None, meta)
    print(f"SAVED: {path}")

    # Reopen to prove the file is valid.
    from PIL import Image

    with Image.open(path) as reopened:
        print(f"REOPEN OK: {reopened.size} mode={reopened.mode}")
    sidecar = Path(path).with_suffix(".json")
    print(f"METADATA SIDECAR: {'exists' if sidecar.exists() else 'missing'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
