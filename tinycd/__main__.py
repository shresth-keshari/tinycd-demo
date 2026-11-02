"""Command line use.

    python -m tinycd                       # run the 6 bundled sample pairs, save outputs/samples.png
    python -m tinycd BEFORE.png AFTER.png  # run on your own pair, save outputs/result.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from . import load_tinycd, overlay, predict_pair

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def _load(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _f1(pred: np.ndarray, gt: np.ndarray) -> float:
    tp, fp, fn = (pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def samples(net) -> Path:
    items = json.loads((ROOT / "docs" / "samples" / "index.json").read_text())
    fig, axes = plt.subplots(len(items), 4, figsize=(11, 2.8 * len(items)))
    for row, s in zip(axes, items):
        d = ROOT / "docs" / "samples" / s["dir"]
        a, b = _load(d / "A.png"), _load(d / "B.png")
        gt = np.array(Image.open(d / "label.png")) > 127
        p = predict_pair(net, a, b)
        score = f"F1={_f1(p >= .5, gt):.2f}" if gt.any() else f"{int((p >= .5).sum())} false-alarm px"
        print(f"{s['dir']:12s} {s['title']:32s} {score}")
        for ax, im, t in zip(row, [a, b, overlay(b, gt.astype(float)), overlay(b, p)],
                             ["Before", "After", "Ground truth", f"TinyCD  {score}"]):
            ax.imshow(im)
            ax.set_title(t, fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "samples.png"
    fig.savefig(path, dpi=110)
    return path


def pair(net, before, after) -> Path:
    a, b = _load(before), _load(after)
    p = predict_pair(net, a, b)
    print(f"{(p >= .5).mean() * 100:.1f}% of pixels marked as changed")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, im, t in zip(axes, [a, b, overlay(b, p)], ["Before", "After", "TinyCD change (red)"]):
        ax.imshow(im)
        ax.set_title(t)
        ax.axis("off")
    plt.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / "result.png"
    fig.savefig(path, dpi=110)
    return path


if __name__ == "__main__":
    net = load_tinycd()
    print(f"TinyCD loaded: {sum(p.numel() for p in net.parameters()):,} parameters")
    if len(sys.argv) == 3:
        out = pair(net, sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 1:
        out = samples(net)
    else:
        sys.exit(__doc__)
    print(f"Saved {out}")
