"""Figures for the report.   python report/make_figures.py"""
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "report" / "figures"
OUT.mkdir(exist_ok=True)

BLUE, GREY, ORANGE, GREEN = "#cfe2f7", "#e6e8eb", "#fde3c4", "#d6efd8"
EDGE = "#44505e"


def box(ax, x, y, w, h, text, fc, fs=7.2):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=EDGE, lw=0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, linespacing=1.15)


def arrow(ax, p, q, rad=0.0, style="-|>", ls="-", color=EDGE):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, lw=0.9, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def architecture():
    fig, ax = plt.subplots(figsize=(7.0, 3.35))
    ax.set_xlim(-1.5, 9.4)
    ax.set_ylim(-0.2, 4.3)
    ax.axis("off")
    cols = [1.4, 3.6, 5.8, 8.0]
    ytop, ymid, ybot = 3.7, 2.1, 0.45
    # top row: shared feature extractor
    box(ax, cols[0], ytop, 1.7, 0.62, "Photo before\nPhoto after", GREY)
    box(ax, cols[1], ytop, 1.7, 0.62, "Blocks 0-1\n(128 x 128)", BLUE)
    box(ax, cols[2], ytop, 1.7, 0.62, "Block 2\n(64 x 64)", BLUE)
    box(ax, cols[3], ytop, 1.7, 0.62, "Block 3\n(32 x 32)", BLUE)
    for a, b in zip(cols, cols[1:]):
        arrow(ax, (a + 0.85, ytop), (b - 0.85, ytop))
    ax.text(5.2, 4.22, "Same feature extractor (EfficientNet-B4, first 4 blocks) applied to both photos",
            ha="center", fontsize=7, style="italic", color=EDGE)
    # middle row: Mix + mask
    labels = ["Mix + mask\n(256)", "Mix + mask\n(128)", "Mix + mask\n(64)", "Mix\n(32)"]
    for x, t in zip(cols, labels):
        box(ax, x, ymid, 1.7, 0.62, t, ORANGE)
        arrow(ax, (x, ytop - 0.33), (x, ymid + 0.33))
    ax.text(cols[0] - 0.95, (ytop + ymid) / 2, "raw\npixels", fontsize=6.3, ha="right", va="center", color=EDGE)
    # bottom row: decoder (right to left)
    box(ax, cols[2], ybot, 1.7, 0.62, "Up x2\n x mask 64", GREEN)
    box(ax, cols[1], ybot, 1.7, 0.62, "Up x2\n x mask 128", GREEN)
    box(ax, cols[0], ybot, 1.7, 0.62, "Up x2\n x mask 256", GREEN)
    box(ax, -0.55, ybot, 1.5, 0.62, "Per-pixel\nclassifier", GREEN)
    arrow(ax, (cols[3], ymid - 0.33), (cols[3], ybot + 0.0))
    arrow(ax, (cols[3], ybot), (cols[2] + 0.85, ybot))
    arrow(ax, (cols[2] - 0.85, ybot), (cols[1] + 0.85, ybot))
    arrow(ax, (cols[1] - 0.85, ybot), (cols[0] + 0.85, ybot))
    arrow(ax, (cols[0] - 0.85, ybot), (-0.55 + 0.75, ybot))
    for x in cols[:3]:   # masks feed the decoder
        arrow(ax, (x, ymid - 0.33), (x, ybot + 0.33), ls="--", color="#b05a00")
    ax.text(cols[3] + 0.1, (ymid + ybot) / 2 + 0.08, "32 x 32\nfeatures", fontsize=6.5, ha="left", va="center", color=EDGE)
    ax.text(-0.55, ybot - 0.55, "P(change) per pixel\n>= 0.5 means changed", fontsize=6.5, ha="center", va="top", color=EDGE)
    ax.text(cols[2] + 0.1, (ymid + ybot) / 2 + 0.08, "mask", fontsize=6.3, ha="left", color="#b05a00")
    fig.savefig(OUT / "architecture.png", dpi=240, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def qualitative():
    from tinycd import load_tinycd, overlay, predict_pair
    net = load_tinycd()
    picks = [("test_102_9", "Large new building"), ("test_109_13", "Small new house"),
             ("test_19_1", "Missed building"), ("test_19_3", "False alarm")]
    fig, axes = plt.subplots(3, 4, figsize=(5.6, 4.3))
    for j, (name, title) in enumerate(picks):
        d = ROOT / "docs" / "samples" / name
        a = np.array(Image.open(d / "A.png").convert("RGB"))
        b = np.array(Image.open(d / "B.png").convert("RGB"))
        gt = np.array(Image.open(d / "label.png")) > 127
        p = predict_pair(net, a, b)
        axes[0, j].imshow(a)
        axes[1, j].imshow(b)
        axes[2, j].imshow(overlay(b, p))
        if gt.any():
            axes[2, j].contour(gt.astype(float), levels=[0.5], colors="cyan", linewidths=0.9)
        axes[0, j].set_title(title, fontsize=7.5)
    for r, t in enumerate(["Before", "After", "TinyCD"]):
        axes[r, 0].set_ylabel(t, fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(wspace=0.03, hspace=0.04, left=0.06, right=0.995, top=0.94, bottom=0.005)
    fig.savefig(OUT / "qualitative.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    qualitative()
    print("figures written to", OUT)
