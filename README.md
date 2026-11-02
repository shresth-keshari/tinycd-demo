# TinyCD: change detection demo

Course project (CE712). A from-scratch re-implementation of **TinyCD** (Codegoni, Lombardi & Ferrari, *Neural Computing and Applications* 35, 2023), checked against the authors' released weights and compared with BIT and ChangeFormer on LEVIR-CD. **No training was done**: every result uses released checkpoints.

TinyCD takes two aerial photos of the same place at different times and marks what changed (e.g. new buildings), using only 285,128 parameters.

## Try it (pick one)

| Way | What you need | Link / command |
|---|---|---|
| **Live demo in the browser** | Just a browser. The model runs locally in the page (ONNX Runtime Web); nothing is uploaded. | https://shresth-keshari.github.io/tinycd-demo/ |
| **Notebook on Colab** | A Google account. Free CPU is enough. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shresth-keshari/tinycd-demo/blob/main/notebooks/TinyCD_demo.ipynb) |
| **One script on your machine** | Python 3.9+ | Windows: `.\run.ps1`   Linux/macOS/Git Bash: `bash run.sh` |

The script creates a virtual environment, installs the packages, **downloads the TinyCD weights from the authors' GitHub repository and verifies their SHA-256**, runs the six sample pairs (saved to `outputs/samples.png`), and opens the live demo at http://127.0.0.1:8000.

Your own images: `python -m tinycd before.png after.png` (same size, aerial view).

## What is here

```
report/TinyCD_report.pdf  3-page report in plain language (LaTeX source + Word version beside it)
notebooks/                Colab-ready walkthrough, outputs included
docs/                     the web demo (GitHub Pages serves this folder): index.html, app.js, tinycd.onnx, samples
tinycd/                   model.py (the network), infer.py (tiled inference), weights.py (download + checksum)
results/                  saved full-test-set numbers (2,048 pairs) for TinyCD, BIT, ChangeFormer; efficiency; variants
scripts/                  export_onnx.py (PyTorch to ONNX, verified), build_site_data.py, serve.py
```

## Results (LEVIR-CD test set, 2,048 pairs, threshold 0.5)

| Model | F1 (%) | Paper F1 | Parameters | GFLOPs (256x256 pair) |
|---|---|---|---|---|
| **TinyCD** | **91.07** | 91.05 | **285,128** | **1.46** |
| BIT | 89.94 | 89.31 | 3,037,026 | 8.76 |
| ChangeFormer V6 | 90.40 | 90.40 | 41,026,674 | 138.80 |

BIT, ChangeFormer and the benchmark numbers were produced once with their official code and weights, on the full LEVIR-CD test split (not bundled, ~1 GB), and saved in `results/`. The notebook recomputes every metric from the saved pixel counts. The live demo shows TinyCD running for real; BIT and ChangeFormer appear there as saved outputs for the sample pairs only. Our BIT F1 is higher than its paper's (89.31); we did not find out why.

## Verification

- The released checkpoint loads into our network with `strict=True` (all keys and shapes agree), and the parameter count (285,128) matches the paper.
- The ONNX model used in the browser matches PyTorch: max probability difference about 3e-6 and identical masks on the test pairs (`python scripts/export_onnx.py` re-checks this).

## Limits

TinyCD was trained on 256x256 crops at 0.5 m per pixel (LEVIR-CD, building changes). It is not expected to work on very different resolutions or sensors. Variant accuracies from the paper's ablation tables are not re-measured (only their parameter counts).

## Sources and licences

- Model: Codegoni et al., [paper](https://doi.org/10.1007/s00521-022-08122-3). Weights are downloaded from [AndreaCodegoni/Tiny_model_4_CD](https://github.com/AndreaCodegoni/Tiny_model_4_CD) and not redistributed here. `tinycd/model.py` is our own implementation of the architecture in the paper.
- BIT ([justchenhao/BIT_CD](https://github.com/justchenhao/BIT_CD)) and ChangeFormer ([wgcban/ChangeFormer](https://github.com/wgcban/ChangeFormer)): only saved results and saved output images are included, no code or weights.
- LEVIR-CD ([Chen & Shi, 2020](https://doi.org/10.3390/rs12101662)): six example pairs are included for academic demonstration. The dataset is for academic use only; commercial use is prohibited.
- Our code: MIT licence (see `LICENSE`).
