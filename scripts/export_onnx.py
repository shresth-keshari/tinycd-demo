"""Export TinyCD (with ImageNet normalisation inside) to ONNX for the in-browser demo, and check it
against PyTorch.   python scripts/export_onnx.py"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinycd import load_tinycd, predict_pair  # noqa: E402
from tinycd.infer import MEAN, STD  # noqa: E402


class Wrapped(torch.nn.Module):
    """Inputs are RGB in [0, 1], shape (1, 3, 256, 256); output is P(change), (1, 1, 256, 256)."""
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("m", MEAN)
        self.register_buffer("s", STD)

    def forward(self, a, b):
        return self.net((a - self.m) / self.s, (b - self.m) / self.s)


def main():
    net = load_tinycd()
    out = ROOT / "docs" / "tinycd.onnx"
    x = torch.rand(1, 3, 256, 256)
    torch.onnx.export(Wrapped(net).eval(), (x, x), str(out), input_names=["before", "after"],
                      output_names=["prob"], opset_version=17, dynamo=False)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    worst = 0.0
    for name in ("test_102_9", "test_19_1", "test_109_13"):
        a = np.array(Image.open(ROOT / "samples" / name / "A.png").convert("RGB"))
        b = np.array(Image.open(ROOT / "samples" / name / "B.png").convert("RGB"))
        ref = predict_pair(net, a, b)
        f = lambda im: (im.astype(np.float32) / 255).transpose(2, 0, 1)[None]
        got = sess.run(None, {"before": f(a), "after": f(b)})[0][0, 0]
        d = float(np.abs(ref - got).max())
        agree = float(((ref >= .5) == (got >= .5)).mean())
        print(f"{name}: max |torch - onnx| = {d:.2e}, mask agreement {agree:.6f}")
        worst = max(worst, d)
    assert worst < 1e-3, "ONNX output deviates from PyTorch"


if __name__ == "__main__":
    main()
