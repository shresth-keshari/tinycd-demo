"""Load TinyCD and run it on image pairs of any size (256x256 tiling at native resolution)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .model import TinyCD, load_official_weights
from .weights import WEIGHTS_PATH, fetch_weights

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)   # ImageNet statistics, as in the authors' code
STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
TILE = 256


def load_tinycd(device: str = "cpu") -> TinyCD:
    """Build TinyCD and load the released weights strictly (every key and shape must match)."""
    net = TinyCD("efficientnet_b4")
    load_official_weights(net, fetch_weights(WEIGHTS_PATH))
    return net.eval().requires_grad_(False).to(device)


def _to_tensor(img_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_u8).permute(2, 0, 1).float().div(255.0)[None]


@torch.no_grad()
def predict_pair(net: TinyCD, a_u8: np.ndarray, b_u8: np.ndarray, batch: int = 4) -> np.ndarray:
    """(H, W, 3) uint8 before/after images -> float32 (H, W) probability of change."""
    if a_u8.shape != b_u8.shape:
        raise ValueError(f"images must have the same size, got {a_u8.shape} and {b_u8.shape}")
    dev = next(net.parameters()).device
    a, b = _to_tensor(a_u8).to(dev), _to_tensor(b_u8).to(dev)
    h, w = a.shape[-2:]
    ph, pw = (-h) % TILE, (-w) % TILE
    if ph or pw:
        mode = "reflect" if ph < h and pw < w else "replicate"
        a, b = F.pad(a, (0, pw, 0, ph), mode=mode), F.pad(b, (0, pw, 0, ph), mode=mode)
    H, W = a.shape[-2:]
    tiles = lambda x: x.unfold(2, TILE, TILE).unfold(3, TILE, TILE).permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, TILE, TILE)
    ta, tb = tiles(a), tiles(b)
    nh, nw = H // TILE, W // TILE
    mean, std = MEAN.to(dev), STD.to(dev)
    out = torch.cat([net((ta[i:i + batch] - mean) / std, (tb[i:i + batch] - mean) / std)
                     for i in range(0, len(ta), batch)])
    out = out.reshape(nh, nw, TILE, TILE).permute(0, 2, 1, 3).reshape(H, W)
    return out[:h, :w].float().cpu().numpy()


def overlay(img_u8: np.ndarray, prob: np.ndarray, threshold: float = 0.5, alpha: float = 0.55) -> np.ndarray:
    """Tint pixels with prob >= threshold red on top of the image."""
    out = img_u8.astype(np.float32).copy()
    m = prob >= threshold
    out[m] = (1 - alpha) * out[m] + alpha * np.array([255, 40, 40], np.float32)
    return out.astype(np.uint8)
