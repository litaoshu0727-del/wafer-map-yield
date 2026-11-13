"""Self-supervised pretraining of the CNN encoder on unlabeled wafer maps (SimCLR).

    python -m wafermap.pretrain --data data/wm811k/LSWMD.pkl --out reports/pretrain --epochs 10
    python -m wafermap.cnn ... --init reports/pretrain/encoder.pt

Only about a fifth of WM-811K is labeled. This learns features from the
rest: two randomly altered copies of each wafer (rotated, flipped, with
some failing dies added or removed) should map to nearby vectors, and
copies of different wafers to distant ones. A defect pattern survives these
changes while the exact die positions do not, so the encoder has to learn
the pattern.

Pretraining uses only first-test, unlabeled wafers from lots that contain
no labeled wafer at all. No lot used for validation or testing is ever
seen here, whatever the split seed.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .cnn import WaferCNN, one_hot, pick_device
from .data import CLASSES, load_dataset, resize


def unlabeled_wafers(path: str, size: int = 64, max_wafers: int | None = None, seed: int = 0) -> np.ndarray:
    """(N, size, size) uint8 maps of first-test wafers from fully unlabeled lots."""
    from .records import classify_lots

    df = load_dataset(path)
    df = df[classify_lots(df) == "first"]
    has_label = df["failureType"].isin(CLASSES)
    labeled_lots = set(df.loc[has_label, "lotName"])
    df = df[~df["lotName"].isin(labeled_lots)]
    if max_wafers and len(df) > max_wafers:
        df = df.sample(max_wafers, random_state=seed)
    print(f"pretraining wafers: {len(df)} from {df['lotName'].nunique()} unlabeled lots", flush=True)
    return np.stack([resize(m, size) for m in df["waferMap"]])


def random_view(x: torch.Tensor, p_add: float = 0.06, p_drop: float = 0.4, zoom: tuple = (0.8, 1.15),
                max_shift: float = 0.08) -> torch.Tensor:
    """One random altered copy of each wafer in a (N, H, W) uint8 batch.

    Per wafer:

    - rotation by a multiple of 90 degrees and a random flip;
    - a random zoom (factor in `zoom`, mostly shrinking so the wafer edge
      stays in view, since edge patterns are defined by it) and shift (up to
      `max_shift` of the width). Without it, the outline of the die grid is identical in both
      views, and the encoder can pair views by outline and fail count alone
      without learning anything about the pattern (the first version did);
    - die noise of random strength: each passing die fails with probability
      up to `p_add`, each failing die passes with probability up to
      `p_drop`, so the fail count no longer identifies a wafer either.
    """
    n, dev = len(x), x.device
    k = torch.randint(4, (n,), device=dev)
    out = x.clone()
    for r in range(1, 4):
        sel = k == r
        if sel.any():
            out[sel] = torch.rot90(x[sel], r, dims=(1, 2))
    flip = torch.rand(n, device=dev) < 0.5
    out[flip] = torch.flip(out[flip], dims=(2,))
    if zoom != (1.0, 1.0) or max_shift > 0:
        scale = 1 / (zoom[0] + torch.rand(n, device=dev) * (zoom[1] - zoom[0]))
        theta = torch.zeros(n, 2, 3, device=dev)
        theta[:, 0, 0] = theta[:, 1, 1] = scale
        theta[:, :, 2] = (torch.rand(n, 2, device=dev) * 2 - 1) * max_shift
        grid = nn.functional.affine_grid(theta, (n, 1, *x.shape[1:]), align_corners=False)
        out = nn.functional.grid_sample(out[:, None].float(), grid, mode="nearest", padding_mode="zeros",
                                        align_corners=False)[:, 0].to(torch.uint8)
    u = torch.rand(out.shape, device=dev)
    add = torch.rand(n, 1, 1, device=dev) * p_add
    drop = torch.rand(n, 1, 1, device=dev) * p_drop
    out = torch.where((out == 1) & (u < add), torch.full_like(out, 2), out)
    out = torch.where((out == 2) & (u < drop), torch.full_like(out, 1), out)
    return out


class SimCLR(nn.Module):
    """Encoder (the CNN's convolution blocks) plus a small projection head used only here."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.encoder = WaferCNN().features
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 256), nn.ReLU(),
                                  nn.Linear(256, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.head(self.encoder(x)), dim=1)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """SimCLR loss: each view must pick out its partner among all 2N - 1 other views."""
    z = torch.cat([z1, z2])
    sim = z @ z.T / temperature
    sim.fill_diagonal_(float("-inf"))
    n = len(z1)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device)
    return nn.functional.cross_entropy(sim, target)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="reports/pretrain")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-wafers", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (for timing)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    X = torch.from_numpy(unlabeled_wafers(args.data, args.size, args.max_wafers, args.seed))
    model = SimCLR().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = len(X) // args.batch_size
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    step, start = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        order = torch.randperm(len(X))[:steps_per_epoch * args.batch_size]
        for batch in order.split(args.batch_size):
            xb = X[batch].to(device)
            loss = nt_xent(model(one_hot(random_view(xb))), model(one_hot(random_view(xb))), args.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            schedule.step()
            total += loss.item()
            step += 1
            if args.max_steps and step >= args.max_steps:
                print(f"{step} steps in {time.time() - start:.0f}s", flush=True)
                return
        print(f"epoch {epoch:2d}  loss {total / steps_per_epoch:.4f}  {time.time() - start:.0f}s", flush=True)
        torch.save(model.encoder.state_dict(), out / "encoder.pt")


if __name__ == "__main__":
    main()
