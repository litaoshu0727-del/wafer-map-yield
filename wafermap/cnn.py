"""CNN classifier on the raw wafer maps (needs PyTorch).

    python -m wafermap.cnn --data data/synthetic.pkl --out reports --epochs 20
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .data import CLASSES, labeled, load_dataset, to_arrays
from .evaluate import score, split_indices


def one_hot(x: torch.Tensor) -> torch.Tensor:
    """(N, H, W) uint8 with values {0,1,2} -> (N, 3, H, W) float: no-die / pass / fail channels.

    Done per batch: one-hot floats for the whole of WM-811K would not fit in memory.
    """
    return nn.functional.one_hot(x.long(), 3).permute(0, 3, 1, 2).float()


def augment(x: torch.Tensor) -> torch.Tensor:
    """Random 90-degree rotation and flip of the batch.

    Every pattern class is defined by shape and radial position, not
    orientation, so these transforms keep the label valid.
    """
    x = torch.rot90(x, int(torch.randint(4, (1,))), dims=(2, 3))
    return torch.flip(x, dims=(3,)) if torch.rand(1) < 0.5 else x


class WaferCNN(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES)):
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
                                 nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(128, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def class_weights(y: np.ndarray, power: float) -> torch.Tensor:
    """Loss weight per class: (inverse class frequency) ** power.

    power 0 treats every wafer equally, power 1 treats every class equally.
    On WM-811K power 1 makes a Near-full wafer count ~1000x a "none" wafer,
    which buys recall on rare patterns at the cost of many false alarms.
    """
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    weights = np.where(counts > 0, (len(y) / (len(CLASSES) * np.maximum(counts, 1))) ** power, 0.0)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, device: str, batch_size: int = 512) -> np.ndarray:
    model.eval()
    preds = [model(one_hot(X[i:i + batch_size].to(device))).argmax(1).cpu()
             for i in range(0, len(X), batch_size)]
    return torch.cat(preds).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--name", default="cnn", help="prefix for the output files")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-power", type=float, default=0.5,
                    help="0 = no class weighting, 1 = full inverse-frequency weighting")
    ap.add_argument("--max-wafers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    df = labeled(load_dataset(args.data))
    if args.max_wafers and len(df) > args.max_wafers:
        df = df.sample(args.max_wafers, random_state=args.seed).reset_index(drop=True)
    X_np, y_np = to_arrays(df, args.size)
    lots = df["lotName"].to_numpy()

    # Test lots are held out once. A validation set, carved out of the training
    # lots, picks the best epoch, so the test score is never used for tuning.
    fit, test = split_indices(y_np, lots, seed=args.seed)
    tr, va = split_indices(y_np[fit], lots[fit], seed=args.seed + 1)
    train, val = fit[tr], fit[va]
    X, y = torch.from_numpy(X_np), torch.from_numpy(y_np)
    X_val, X_test = X[torch.from_numpy(val)], X[torch.from_numpy(test)]

    model = WaferCNN().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_np[train], args.weight_power).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"device={device}  train={len(train)}  val={len(val)}  test={len(test)}  "
          f"weight_power={args.weight_power}", flush=True)

    train_t = torch.from_numpy(train)
    best_f1, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_t[torch.randperm(len(train_t))].split(args.batch_size):
            xb, yb = augment(one_hot(X[batch].to(device))), y[batch].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        schedule.step()

        val_f1 = f1_score(y_np[val], predict(model, X_val, device), labels=list(range(len(CLASSES))),
                          average="macro", zero_division=0)
        if val_f1 > best_f1:
            best_f1, best_epoch, best_state = val_f1, epoch, copy.deepcopy(model.state_dict())
        print(f"epoch {epoch:2d}  train loss {total / len(train_t):.4f}  val macro-F1 {val_f1:.4f}", flush=True)

    print(f"best epoch {best_epoch}  val macro-F1 {best_f1:.4f}")
    model.load_state_dict(best_state)
    score(y_np[test], predict(model, X_test, device), args.out, args.name)
    torch.save(best_state, Path(args.out) / f"{args.name}.pt")


if __name__ == "__main__":
    main()
