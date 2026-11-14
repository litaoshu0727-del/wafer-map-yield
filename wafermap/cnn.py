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

from .data import CLASSES, load_labeled, to_arrays
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
def predict_proba(model: nn.Module, X: torch.Tensor, device: str, batch_size: int = 512) -> np.ndarray:
    """Softmax class probabilities, (N, n_classes)."""
    model.eval()
    probs = [model(one_hot(X[i:i + batch_size].to(device))).softmax(1).cpu()
             for i in range(0, len(X), batch_size)]
    return torch.cat(probs).numpy()


def predict(model: nn.Module, X: torch.Tensor, device: str, batch_size: int = 512) -> np.ndarray:
    return predict_proba(model, X, device, batch_size).argmax(1)


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def subsample_lots(idx: np.ndarray, lots: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Keep the wafers of a random `fraction` of the lots in `idx`.

    Whole lots are kept or dropped, which is how labelling budgets work in
    practice: an engineer reviews a lot, not scattered wafers.
    """
    if fraction >= 1:
        return idx
    unique = np.unique(lots[idx])
    rng = np.random.default_rng(seed)
    keep = rng.choice(unique, size=max(1, round(fraction * len(unique))), replace=False)
    return idx[np.isin(lots[idx], keep)]


def splits(y: np.ndarray, lots: np.ndarray, seed: int):
    """Train / validation / test indices, every lot on one side only.

    Test lots are held out once. Validation lots, carved out of the training
    lots, pick the best epoch, so the test score is never used for tuning.
    The same seed always gives the same split, which pairs runs that differ
    only in their settings.
    """
    fit, test = split_indices(y, lots, seed=seed)
    tr, va = split_indices(y[fit], lots[fit], seed=seed + 1)
    return fit[tr], fit[va], test


def fit(X: torch.Tensor, y: torch.Tensor, train: np.ndarray, val: np.ndarray, device: str, *,
        epochs: int = 20, batch_size: int = 128, lr: float = 2e-3, weight_power: float = 0.5,
        init: str | None = None, encoder_lr_scale: float = 1.0) -> tuple[nn.Module, int, float]:
    """Train on `train`, keep the epoch with the best macro-F1 on `val`.

    X is (N, H, W) uint8 and y (N,) class indices; `train` and `val` index
    into them. Returns the model with its best weights loaded, the best
    epoch and its validation macro-F1.
    """
    y_np = y.numpy()
    model = WaferCNN().to(device)
    if init:
        model.features.load_state_dict(torch.load(init, map_location=device))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_np[train], weight_power).to(device))
    opt = torch.optim.Adam([{"params": model.features.parameters(), "lr": lr * encoder_lr_scale},
                            {"params": model.head.parameters(), "lr": lr}])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    train_t, X_val = torch.from_numpy(train), X[torch.from_numpy(val)]
    best_f1, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_t[torch.randperm(len(train_t))].split(batch_size):
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
    return model, best_epoch, float(best_f1)


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
    ap.add_argument("--seed", type=int, default=0, help="sets both the lot split and the model init")
    ap.add_argument("--train-fraction", type=float, default=1.0,
                    help="keep only this share of the training lots (labelling-budget experiments)")
    ap.add_argument("--init", default=None,
                    help="encoder weights from wafermap.pretrain to start from instead of random")
    ap.add_argument("--encoder-lr-scale", type=float, default=1.0,
                    help="learning-rate multiplier for the convolution blocks (e.g. 0.1 with --init)")
    ap.add_argument("--all-records", action="store_true",
                    help="keep repeat/retest records of the same wafers (dropped by default)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()

    df = load_labeled(args.data, args.all_records)
    if args.max_wafers and len(df) > args.max_wafers:
        df = df.sample(args.max_wafers, random_state=args.seed).reset_index(drop=True)
    X_np, y_np = to_arrays(df, args.size)
    lots = df["lotName"].to_numpy()

    train, val, test = splits(y_np, lots, args.seed)
    train = subsample_lots(train, lots, args.train_fraction, args.seed)
    X, y = torch.from_numpy(X_np), torch.from_numpy(y_np)
    X_val, X_test = X[torch.from_numpy(val)], X[torch.from_numpy(test)]

    print(f"device={device}  train={len(train)}  val={len(val)}  test={len(test)}  "
          f"weight_power={args.weight_power}  train_fraction={args.train_fraction}  "
          f"init={args.init or 'random'}", flush=True)
    model, best_epoch, best_f1 = fit(X, y, train, val, device, epochs=args.epochs, batch_size=args.batch_size,
                                     lr=args.lr, weight_power=args.weight_power, init=args.init,
                                     encoder_lr_scale=args.encoder_lr_scale)
    best_state = model.state_dict()
    val_proba, test_proba = predict_proba(model, X_val, device), predict_proba(model, X_test, device)
    out = Path(args.out)
    score(y_np[test], test_proba.argmax(1), out, args.name,
          {"seed": args.seed, "weight_power": args.weight_power, "train_fraction": args.train_fraction,
           "init": args.init or "random", "encoder_lr_scale": args.encoder_lr_scale, "n_train": int(len(train)), "best_epoch": best_epoch,
           "val_macro_f1": best_f1})
    torch.save(best_state, out / f"{args.name}.pt")
    # Kept so later analysis (confidence routing, error review, Grad-CAM) needs no retraining.
    np.savez_compressed(out / f"{args.name}_pred.npz", val=val, val_proba=val_proba,
                        test=test, test_proba=test_proba, lot=lots[test])


if __name__ == "__main__":
    main()
