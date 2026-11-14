"""Pseudo-labelling: let a trained CNN label unlabeled wafers, then retrain on both.

    python -m wafermap.pseudo --data data/wm811k/LSWMD.pkl --out reports/wm811k_pseudo \\
        --name f0.1_s0 --seed 0 --train-fraction 0.1 --epochs 40

1. A teacher is trained on the labeled training lots (or a share of them).
2. It predicts every first-test wafer from lots without any label. Only
   predictions at or above `--confidence` are kept as pseudo-labels.
3. A student is trained from scratch on the labeled wafers plus the
   pseudo-labeled ones. Validation and test lots are the same real labels
   for both, so the two scores are directly comparable.

The unlabeled pool shares no lot with the labeled data, so no validation
or test wafer can enter training through a pseudo-label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cnn import fit, pick_device, predict_proba, splits, subsample_lots
from .data import CLASSES, load_labeled, to_arrays
from .evaluate import metrics
from .pretrain import unlabeled_wafers

NONE = CLASSES.index("none")


def select_pseudo(proba: np.ndarray, confidence: float, none_ratio: float, seed: int = 0) -> np.ndarray:
    """Indices of the unlabeled wafers to keep as pseudo-labeled.

    All pattern predictions at or above `confidence` are kept. Confident
    "none" predictions vastly outnumber them, so they are subsampled to at
    most `none_ratio` times the number of kept pattern wafers.
    """
    sure = proba.max(1) >= confidence
    label = proba.argmax(1)
    pattern = np.flatnonzero(sure & (label != NONE))
    none = np.flatnonzero(sure & (label == NONE))
    limit = int(none_ratio * len(pattern))
    if len(none) > limit:
        none = np.random.default_rng(seed).choice(none, size=limit, replace=False)
    return np.sort(np.concatenate([pattern, none]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="reports/pseudo")
    ap.add_argument("--name", default="pseudo")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--weight-power", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-fraction", type=float, default=1.0)
    ap.add_argument("--confidence", type=float, default=0.9)
    ap.add_argument("--none-ratio", type=float, default=3.0,
                    help="keep at most this many pseudo 'none' wafers per pseudo pattern wafer")
    ap.add_argument("--max-unlabeled", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    df = load_labeled(args.data)
    X_np, y_np = to_arrays(df, args.size)
    lots = df["lotName"].to_numpy()
    train, val, test = splits(y_np, lots, args.seed)
    train = subsample_lots(train, lots, args.train_fraction, args.seed)
    X, y = torch.from_numpy(X_np), torch.from_numpy(y_np)
    X_test = X[torch.from_numpy(test)]
    opts = dict(epochs=args.epochs, weight_power=args.weight_power)

    print(f"--- teacher: {len(train)} labeled wafers", flush=True)
    teacher, *_ = fit(X, y, train, val, device, **opts)
    teacher_scores = metrics(y_np[test], predict_proba(teacher, X_test, device).argmax(1))

    U = torch.from_numpy(unlabeled_wafers(args.data, args.size, args.max_unlabeled, args.seed))
    proba = predict_proba(teacher, U, device)
    keep = select_pseudo(proba, args.confidence, args.none_ratio, args.seed)
    pseudo_y = proba.argmax(1)[keep]
    counts = {c: int((pseudo_y == i).sum()) for i, c in enumerate(CLASSES)}
    print(f"--- pseudo-labels: {len(keep)} of {len(U)} unlabeled wafers kept "
          f"({(proba.max(1) >= args.confidence).mean():.0%} were confident)  {counts}", flush=True)

    X_all = torch.cat([X, U[torch.from_numpy(keep)]])
    y_all = torch.cat([y, torch.from_numpy(pseudo_y)])
    student_train = np.concatenate([train, len(X) + np.arange(len(keep))])
    print(f"--- student: {len(train)} labeled + {len(keep)} pseudo-labeled wafers", flush=True)
    torch.manual_seed(args.seed)
    student, *_ = fit(X_all, y_all, student_train, val, device, **opts)
    student_scores = metrics(y_np[test], predict_proba(student, X_test, device).argmax(1))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"name": args.name, "seed": args.seed, "train_fraction": args.train_fraction,
              "confidence": args.confidence, "n_labeled": int(len(train)), "n_unlabeled": int(len(U)),
              "n_pseudo": int(len(keep)), "pseudo_counts": counts,
              "teacher": teacher_scores, "student": student_scores}
    (out / f"{args.name}_pseudo.json").write_text(json.dumps(result, indent=2))
    torch.save(student.state_dict(), out / f"{args.name}_student.pt")
    print(f"teacher macro-F1 {teacher_scores['macro_f1']:.4f}  student macro-F1 {student_scores['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
