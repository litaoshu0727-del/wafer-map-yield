"""Send the wafers a classifier is unsure about to manual review.

No PyTorch needed: works on any (N, n_classes) probability matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .data import CLASSES


def coverage_curve(proba: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Accuracy of the most confident k wafers, for every k.

    Row k: decide the k most confident wafers automatically, send the rest
    to review. `threshold` is the lowest confidence still decided.
    """
    conf = proba.max(1)
    order = np.argsort(-conf, kind="stable")
    correct = (proba.argmax(1) == y)[order]
    k = np.arange(1, len(y) + 1)
    return pd.DataFrame({"coverage": k / len(y), "threshold": conf[order],
                         "accuracy": np.cumsum(correct) / k})


def pick_threshold(proba: np.ndarray, y: np.ndarray, target_accuracy: float) -> float:
    """Lowest confidence threshold whose auto-decided wafers still reach the target accuracy."""
    curve = coverage_curve(proba, y)
    ok = curve[curve["accuracy"] >= target_accuracy]
    return float(ok["threshold"].iloc[-1]) if len(ok) else 1.0


def route(proba: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """Scores of the auto-decided wafers and the share of each class sent to review."""
    auto = proba.max(1) >= threshold
    pred = proba.argmax(1)
    labels = list(range(len(CLASSES)))
    return {
        "threshold": threshold,
        "coverage": float(auto.mean()),
        "auto_accuracy": float((pred[auto] == y[auto]).mean()) if auto.any() else float("nan"),
        "auto_macro_f1": float(f1_score(y[auto], pred[auto], labels=labels, average="macro",
                                        zero_division=0)) if auto.any() else float("nan"),
        "review_share": {c: float((~auto[y == i]).mean()) for i, c in enumerate(CLASSES) if (y == i).any()},
    }
