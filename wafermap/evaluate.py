"""Shared split and scoring, so the baseline and the CNN are compared on the same wafers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold

from . import plots
from .data import CLASSES


def split_indices(y: np.ndarray, groups: np.ndarray, seed: int = 0):
    """80/20 train/test indices, stratified by class, with each lot kept on one side.

    Wafers from one lot often share the same defect signature, so splitting
    them across train and test would leak the answer and inflate the score.
    """
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return next(splitter.split(np.zeros(len(y)), y, groups))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Macro and per-class scores as plain numbers, for comparing runs."""
    idx = list(range(len(CLASSES)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=idx, zero_division=0)
    return {"macro_f1": float(f.mean()), "macro_precision": float(p.mean()),
            "macro_recall": float(r.mean()), "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": dict(zip(CLASSES, map(float, f))), "precision": dict(zip(CLASSES, map(float, p))),
            "recall": dict(zip(CLASSES, map(float, r)))}


def score(y_true: np.ndarray, y_pred: np.ndarray, out_dir: str | Path, name: str,
          extra: dict | None = None) -> float:
    """Print per-class metrics, save the confusion matrix and a metrics JSON, return macro-F1.

    Macro-F1 is the headline number: most wafers are "none", so accuracy
    stays high even when every rare pattern is missed.
    """
    idx = list(range(len(CLASSES)))
    macro = f1_score(y_true, y_pred, labels=idx, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=idx, target_names=CLASSES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=idx)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_report.txt").write_text(f"macro-F1: {macro:.4f}\n\n{report}")
    (out_dir / f"{name}_metrics.json").write_text(
        json.dumps({"name": name, **(extra or {}), **metrics(y_true, y_pred)}, indent=2))
    plots.confusion(cm, out_dir / f"{name}_confusion.png", f"{name}: macro-F1 {macro:.3f}")
    print(report)
    print(f"{name} macro-F1: {macro:.4f}")
    return float(macro)
