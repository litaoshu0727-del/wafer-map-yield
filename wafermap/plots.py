"""Figures for the report. Every function saves a PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from .data import CLASSES

WAFER_CMAP = ListedColormap(["white", "#d9dee5", "#c0392b"])  # no die / pass / fail
BAR = "#4c6a92"
ALERT = "#c0392b"


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def gallery(df: pd.DataFrame, path: str | Path) -> Path:
    """One example wafer per defect pattern."""
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(2 * len(CLASSES), 2.4))
    for ax, name in zip(axes, CLASSES):
        ax.axis("off")
        ax.set_title(name, fontsize=10)
        rows = df[df["failureType"] == name]
        if len(rows):
            ax.imshow(rows.iloc[0]["waferMap"], cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
    return _save(fig, path)


def pareto(table: pd.DataFrame, path: str | Path) -> Path:
    """Failing dies per pattern, with the cumulative share on a second axis."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(table.index, table["fail_dies"], color=BAR)
    ax.set_ylabel("Failing dies")
    ax.tick_params(axis="x", rotation=30)
    ax2 = ax.twinx()
    ax2.plot(table.index, table["cum_share"] * 100, color=ALERT, marker="o")
    ax2.set_ylabel("Cumulative share of yield loss (%)")
    ax2.set_ylim(0, 105)
    ax.set_title("Yield loss Pareto by defect pattern")
    return _save(fig, path)


def lot_trend(lots: pd.DataFrame, path: str | Path) -> Path:
    """Mean yield per lot in file order, excursion lots highlighted."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    x = np.arange(len(lots))
    ax.plot(x, lots["mean_yield"] * 100, color=BAR, lw=1)
    flagged = lots["excursion"].to_numpy()
    ax.scatter(x[flagged], lots["mean_yield"][flagged] * 100, color=ALERT, zorder=3,
               label="Excursion: wafers share one pattern")
    ax.set_xlabel("Lot (file order)")
    ax.set_ylabel("Mean wafer yield (%)")
    ax.set_title("Lot yield trend")
    ax.legend(frameon=False)
    return _save(fig, path)


def confusion(cm: np.ndarray, path: str | Path, title: str) -> Path:
    """Row-normalised confusion matrix (each row is the recall breakdown of a true class)."""
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)
