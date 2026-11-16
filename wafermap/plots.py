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


def commonality_figure(null: pd.DataFrame, power: pd.DataFrame, path: str | Path) -> Path:
    """Left: false alarms with no guilty tool. Right: how often a planted tool is found."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 3.9))
    x = np.arange(len(null))
    bars = [("fisher", "Fisher + BH", ALERT), ("shift", "shift-calibrated", "#a9bbd3"),
            ("local", "shift-calibrated, time-stratified", "#1f3552")]
    for i, (key, label, color) in enumerate(bars):
        left.bar(x + 0.27 * (i - 1), 100 * null[f"{key}_false_alarm"], 0.25, color=color, label=label)
    left.axhline(5, color="0.4", lw=1, ls="--")
    left.text(-0.45, 7, "5% target", fontsize=8, color="0.3")
    left.set_xticks(x, [f"{int(r)}" for r in null["run_length"]])
    left.set_xlabel("mean lots in a row on the same tool (synthetic routes)")
    left.set_ylabel("runs accusing an innocent tool (%)")
    left.set_title("No guilty tool: false alarms", fontsize=10)
    left.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)

    shades = ["#a9bbd3", "#4c6a92", "#1f3552"]
    for shade, (lots, g) in zip(shades, power.groupby("lots")):
        right.plot(100 * g["strength"], 100 * g["local_found"], marker="o", lw=2, color=shade,
                   label=f"{lots:,} lots (~{g['mean_bad_lots'].iloc[0]:.0f} excursions)")
    right.set_xlabel("share of excursion lots sent through the planted tool (%)")
    right.set_ylabel("runs in which it is found (%)")
    right.set_ylim(-3, 103)
    right.set_title("One planted tool: detection by the time-stratified test", fontsize=10)
    right.legend(frameon=False, fontsize=8)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def spatial_figure(lab: pd.DataFrame, path: str | Path) -> Path:
    """Spread of the join-count z per labeled class (5th to 95th percentile, box = quartiles)."""
    order = lab.groupby("label")["z"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, name in enumerate(order):
        lo, q1, med, q3, hi = np.percentile(lab.loc[lab["label"] == name, "z"], [5, 25, 50, 75, 95])
        color = "0.55" if name == "none" else BAR
        ax.plot([lo, hi], [i, i], color=color, lw=1.5)
        ax.barh(i, q3 - q1, left=q1, height=0.5, color=color)
        ax.plot([med, med], [i - 0.25, i + 0.25], color="white", lw=2)
    ax.axvline(1.645, color=ALERT, lw=1, ls="--")
    ax.text(1.645, len(order) - 0.4, " 5% level", color=ALERT, fontsize=8, va="center")
    ax.set_xscale("symlog", linthresh=5)
    ax.set_xlim(-4, 60)
    ax.set_xticks([-2, 0, 2, 5, 10, 20, 40], ["-2", "0", "2", "5", "10", "20", "40"])
    ax.minorticks_off()
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("join-count z: fail-fail neighbour pairs above chance, in standard deviations")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)
