"""Labelling-budget curve: macro-F1 against the share of labeled training lots.

    python -m wafermap.budget --runs reports/wm811k_budget --out reports/wm811k_budget

Reads the `*_metrics.json` files written by wafermap.cnn, which record the
seed, `train_fraction` and `init` (random or a pretrained encoder). For
each fraction it compares random and pretrained starts seed by seed: the
same seed means the same test lots and the same subset of training lots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import _save

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order: random start first


def load(folder: str | Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(folder).glob("*_metrics.json")):
        m = json.loads(path.read_text())
        if "train_fraction" not in m:
            continue
        init = m.get("init", "random")
        start = "random" if init == "random" else f"{Path(init).parent.name}, encoder lr x{m.get('encoder_lr_scale', 1.0):g}"
        rows.append({"start": start,
                     "fraction": float(m["train_fraction"]), "seed": int(m["seed"]),
                     "n_train": int(m["n_train"]), "macro_f1": m["macro_f1"],
                     "macro_recall": m["macro_recall"], "macro_precision": m["macro_precision"]})
    return pd.DataFrame(rows)


def table(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread per start and fraction, plus each pretrained start's paired gain over random."""
    stats = (runs.groupby(["fraction", "start"])
             .agg(runs=("seed", "size"), n_train=("n_train", "mean"),
                  f1_mean=("macro_f1", "mean"), f1_sd=("macro_f1", "std"))
             .unstack("start"))
    wide = runs.pivot_table(index=["fraction", "seed"], columns="start", values="macro_f1")
    for start in wide.columns.drop("random", errors="ignore"):
        diff = (wide[start] - wide["random"]).dropna().groupby("fraction")
        stats[(f"gain: {start}", "mean")] = diff.mean()
        stats[(f"gain: {start}", "sd")] = diff.std()
        stats[(f"gain: {start}", "seeds_up")] = diff.apply(lambda d: f"{int((d > 0).sum())}/{len(d)}")
    return stats


def plot(runs: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    starts = ["random"] + sorted(set(runs["start"]) - {"random"})
    for start, color in zip(starts, COLORS):
        group = runs[runs["start"] == start]
        g = group.groupby("fraction")["macro_f1"].agg(["mean", "std"]).reset_index()
        ax.errorbar(g["fraction"] * 100, g["mean"], yerr=g["std"].fillna(0), color=color, lw=2,
                    marker="o", ms=7, mec="white", mew=1.5, capsize=3,
                    label="Random start" if start == "random" else f"Pretrained ({start})")
    ax.set_xscale("log")
    ticks = sorted(runs["fraction"].unique() * 100)
    ax.set_xticks(ticks, [f"{t:g}%" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Share of labeled training lots used")
    ax.set_ylabel("Test macro-F1 (mean ± sd over seeds)")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("How many labels does the CNN need?")
    return _save(fig, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = load(args.runs)
    out = Path(args.out or args.runs)
    t = table(runs)
    t.to_csv(out / "budget_summary.csv")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(t.round(4).to_string())
    print(plot(runs, out / "budget.png"))


if __name__ == "__main__":
    main()
