"""Summarise repeated runs: mean and spread over seeds, and paired differences.

    python -m wafermap.seeds --runs reports/wm811k_seeds

Reads every `<config>_s<seed>_metrics.json` in the folder. Runs with the
same seed share the same lot split, so two configs are compared seed by
seed (paired), which removes the split-to-split variation from the gap.
"""
from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CLASSES

RUN = re.compile(r"(?P<config>.+)_s(?P<seed>\d+)_metrics\.json$")


def load_runs(folder: str | Path) -> pd.DataFrame:
    """One row per run: config, seed, macro scores and per-class F1."""
    rows = []
    for path in sorted(Path(folder).glob("*_metrics.json")):
        m = RUN.match(path.name)
        if not m:
            continue
        data = json.loads(path.read_text())
        rows.append({"config": m["config"], "seed": int(m["seed"]),
                     **{k: data[k] for k in ("macro_f1", "macro_precision", "macro_recall", "accuracy")},
                     **{f"f1_{c}": data["f1"][c] for c in CLASSES}})
    return pd.DataFrame(rows)


def summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample standard deviation of each score per config."""
    cols = [c for c in runs.columns if c not in ("config", "seed")]
    grouped = runs.groupby("config")[cols]
    out = grouped.mean().round(3).astype(str) + " ± " + grouped.std(ddof=1).round(3).astype(str)
    out.insert(0, "runs", runs.groupby("config").size())
    return out


def paired(runs: pd.DataFrame, a: str, b: str, metric: str = "macro_f1") -> dict:
    """Seed-by-seed difference b - a on the seeds both configs have."""
    wide = runs.pivot(index="seed", columns="config", values=metric)[[a, b]].dropna()
    diff = (wide[b] - wide[a]).to_numpy()
    sd = diff.std(ddof=1) if len(diff) > 1 else float("nan")
    return {"a": a, "b": b, "seeds": len(diff), "mean_diff": float(diff.mean()), "sd_diff": float(sd),
            "b_wins": int((diff > 0).sum()), "diffs": np.round(diff, 4).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True, help="folder with the *_metrics.json files")
    args = ap.parse_args()

    runs = load_runs(args.runs)
    if runs.empty:
        raise SystemExit(f"no <config>_s<seed>_metrics.json files in {args.runs}")
    table = summary(runs)
    table.to_csv(Path(args.runs) / "seed_summary.csv")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.T.to_string())
    print()
    for a, b in combinations(sorted(runs["config"].unique()), 2):
        p = paired(runs, a, b)
        print(f"{b} - {a}: {p['mean_diff']:+.4f} ± {p['sd_diff']:.4f} over {p['seeds']} seeds, "
              f"{b} higher in {p['b_wins']}  {p['diffs']}")


if __name__ == "__main__":
    main()
