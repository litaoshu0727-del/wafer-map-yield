"""How much do the conclusions depend on thresholds that were chosen by hand?

    python -m wafermap.sensitivity --data data/wm811k/LSWMD.pkl --out reports/wm811k_sensitivity

Two sets of thresholds are judgement calls with no ground truth behind them:

- the four in `records.py` that decide whether a lot is a repeat or retest
  record of the previous lot;
- the two in the excursion rule (share of wafers with one pattern, and the
  minimum number of labeled wafers in the lot).

Each is varied on its own around the default while the others stay fixed,
and the headline numbers are recomputed. A conclusion that moves little
across the range does not hinge on the exact value picked.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CLASSES, load_dataset
from .records import DEFAULTS, lot_pair_stats, tag_lots

RECORD_GRID = {"min_shared": [0.6, 0.7, 0.8, 0.9, 1.0], "nested": [0.8, 0.9, 0.95, 1.0],
               "same_size": [0.6, 0.7, 0.8, 0.9], "few_fails": [0.1, 0.2, 0.3, 0.4]}
SHARE_GRID = [0.2, 0.3, 0.4, 0.5]
WAFER_GRID = [3, 5, 10]


def lot_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per lot: wafers, labeled wafers, die counts, and its most common labeled pattern."""
    die = df["waferMap"].map(lambda m: int((m > 0).sum()))
    fail = df["waferMap"].map(lambda m: int((m == 2).sum()))
    labeled = df["failureType"].isin(CLASSES)
    frame = pd.DataFrame({"lot": df["lotName"].to_numpy(), "labeled": labeled.to_numpy(),
                          "yield": (1 - fail / die).to_numpy(), "pattern": df["failureType"].to_numpy()})
    lots = frame.groupby("lot", sort=False).agg(n_wafers=("labeled", "size"), n_labeled=("labeled", "sum"))
    lab = frame[frame["labeled"]]
    lots["mean_yield"] = lab.groupby("lot")["yield"].mean()
    patterned = lab[lab["pattern"] != "none"]
    top = patterned.groupby(["lot", "pattern"]).size().rename("n").reset_index()
    top = top.sort_values(["lot", "n", "pattern"], ascending=[True, False, True]).drop_duplicates("lot")
    top = top.set_index("lot")
    lots["top_pattern"] = top["pattern"]
    lots["top_count"] = top["n"].reindex(lots.index).fillna(0).astype(int)
    return lots


def excursions(lots: pd.DataFrame, min_share: float = 0.3, min_wafers: int = 5) -> dict:
    """Excursion summary over lots with labeled wafers (same rule as `analysis.lot_summary`)."""
    eligible = lots[lots["n_labeled"] >= min_wafers]
    flagged = eligible[eligible["top_count"] >= min_share * eligible["n_labeled"]]
    normal = eligible.drop(flagged.index)
    mix = flagged["top_pattern"].value_counts()
    return {"eligible_lots": len(eligible), "excursions": len(flagged),
            "excursion_share": len(flagged) / max(len(eligible), 1),
            "yield_gap_points": 100 * (normal["mean_yield"].mean() - flagged["mean_yield"].mean()),
            "top_pattern": mix.index[0] if len(mix) else "", "top_pattern_share": mix.iloc[0] / max(len(flagged), 1),
            "center_excursions": int(mix.get("Center", 0))}


def record_sweep(stats: pd.DataFrame, lots: pd.DataFrame) -> pd.DataFrame:
    """Vary each record threshold on its own; recount records and excursions each time."""
    rows = []
    for name, grid in RECORD_GRID.items():
        for value in grid:
            tags = tag_lots(stats, **{**DEFAULTS, name: value})
            later = lots.index.isin(tags.index[tags != "first"])
            first = lots[~later]
            rows.append({"threshold": name, "value": value, "default": value == DEFAULTS[name],
                         "repeat_lots": int((tags == "repeat").sum()), "retest_lots": int((tags == "retest").sum()),
                         "later_lot_share": later.mean(),
                         "wafers_in_later_records": int(lots.loc[later, "n_wafers"].sum()),
                         "labeled_wafers_dropped": int(lots.loc[later, "n_labeled"].sum()),
                         **{k: v for k, v in excursions(first).items()
                            if k in ("excursions", "center_excursions", "yield_gap_points")}})
    return pd.DataFrame(rows)


def excursion_sweep(first_lots: pd.DataFrame) -> pd.DataFrame:
    rows = [{"min_share": s, "min_wafers": w, "default": (s, w) == (0.3, 5), **excursions(first_lots, s, w)}
            for w in WAFER_GRID for s in SHARE_GRID]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="reports/sensitivity")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    stats = lot_pair_stats(df)
    lots = lot_table(df)
    stats.to_csv(out / "lot_pair_stats.csv")

    records = record_sweep(stats, lots)
    records.to_csv(out / "record_thresholds.csv", index=False)
    default_tags = tag_lots(stats)
    first = lots[~lots.index.isin(default_tags.index[default_tags != "first"])]
    exc = excursion_sweep(first)
    exc.to_csv(out / "excursion_thresholds.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(records.round(3).to_string(index=False))
        print()
        print(exc.round(3).to_string(index=False))
        # How sharply separated are the pairs the rules act on?
        s = stats[stats["fails_prev"] > 0]
        print("\nshared-slot share of consecutive lots:",
              np.histogram(s["shared"], bins=[0, 0.2, 0.4, 0.6, 0.8, 0.999, 1.01])[0].tolist(),
              "for bins 0-0.2, -0.4, -0.6, -0.8, -<1, 1")


if __name__ == "__main__":
    main()
