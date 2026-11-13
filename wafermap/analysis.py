"""Lot-level yield analysis built on the per-wafer features."""
from __future__ import annotations

import pandas as pd


def loss_breakdown(feats: pd.DataFrame) -> dict[str, float]:
    """Split overall yield loss into clustered (systematic) and isolated (random) fails."""
    n_die = feats["n_die"].sum()
    clustered = feats["n_fail_clustered"].sum()
    isolated = feats["n_fail"].sum() - clustered
    return {
        "yield": float(1 - feats["n_fail"].sum() / n_die),
        "systematic_loss": float(clustered / n_die),
        "random_loss": float(isolated / n_die),
    }


def pattern_pareto(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Which defect patterns cost the most failing dies (labeled wafers only)."""
    d = feats[["n_die", "n_fail", "yield"]].assign(pattern=df["failureType"].to_numpy())
    d = d.dropna(subset=["pattern"])
    out = d.groupby("pattern").agg(
        n_wafers=("n_fail", "size"),
        fail_dies=("n_fail", "sum"),
        mean_yield=("yield", "mean"),
    ).sort_values("fail_dies", ascending=False)
    out["loss_share"] = out["fail_dies"] / out["fail_dies"].sum()
    out["cum_share"] = out["loss_share"].cumsum()
    return out


def lot_summary(df: pd.DataFrame, feats: pd.DataFrame,
                min_share: float = 0.3, min_wafers: int = 5) -> pd.DataFrame:
    """Per-lot yield and the pattern its wafers have in common.

    A lot is flagged as an excursion when at least `min_share` of its wafers
    show the same defect pattern: a shared signature points at a shared cause
    (a tool or step the whole lot went through) rather than wafer-level noise.
    """
    d = feats[["yield"]].assign(lot=df["lotName"].to_numpy(), pattern=df["failureType"].to_numpy())
    rows = []
    for lot, g in d.groupby("lot", sort=False):
        counts = g["pattern"][g["pattern"].notna() & (g["pattern"] != "none")].value_counts()
        rows.append({
            "lot": lot,
            "n_wafers": len(g),
            "mean_yield": g["yield"].mean(),
            "min_yield": g["yield"].min(),
            "dominant_pattern": counts.index[0] if len(counts) else None,
            "pattern_share": counts.iloc[0] / len(g) if len(counts) else 0.0,
        })
    out = pd.DataFrame(rows).set_index("lot")
    out["excursion"] = (out["n_wafers"] >= min_wafers) & (out["pattern_share"] >= min_share)
    return out


def slot_yield(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Mean yield by wafer slot: a slot-dependent dip hints at a handling or batch-tool effect."""
    d = feats[["yield"]].assign(slot=df["waferIndex"].to_numpy())
    return d.groupby("slot")["yield"].agg(["mean", "count"])
