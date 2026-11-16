"""A statistical test for "are the failing dies on this wafer clustered?".

    python -m wafermap.spatial --data data/wm811k/LSWMD.pkl --out reports/wm811k_spatial

`features.py` calls a fail systematic when it sits in a connected group of
five or more. That is a rule with a hand-picked size and no error rate:
on a wafer with many fails, groups of five form by chance.

The join-count statistic gives the same question a null hypothesis. Count
the pairs of neighbouring dies (sharing an edge) that both fail. If the
wafer's k fails were dropped on its n dies at random, the expected count
and its variance are known in closed form (Cliff & Ord, sampling without
replacement), so each wafer gets a z-score: how many more fail-fail
neighbours it has than chance would give, in standard deviations.

"At random" here means every die is equally likely to fail. A wafer whose
fail rate simply rises towards the edge is not random in that sense and
will be flagged; `radial_z` repeats the test with the fails shuffled only
inside rings of equal radius, which asks for clustering beyond the radial
profile.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .data import CLASSES, load_dataset
from .features import MIN_CLUSTER, _EIGHT_CONNECTED, polar_grid

Z_CRIT = 1.645  # one-sided 5%
N_RINGS = 8


def _joins(mask: np.ndarray) -> int:
    """Pairs of horizontally or vertically adjacent cells that are both set."""
    return int((mask[:, 1:] & mask[:, :-1]).sum() + (mask[1:] & mask[:-1]).sum())


def join_count(wmap: np.ndarray) -> dict[str, float]:
    """Fail-fail neighbour pairs on one wafer against random placement of the same number of fails."""
    die, fail = wmap > 0, wmap == 2
    n, k = int(die.sum()), int(fail.sum())
    total, both = _joins(die), _joins(fail)
    out = {"n_die": n, "n_fail": k, "joins": total, "fail_joins": both, "expected": 0.0, "z": 0.0}
    if k < 2 or n < 4 or total == 0:
        return out
    # Neighbours of each die, to count pairs of joins that share a die.
    degree = np.zeros(die.shape, dtype=np.int64)
    side = die[:, 1:] & die[:, :-1]
    degree[:, 1:] += side
    degree[:, :-1] += side
    above = die[1:] & die[:-1]
    degree[1:] += above
    degree[:-1] += above
    sharing = int((degree * (degree - 1)).sum()) // 2
    p2 = k * (k - 1) / (n * (n - 1))
    p3 = p2 * (k - 2) / (n - 2)
    p4 = p3 * (k - 3) / (n - 3)
    expected = total * p2
    var = total * p2 + 2 * sharing * p3 + (total * (total - 1) - 2 * sharing) * p4 - expected ** 2
    out["expected"] = expected
    out["z"] = float((both - expected) / np.sqrt(var)) if var > 1e-12 else 0.0
    return out


def radial_z(wmap: np.ndarray, rng: np.random.Generator, n_shuffles: int = 200) -> float:
    """z of the fail-fail neighbour count against shuffles that keep each ring's number of fails."""
    die, fail = wmap > 0, wmap == 2
    if fail.sum() < 2:
        return 0.0
    r, _ = polar_grid(wmap.shape)
    ring = np.minimum((r * N_RINGS).astype(int), N_RINGS - 1)[die]
    order = np.argsort(ring, kind="stable")
    bounds = np.flatnonzero(np.diff(ring[order])) + 1
    flat = fail[die][order]
    rows, cols = np.nonzero(die)
    rows, cols = rows[order], cols[order]
    null = np.empty(n_shuffles)
    grid = np.zeros(die.shape, dtype=bool)
    for i in range(n_shuffles):
        shuffled = np.concatenate([rng.permutation(part) for part in np.split(flat, bounds)])
        grid[:] = False
        grid[rows[shuffled], cols[shuffled]] = True
        null[i] = _joins(grid)
    sd = null.std()
    return float((_joins(fail) - null.mean()) / sd) if sd > 0 else 0.0


def clustered_fails(wmap: np.ndarray) -> int:
    """Fails in connected groups of `MIN_CLUSTER` or more: the rule used in `features.py`."""
    labels, n_groups = ndimage.label(wmap == 2, structure=_EIGHT_CONNECTED)
    if not n_groups:
        return 0
    sizes = np.bincount(labels.ravel())[1:]
    return int(sizes[sizes >= MIN_CLUSTER].sum())


def wafer_table(df: pd.DataFrame) -> pd.DataFrame:
    """Join-count result and the connected-group count for every wafer in `df`."""
    rows = [{**join_count(m), "n_fail_clustered": clustered_fails(m)} for m in df["waferMap"]]
    out = pd.DataFrame(rows, index=df.index)
    out["lot"] = df["lotName"].to_numpy()
    out["label"] = df["failureType"].where(df["failureType"].isin(CLASSES)).to_numpy()
    out["excess_joins"] = out["fail_joins"] - out["expected"]
    return out


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Chance that a random positive scores above a random negative (ties count half)."""
    rank = pd.Series(score).rank().to_numpy()
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((rank[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def by_label(table: pd.DataFrame) -> pd.DataFrame:
    """Per labeled class: how often the test fires and how the group rule compares."""
    lab = table.dropna(subset=["label"])
    rows = []
    for name in CLASSES:
        g = lab[lab["label"] == name]
        if len(g):
            rows.append({"label": name, "wafers": len(g), "median_z": g["z"].median(),
                         "flagged": (g["z"] > Z_CRIT).mean(), "z_above_5": (g["z"] > 5).mean(),
                         "has_group": (g["n_fail_clustered"] > 0).mean(),
                         "fails_in_groups": g["n_fail_clustered"].sum() / max(g["n_fail"].sum(), 1)})
    return pd.DataFrame(rows)


def loss_split(table: pd.DataFrame) -> dict[str, float]:
    """Yield loss by wafer type, and the group rule's split for comparison."""
    n_die = table["n_die"].sum()
    sig = table["z"] > Z_CRIT
    return {"wafers": len(table), "yield_loss": table["n_fail"].sum() / n_die,
            "wafers_flagged": sig.mean(),
            "loss_on_flagged_wafers": table.loc[sig, "n_fail"].sum() / n_die,
            "loss_on_other_wafers": table.loc[~sig, "n_fail"].sum() / n_die,
            "loss_in_groups": table["n_fail_clustered"].sum() / n_die,
            "loss_in_groups_on_unflagged_wafers": table.loc[~sig, "n_fail_clustered"].sum() / n_die}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="reports/spatial")
    ap.add_argument("--radial-sample", type=int, default=3000,
                    help="labeled wafers per class (at most) for the ring-shuffle test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from .records import classify_lots
    df = load_dataset(args.data)
    df = df[classify_lots(df) == "first"].reset_index(drop=True)
    table = wafer_table(df)
    table.to_csv(out / "join_counts.csv", index=False)

    lab = table.dropna(subset=["label"])
    pattern = (lab["label"] != "none").to_numpy()
    summary = {"auc_z": auc(lab["z"].to_numpy(), pattern),
               "auc_group_share": auc((lab["n_fail_clustered"] / lab["n_fail"].clip(lower=1)).to_numpy(), pattern),
               "auc_fail_rate": auc((lab["n_fail"] / lab["n_die"]).to_numpy(), pattern)}
    classes = by_label(table)

    rng = np.random.default_rng(args.seed)
    sample = pd.concat([g.sample(min(len(g), args.radial_sample), random_state=args.seed)
                        for _, g in lab.groupby("label")])
    radial = sample[["label", "z"]].copy()
    radial["z_radial"] = [radial_z(df.at[i, "waferMap"], rng) for i in sample.index]
    radial.to_csv(out / "radial_sample.csv")
    ring = radial.groupby("label").agg(wafers=("z", "size"), flagged=("z", lambda s: (s > Z_CRIT).mean()),
                                       flagged_radial=("z_radial", lambda s: (s > Z_CRIT).mean()))
    classes = classes.merge(ring[["flagged_radial"]], left_on="label", right_index=True)
    classes.to_csv(out / "by_label.csv", index=False)
    summary["auc_z_sample"] = auc(radial["z"].to_numpy(), (radial["label"] != "none").to_numpy())
    summary["auc_z_radial_sample"] = auc(radial["z_radial"].to_numpy(), (radial["label"] != "none").to_numpy())

    splits = pd.DataFrame({"all first-test wafers": loss_split(table), "labeled": loss_split(lab),
                           "unlabeled": loss_split(table[table["label"].isna()])}).T
    splits.to_csv(out / "loss_split.csv")
    pd.Series(summary).to_csv(out / "summary.csv", header=["value"])

    from .plots import spatial_figure
    spatial_figure(lab, out / "spatial_z.png")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(classes.round(3).to_string(index=False))
        print()
        print(splits.round(4).to_string())
        print()
        print({k: round(v, 3) for k, v in summary.items()})


if __name__ == "__main__":
    main()
