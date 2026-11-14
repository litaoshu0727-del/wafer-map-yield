"""Recognise lots that are another record of the previous lot's wafers.

WM-811K does not document it, but consecutive lot numbers often hold the same
wafers recorded again at a later test step: the slots match, the die layouts
are identical, and the fail maps are related in one of two regular ways.
Treating those records as separate lots counts wafers twice, and can put the
same wafer on both sides of a train/test split. The tags below are inferences
from the maps, not documented facts.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

MIN_SHARED = 0.8  # share of slots the two lots must have in common, with identical die layouts
NESTED = 0.9      # share of the smaller fail map that must lie inside the larger one
SAME_SIZE = 0.8   # 'repeat': the two fail maps are at least this close in size
FEW_FAILS = 0.2   # 'retest': under this fraction of the previous lot's fails
DEFAULTS = {"min_shared": MIN_SHARED, "nested": NESTED, "same_size": SAME_SIZE, "few_fails": FEW_FAILS}


def lot_number(lot_name: str) -> int:
    digits = re.sub(r"\D", "", lot_name)
    return int(digits) if digits else 0


def pair_stats(prev: dict[int, tuple], cur: dict[int, tuple]) -> dict[str, float]:
    """The numbers the tagging rules look at, for one lot and the lot numbered just before it.

    Each lot is {slot: (die mask, fail mask)}. Only slots present in both lots
    with an identical die layout are compared; `shared` is their share of the
    smaller lot. Kept separate from the decision so that the thresholds can be
    varied without going through the wafer maps again.
    """
    shared = [s for s in cur if s in prev and prev[s][0].shape == cur[s][0].shape
              and np.array_equal(prev[s][0], cur[s][0])]
    return {"shared": len(shared) / min(len(prev), len(cur)),
            "n_die": sum(int(cur[s][0].sum()) for s in shared),
            "fails_prev": sum(int(prev[s][1].sum()) for s in shared),
            "fails_cur": sum(int(cur[s][1].sum()) for s in shared),
            "overlap": sum(int((prev[s][1] & cur[s][1]).sum()) for s in shared)}


def decide(stats: dict[str, float], min_shared: float = MIN_SHARED, nested: float = NESTED,
           same_size: float = SAME_SIZE, few_fails: float = FEW_FAILS) -> str:
    """'first', 'repeat' or 'retest' from the pair statistics.

    - 'repeat': almost the same fail map, usually with a few more fails. It
      reads as a later, cumulative snapshot of the same wafers.
    - 'retest': far fewer fails, and either none of them on a die that already
      failed, or all of them. It reads as a record of only the dies that
      changed at a later step.
    """
    if stats["shared"] < min_shared or stats["fails_prev"] == 0:
        return "first"
    fails_prev, fails_cur, overlap = stats["fails_prev"], stats["fails_cur"], stats["overlap"]
    small, large = min(fails_prev, fails_cur), max(fails_prev, fails_cur)
    # One map contains the other. Rule out wafers so full of fails that this happens by chance.
    is_nested = small > 0 and overlap >= nested * small and large < 0.5 * stats["n_die"]
    if is_nested and small >= same_size * large:
        return "repeat"
    if fails_cur < few_fails * fails_prev and (overlap == 0 or is_nested):
        return "retest"
    return "first"


def relation(prev: dict[int, tuple], cur: dict[int, tuple], **thresholds) -> str:
    """How the lot `cur` relates to the lot numbered just before it: 'first', 'repeat' or 'retest'."""
    return decide(pair_stats(prev, cur), **thresholds)


def lot_pair_stats(df: pd.DataFrame) -> pd.DataFrame:
    """`pair_stats` for every lot whose number directly follows the previous lot's (index: lot name).

    Needs every wafer of the dataset, labeled or not, with the rows of a lot
    contiguous and lots in lot-number order, as in WM-811K.
    """
    rows: dict[str, dict] = {}
    prev_seq, prev, cur_lot, cur = None, {}, None, {}

    def close() -> None:
        nonlocal prev_seq, prev
        if cur_lot is None:
            return
        seq = lot_number(cur_lot)
        if prev_seq is not None and seq == prev_seq + 1:
            rows[cur_lot] = pair_stats(prev, cur)
        prev_seq, prev = seq, cur

    for wmap, lot, slot in zip(df["waferMap"], df["lotName"], df["waferIndex"]):
        if lot != cur_lot:
            close()
            cur_lot, cur = lot, {}
        cur[int(slot)] = (wmap > 0, wmap == 2)
    close()
    return pd.DataFrame.from_dict(rows, orient="index")


def tag_lots(stats: pd.DataFrame, **thresholds) -> pd.Series:
    """'repeat' / 'retest' / 'first' per lot in `stats`; lots not in `stats` are first records."""
    return pd.Series([decide(row, **thresholds) for row in stats.to_dict("records")], index=stats.index,
                     dtype=object)


def classify_lots(df: pd.DataFrame, **thresholds) -> pd.Series:
    """Tag every wafer's lot as 'first', 'repeat' or 'retest' (a Series aligned with `df`)."""
    tags = tag_lots(lot_pair_stats(df), **thresholds)
    return df["lotName"].map(tags).fillna("first").rename("record")
