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


def lot_number(lot_name: str) -> int:
    digits = re.sub(r"\D", "", lot_name)
    return int(digits) if digits else 0


def relation(prev: dict[int, tuple], cur: dict[int, tuple]) -> str:
    """How the lot `cur` relates to the lot numbered just before it: 'first', 'repeat' or 'retest'.

    Each lot is {slot: (die mask, fail mask)}. Only slots present in both lots
    with an identical die layout are compared.

    - 'repeat': almost the same fail map, usually with a few more fails. It
      reads as a later, cumulative snapshot of the same wafers.
    - 'retest': far fewer fails, and either none of them on a die that already
      failed, or all of them. It reads as a record of only the dies that
      changed at a later step.
    """
    shared = [s for s in cur if s in prev and prev[s][0].shape == cur[s][0].shape
              and np.array_equal(prev[s][0], cur[s][0])]
    if len(shared) < MIN_SHARED * min(len(prev), len(cur)):
        return "first"
    n_die = sum(int(cur[s][0].sum()) for s in shared)
    fails_prev = sum(int(prev[s][1].sum()) for s in shared)
    fails_cur = sum(int(cur[s][1].sum()) for s in shared)
    overlap = sum(int((prev[s][1] & cur[s][1]).sum()) for s in shared)
    small, large = min(fails_prev, fails_cur), max(fails_prev, fails_cur)
    if fails_prev == 0:
        return "first"
    # One map contains the other. Rule out wafers so full of fails that this happens by chance.
    nested = small > 0 and overlap >= NESTED * small and large < 0.5 * n_die
    if nested and small >= SAME_SIZE * large:
        return "repeat"
    if fails_cur < FEW_FAILS * fails_prev and (overlap == 0 or nested):
        return "retest"
    return "first"


def classify_lots(df: pd.DataFrame) -> pd.Series:
    """Tag every wafer's lot as 'first', 'repeat' or 'retest' (a Series aligned with `df`).

    Needs every wafer of the dataset, labeled or not, with the rows of a lot
    contiguous and lots in lot-number order, as in WM-811K.
    """
    tags: dict[str, str] = {}
    prev_seq, prev, cur_lot, cur = None, {}, None, {}

    def close() -> None:
        nonlocal prev_seq, prev
        if cur_lot is None:
            return
        seq = lot_number(cur_lot)
        tags[cur_lot] = relation(prev, cur) if prev_seq is not None and seq == prev_seq + 1 else "first"
        prev_seq, prev = seq, cur

    for wmap, lot, slot in zip(df["waferMap"], df["lotName"], df["waferIndex"]):
        if lot != cur_lot:
            close()
            cur_lot, cur = lot, {}
        cur[int(slot)] = (wmap > 0, wmap == 2)
    close()
    return df["lotName"].map(tags).rename("record")
