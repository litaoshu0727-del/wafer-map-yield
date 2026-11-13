"""Load wafer map datasets into one schema.

Schema (one row per wafer):
    waferMap        2-D uint8 array: 0 = no die, 1 = pass, 2 = fail
    dieSize         number of dies on the wafer
    lotName         lot identifier
    waferIndex      slot number inside the lot
    trainTestLabel  "Training" / "Test" / None
    failureType     one of CLASSES, or None when unlabeled

Works for the real WM-811K file (LSWMD.pkl) and for files written by
`wafermap.synthetic`.
"""
from __future__ import annotations

import copyreg
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
    "Near-full", "Random", "Scratch", "none",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def _flatten(x):
    """WM-811K stores labels as nested arrays: [['Center']] or [] when unlabeled."""
    if x is None or isinstance(x, str):
        return x
    if isinstance(x, float) and np.isnan(x):
        return None
    arr = np.asarray(x).ravel()
    return str(arr[0]) if arr.size else None


class _Raw:
    """Stand-in for an old pandas class: keeps the pickled state instead of rebuilding it."""

    def __setstate__(self, state):
        self.state = state


def _raw_index(cls, fields):
    return fields


class _LegacyUnpickler(pickle.Unpickler):
    """Reads LSWMD.pkl, a Python 2 / pandas 0.x pickle that current pandas cannot load.

    pandas objects come back as raw state, which `_load_legacy` reassembles.
    Only numpy and a few builtins are allowed to be constructed; anything else
    in the file is refused.
    """

    _BUILTINS = {
        ("copy_reg", "_reconstructor"): copyreg._reconstructor,
        ("__builtin__", "object"): object,
        ("__builtin__", "slice"): slice,
    }

    def find_class(self, module, name):
        if module.startswith("pandas."):
            return _raw_index if name == "_new_Index" else _Raw
        if module == "numpy" or module.startswith("numpy."):
            return super().find_class(module.replace("numpy.core", "numpy._core"), name)
        if (module, name) in self._BUILTINS:
            return self._BUILTINS[(module, name)]
        raise pickle.UnpicklingError(f"unexpected object in pickle: {module}.{name}")


def _load_legacy(path: str | Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        frame = _LegacyUnpickler(f, encoding="latin1").load()
    manager = frame.state["_data"].state[3]["0.14.1"]
    columns = [str(c) for c in manager["axes"][0]["data"]]
    data = {}
    for block in manager["blocks"]:
        positions = np.arange(len(columns))[block["mgr_locs"]]
        for row, pos in zip(block["values"], np.atleast_1d(positions)):
            data[columns[pos]] = list(row) if row.dtype == object else row
    return pd.DataFrame({c: data[c] for c in columns})


def load_dataset(path: str | Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        header = f.read(2)
    legacy = header[:1] != b"\x80" or header[1] < 3  # pickle protocol 2 or older
    df = _load_legacy(path) if legacy else pd.read_pickle(path)
    df = df.rename(columns={"trianTestLabel": "trainTestLabel"})  # typo in the original
    for col in ("failureType", "trainTestLabel"):
        df[col] = df[col].map(_flatten).astype(object)
    df["waferIndex"] = df["waferIndex"].astype(int)
    df["waferMap"] = df["waferMap"].map(lambda m: np.asarray(m, dtype=np.uint8))
    return df.reset_index(drop=True)


def labeled(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["failureType"].isin(CLASSES)].reset_index(drop=True)


def load_labeled(path: str | Path, all_records: bool = False) -> pd.DataFrame:
    """Labeled wafers ready for analysis.

    By default, lots that are a repeat or retest record of the previous lot's
    wafers are dropped (see `wafermap.records`), so each physical wafer is
    counted once. The tagging runs on the full dataset before the unlabeled
    wafers are removed, because it compares neighbouring lots.
    """
    from .records import classify_lots

    df = load_dataset(path)
    if not all_records:
        record = classify_lots(df)
        kept = labeled(df[record == "first"])
        dropped = len(labeled(df)) - len(kept)
        print(f"dropped {dropped} labeled wafers in repeat/retest records "
              f"({int((record != 'first').sum())} wafers in such records overall)")
        return kept
    return labeled(df)


def resize(wmap: np.ndarray, size: int = 64) -> np.ndarray:
    """Nearest-neighbour resize, so the values stay exactly {0, 1, 2}."""
    h, w = wmap.shape
    rows = ((np.arange(size) + 0.5) * h / size).astype(int)
    cols = ((np.arange(size) + 0.5) * w / size).astype(int)
    return wmap[np.ix_(rows, cols)]


def to_arrays(df: pd.DataFrame, size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Stack wafers into X (N, size, size) uint8 and y (N,) class indices."""
    X = np.stack([resize(m, size) for m in df["waferMap"]])
    y = np.array(df["failureType"].map(CLASS_TO_IDX), dtype=np.int64)  # writable copy
    return X, y
