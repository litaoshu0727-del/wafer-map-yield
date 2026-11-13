"""Per-wafer spatial features: where on the wafer the yield loss sits and how clustered it is."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

CENTER_R = 0.4       # normalised radius below which a die counts as "center"
EDGE_R = 0.8         # ... and above which it counts as "edge"
N_RADIAL = 5         # rings in the radial fail-rate profile
N_SECTORS = 8        # angular sectors
MIN_CLUSTER = 5      # connected fails needed to call a group systematic
_EIGHT_CONNECTED = np.ones((3, 3), dtype=int)

FEATURE_COLUMNS = (
    ["yield", "fail_center", "fail_mid", "fail_edge", "edge_minus_center"]
    + [f"radial_{i}" for i in range(N_RADIAL)]
    + ["sector_max", "sector_cv", "n_clusters", "largest_cluster_frac",
       "clustered_frac", "largest_cluster_elongation", "largest_cluster_radius"]
)


def polar_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Radius (0 = wafer centre, ~1 = edge) and angle of every cell."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx - (w - 1) / 2) / (w / 2)
    y = (yy - (h - 1) / 2) / (h / 2)
    return np.hypot(x, y), np.arctan2(y, x)


def _rate(fail: np.ndarray, die: np.ndarray, region: np.ndarray) -> float:
    n = (die & region).sum()
    return float((fail & region).sum() / n) if n else 0.0


def wafer_features(wmap: np.ndarray) -> dict[str, float]:
    die, fail = wmap > 0, wmap == 2
    n_die, n_fail = int(die.sum()), int(fail.sum())
    r, theta = polar_grid(wmap.shape)

    out: dict[str, float] = {
        "n_die": n_die,
        "n_fail": n_fail,
        "yield": 1 - n_fail / n_die if n_die else np.nan,
        "fail_center": _rate(fail, die, r < CENTER_R),
        "fail_mid": _rate(fail, die, (r >= CENTER_R) & (r < EDGE_R)),
        "fail_edge": _rate(fail, die, r >= EDGE_R),
    }
    out["edge_minus_center"] = out["fail_edge"] - out["fail_center"]

    ring = np.minimum((r * N_RADIAL).astype(int), N_RADIAL - 1)
    for i in range(N_RADIAL):
        out[f"radial_{i}"] = _rate(fail, die, ring == i)

    sector = ((theta + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS
    sector_rates = np.array([_rate(fail, die, sector == s) for s in range(N_SECTORS)])
    out["sector_max"] = float(sector_rates.max())
    out["sector_cv"] = float(sector_rates.std() / sector_rates.mean()) if sector_rates.mean() else 0.0

    # Connected groups of failing dies: isolated fails look random, groups look systematic.
    labels, n_groups = ndimage.label(fail, structure=_EIGHT_CONNECTED)
    sizes = np.bincount(labels.ravel())[1:] if n_groups else np.array([], dtype=int)
    big = sizes[sizes >= MIN_CLUSTER]
    out["n_clusters"] = int(big.size)
    out["n_fail_clustered"] = int(big.sum())
    out["clustered_frac"] = float(big.sum() / n_fail) if n_fail else 0.0
    out["largest_cluster_frac"] = float(sizes.max() / n_fail) if n_fail else 0.0

    # Shape and position of the largest group: long and thin suggests a scratch.
    out["largest_cluster_elongation"] = 1.0
    out["largest_cluster_radius"] = 0.0
    if sizes.size and sizes.max() >= 3:
        ys, xs = np.nonzero(labels == sizes.argmax() + 1)
        eig = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys])))
        out["largest_cluster_elongation"] = float(np.sqrt(eig[1] / max(eig[0], 1e-6)))
        out["largest_cluster_radius"] = float(r[ys, xs].mean())
    return out


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row of features per wafer, indexed like `df`."""
    return pd.DataFrame([wafer_features(m) for m in df["waferMap"]], index=df.index)
