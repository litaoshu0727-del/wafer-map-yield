"""Generate synthetic wafer maps in the WM-811K schema.

This exists so the whole pipeline can be run and tested before the real data
is downloaded. Results on synthetic data say nothing about real performance.

    python -m wafermap.synthetic --out data/synthetic.pkl --n-lots 80
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Class mix for ordinary lots: mostly pattern-free, like the real data.
PRIOR = {
    "none": 0.70, "Edge-Ring": 0.07, "Edge-Loc": 0.05, "Center": 0.045,
    "Loc": 0.04, "Scratch": 0.03, "Random": 0.025, "Donut": 0.02, "Near-full": 0.02,
}
EXCURSION_LOT_RATE = 0.25    # share of lots with one systematic problem
EXCURSION_WAFER_RATE = 0.60  # share of wafers in such a lot that show it


def _grid(n: int):
    c = (n - 1) / 2
    yy, xx = np.mgrid[0:n, 0:n]
    x, y = (xx - c) / (n / 2), (yy - c) / (n / 2)
    return np.hypot(x, y), np.arctan2(y, x), x, y


def make_wafer(pattern: str, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    n = n or int(rng.integers(26, 46))
    r, theta, x, y = _grid(n)
    p = np.full((n, n), rng.uniform(0.01, 0.06))  # background random fails
    hi = rng.uniform(0.6, 0.95)                   # fail probability inside the pattern

    if pattern == "Center":
        p[r < rng.uniform(0.2, 0.35)] = hi
    elif pattern == "Donut":
        inner = rng.uniform(0.3, 0.45)
        p[(r > inner) & (r < inner + rng.uniform(0.15, 0.25))] = hi
    elif pattern == "Edge-Ring":
        p[r > rng.uniform(0.85, 0.92)] = hi
    elif pattern == "Edge-Loc":
        delta = np.angle(np.exp(1j * (theta - rng.uniform(-np.pi, np.pi))))
        p[(r > 0.78) & (np.abs(delta) < rng.uniform(0.3, 0.6))] = hi
    elif pattern == "Loc":
        rc, a = rng.uniform(0.2, 0.6), rng.uniform(-np.pi, np.pi)
        p[np.hypot(x - rc * np.cos(a), y - rc * np.sin(a)) < rng.uniform(0.12, 0.22)] = hi
    elif pattern == "Near-full":
        p[:] = rng.uniform(0.8, 0.97)
    elif pattern == "Random":
        p[:] = rng.uniform(0.2, 0.4)
    elif pattern == "Scratch":
        cx, cy = rng.uniform(-0.3, 0.3, size=2)
        a, half = rng.uniform(0, np.pi), rng.uniform(0.3, 0.6)
        vx, vy = 2 * half * np.cos(a), 2 * half * np.sin(a)
        x0, y0 = cx - vx / 2, cy - vy / 2
        t = np.clip(((x - x0) * vx + (y - y0) * vy) / (vx**2 + vy**2), 0, 1)
        dist = np.hypot(x - (x0 + t * vx), y - (y0 + t * vy))
        p[dist < max(0.04, 1.2 / n)] = hi
    elif pattern != "none":
        raise ValueError(f"unknown pattern: {pattern}")

    wmap = np.where(rng.random((n, n)) < p, 2, 1).astype(np.uint8)
    wmap[r > 1.0] = 0
    return wmap


def make_dataset(n_lots: int = 80, wafers_per_lot: int = 25, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names, probs = list(PRIOR), np.array(list(PRIOR.values()))
    defect_names = [c for c in names if c != "none"]
    defect_probs = np.array([PRIOR[c] for c in defect_names])
    defect_probs /= defect_probs.sum()

    rows = []
    for lot in range(1, n_lots + 1):
        n = int(rng.integers(26, 46))  # one product (die grid) per lot
        excursion = rng.choice(defect_names, p=defect_probs) if rng.random() < EXCURSION_LOT_RATE else None
        for slot in range(1, wafers_per_lot + 1):
            if excursion and rng.random() < EXCURSION_WAFER_RATE:
                pattern = excursion
            else:
                pattern = rng.choice(names, p=probs)
            wmap = make_wafer(str(pattern), rng, n)
            rows.append({
                "waferMap": wmap,
                "dieSize": int((wmap > 0).sum()),
                "lotName": f"lot{lot}",
                "waferIndex": slot,
                "trainTestLabel": "Training" if rng.random() < 0.8 else "Test",
                "failureType": str(pattern),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/synthetic.pkl")
    ap.add_argument("--n-lots", type=int, default=80)
    ap.add_argument("--wafers-per-lot", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = make_dataset(args.n_lots, args.wafers_per_lot, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)
    print(f"wrote {len(df)} wafers in {args.n_lots} lots to {out}")
    print(df["failureType"].value_counts().to_string())


if __name__ == "__main__":
    main()
