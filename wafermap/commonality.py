"""Tool commonality analysis, tested on real lot outcomes with synthetic tool routes.

    python -m wafermap.commonality --data data/wm811k/LSWMD.pkl --out reports/wm811k_commonality

Commonality analysis asks: did the bad lots pass through one tool more often
than chance would explain? WM-811K has no tool history, so it cannot be done
on the real data. What can be checked is the method itself:

- the lot outcomes are real (which first-test lots are excursions, in file
  order, which is taken as time order);
- the tool routes are SYNTHETIC: every lot is given a tool at each of a few
  process steps by a random rule;
- a "guilty" tool is planted by sending a share of the real excursion lots
  through it. Because the answer is known, false alarms and misses can be
  counted.

Nothing here says anything about which tools caused the excursions in
WM-811K. The result is how the test behaves on outcomes distributed like
real ones: how often it accuses an innocent tool, and how strong a cause
has to be before it is found.

Three tests are compared:

- `fisher_scan`: one-sided Fisher exact test per tool, Benjamini-Hochberg
  correction over all tools. It assumes lots are independent;
- `shift_scan`, `p_shift`: the same statistic, but its null distribution
  comes from sliding the outcome sequence against the routes (circular
  shifts), with the largest statistic over all tools taken per shift. This
  keeps the runs in both sequences, so excursions that come in bursts and
  tools that are used in campaigns do not count as evidence;
- `shift_scan`, `p_local`: calibrated the same way, but the statistic only
  compares bad and good lots inside the same short stretch of time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from .data import load_dataset
from .records import lot_pair_stats, tag_lots
from .sensitivity import lot_table

N_STEPS, N_TOOLS = 10, 4
ALPHA = 0.05
WINDOW = 20  # lots per stretch for the time-stratified test


def excursion_flags(lots: pd.DataFrame, min_share: float = 0.3, min_wafers: int = 5) -> pd.Series:
    """True/False per eligible lot (enough labeled wafers), in the order of `lots`."""
    eligible = lots[lots["n_labeled"] >= min_wafers]
    return (eligible["top_count"] >= min_share * eligible["n_labeled"]).rename("excursion")


def make_routes(n_lots: int, rng: np.random.Generator, n_steps: int = N_STEPS, n_tools: int = N_TOOLS,
                run_length: float = 1.0) -> np.ndarray:
    """Synthetic tool per lot and step, shape (n_lots, n_steps).

    `run_length` is the mean number of consecutive lots a step keeps on the
    same tool (1 = every lot picks independently).
    """
    routes = np.empty((n_lots, n_steps), dtype=np.int64)
    for s in range(n_steps):
        switch = rng.random(n_lots) < 1.0 / run_length
        switch[0] = True
        start = np.maximum.accumulate(np.where(switch, np.arange(n_lots), 0))
        routes[:, s] = rng.integers(n_tools, size=n_lots)[start]
    return routes


def couple(routes: np.ndarray, coupling: float, rng: np.random.Generator) -> np.ndarray:
    """Dedicated paths: with chance `coupling` a lot's tool at step 1 has the same number as at step 0."""
    routes = routes.copy()
    if coupling > 0:
        tied = rng.random(len(routes)) < coupling
        routes[tied, 1] = routes[tied, 0]
    return routes


def plant(routes: np.ndarray, bad: np.ndarray, step: int, tool: int, strength: float,
          rng: np.random.Generator) -> np.ndarray:
    """Send a share `strength` of the bad lots through `tool` at `step` (copy of the routes)."""
    routes = routes.copy()
    hit = bad & (rng.random(len(bad)) < strength)
    routes[hit, step] = tool
    return routes


def _one_hot(routes: np.ndarray, n_tools: int) -> np.ndarray:
    """(n_lots, n_steps * n_tools) indicator of the lot having used each tool."""
    n_lots, n_steps = routes.shape
    out = np.zeros((n_lots, n_steps * n_tools), dtype=np.float32)
    out[np.arange(n_lots)[:, None], routes + n_tools * np.arange(n_steps)] = 1
    return out


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Adjusted p-values (q-values) controlling the false discovery rate."""
    order = np.argsort(p)
    scaled = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.empty_like(p)
    q[order] = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    return q


def fisher_scan(routes: np.ndarray, bad: np.ndarray, n_tools: int = N_TOOLS) -> pd.DataFrame:
    """One row per (step, tool): lots through it, bad lots among them, and the one-sided test."""
    hot = _one_hot(routes, n_tools)
    through = hot.sum(0)
    bad_through = hot[bad].sum(0)
    n, k = len(bad), int(bad.sum())
    p = hypergeom.sf(bad_through - 1, n, k, through)
    sd = np.sqrt(through * (k / n) * (1 - k / n) * (n - through) / max(n - 1, 1))
    table = pd.DataFrame({"step": np.repeat(np.arange(routes.shape[1]), n_tools),
                          "tool": np.tile(np.arange(n_tools), routes.shape[1]),
                          "lots": through.astype(int), "bad_lots": bad_through.astype(int),
                          "expected_bad": through * k / n,
                          "z": (bad_through - through * k / n) / np.where(sd > 0, sd, np.inf), "p": p})
    table["q"] = benjamini_hochberg(p)
    return table


def _local_z(outcomes: np.ndarray, hot: np.ndarray, window: int) -> np.ndarray:
    """z per tool comparing bad and good lots only inside the same stretch of `window` lots.

    `outcomes` is (n_sequences, n_lots). Within each stretch the expected
    number of bad lots through a tool uses that stretch's own bad rate
    (the Cochran-Mantel-Haenszel statistic), so a stretch where everything
    is bad says nothing about any tool.
    """
    n = hot.shape[0]
    ids = np.arange(n) // window
    member = np.zeros((ids[-1] + 1, n), dtype=np.float32)
    member[ids, np.arange(n)] = 1
    size = member.sum(1)
    through = member @ hot                       # (stretches, tools)
    n_bad = outcomes @ member.T                  # (sequences, stretches)
    excess = outcomes @ hot - (n_bad / size) @ through
    spread = n_bad * (size - n_bad) / (size ** 2 * np.maximum(size - 1, 1))
    var = spread @ (through * (size[:, None] - through))
    return excess / np.sqrt(np.where(var > 0, var, np.inf))


def shift_scan(routes: np.ndarray, bad: np.ndarray, rng: np.random.Generator, n_tools: int = N_TOOLS,
               n_shifts: int = 499, window: int = WINDOW, min_shift: int | None = None) -> pd.DataFrame:
    """`fisher_scan` plus two p-values calibrated by circular shifts.

    Shifting the outcomes against the routes breaks any link between a lot's
    outcome and its tools but keeps how both sequences cluster in time. Each
    p-value is the share of shifts whose largest z over all tools reaches
    this tool's z, which also corrects for testing many tools at once.

    - `p_shift` uses the plain z of `fisher_scan`;
    - `p_local` uses `_local_z`, which compares lots only with their
      neighbours in time (`z_local`).
    """
    table = fisher_scan(routes, bad, n_tools)
    n = len(bad)
    hot = _one_hot(routes, n_tools)
    through = hot.sum(0)
    rate = bad.mean()
    sd = np.sqrt(through * rate * (1 - rate) * (n - through) / max(n - 1, 1))
    sd = np.where(sd > 0, sd, np.inf)
    if min_shift is None:
        min_shift = max(1, n // 20)
    shifts = rng.integers(min_shift, n - min_shift + 1, size=n_shifts)
    rolled = np.stack([np.roll(bad, s) for s in shifts]).astype(np.float32)
    null_max = (((rolled @ hot) - through * rate) / sd).max(1)
    table["p_shift"] = [(1 + (null_max >= z).sum()) / (1 + n_shifts) for z in table["z"]]
    table["z_local"] = _local_z(bad[None].astype(np.float32), hot, window)[0]
    null_local = _local_z(rolled, hot, window).max(1)
    table["p_local"] = [(1 + (null_local >= z).sum()) / (1 + n_shifts) for z in table["z_local"]]
    return table


METHODS = {"fisher": "q", "shift": "p_shift", "local": "p_local"}


def flagged(table: pd.DataFrame, method: str) -> set[tuple[int, int]]:
    """(step, tool) pairs a method accuses at the 5% level."""
    hit = table[METHODS[method]] < ALPHA
    return set(zip(table.loc[hit, "step"], table.loc[hit, "tool"]))


def bad_runs(bad: np.ndarray) -> np.ndarray:
    """Lengths of the stretches of consecutive bad lots."""
    edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
    return edges[1::2] - edges[::2]


def burstiness(bad: np.ndarray, lags=(1, 5, 20, 100)) -> dict[int, float]:
    """How much more likely a lot is bad when the lot `lag` places earlier was bad (1 = no clustering)."""
    return {lag: float(bad[lag:][bad[:-lag]].mean() / bad.mean()) for lag in lags}


def false_alarms(bad: np.ndarray, run_lengths, n_runs: int, seed: int, window: int = WINDOW) -> pd.DataFrame:
    """No tool is guilty. Share of runs in which each test accuses at least one tool."""
    rows = []
    for run_length in run_lengths:
        rng = np.random.default_rng(seed)
        hits = dict.fromkeys(METHODS, 0)
        for _ in range(n_runs):
            table = shift_scan(make_routes(len(bad), rng, run_length=run_length), bad, rng, window=window)
            for method in hits:
                hits[method] += bool(flagged(table, method))
        rows.append({"run_length": run_length, "window": window, "runs": n_runs,
                     **{f"{m}_false_alarm": hits[m] / n_runs for m in hits}})
    return pd.DataFrame(rows)


def power(bad: np.ndarray, strengths, n_lots, run_length: float, n_runs: int, seed: int,
          coupling: float = 0.0, window: int = WINDOW) -> pd.DataFrame:
    """One tool is guilty. How often each test finds it, and how often it also accuses another tool.

    `n_lots` are numbers of consecutive lots to analyse (None = all): a
    random stretch of that length is cut out of the real sequence each run.
    The guilty tool is at step 0; "partner" is the tool with the same number
    at step 1, which `coupling` ties to it.
    """
    rows = []
    for length in n_lots:
        for strength in strengths:
            rng = np.random.default_rng(seed)
            found, partner, other = (dict.fromkeys(METHODS, 0) for _ in range(3))
            n_bad = []
            for _ in range(n_runs):
                if length is None or length >= len(bad):
                    part = bad
                else:
                    start = rng.integers(0, len(bad) - length + 1)
                    part = bad[start:start + length]
                if not part.any():
                    continue
                n_bad.append(part.sum())
                routes = make_routes(len(part), rng, run_length=run_length)
                guilty = (0, int(rng.integers(N_TOOLS)))
                routes = couple(plant(routes, part, *guilty, strength, rng), coupling, rng)
                table = shift_scan(routes, part, rng, window=window)
                for method in METHODS:
                    hits = flagged(table, method)
                    found[method] += guilty in hits
                    partner[method] += (1, guilty[1]) in hits
                    other[method] += bool(hits - {guilty, (1, guilty[1])})
            runs = max(len(n_bad), 1)
            rows.append({"lots": len(part), "mean_bad_lots": float(np.mean(n_bad)), "strength": strength,
                         "run_length": run_length, "coupling": coupling, "window": window, "runs": runs,
                         **{f"{m}_found": found[m] / runs for m in METHODS},
                         **{f"{m}_partner_accused": partner[m] / runs for m in METHODS},
                         **{f"{m}_other_accused": other[m] / runs for m in METHODS}})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", help="wafer data; not needed with --lots")
    ap.add_argument("--lots", help="lots.csv written by an earlier run (skips reading the wafer maps)")
    ap.add_argument("--out", default="reports/commonality")
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.lots:
        flags = pd.read_csv(args.lots, index_col=0)["excursion"]
    else:
        df = load_dataset(args.data)
        tags = tag_lots(lot_pair_stats(df))
        lots = lot_table(df)
        flags = excursion_flags(lots[~lots.index.isin(tags.index[tags != "first"])])
        flags.to_csv(out / "lots.csv")
    bad = flags.to_numpy(dtype=bool)
    runs = np.sort(bad_runs(bad))[::-1]
    print(f"{len(bad)} eligible first-test lots, {bad.sum()} excursions ({bad.mean():.1%})")
    print(f"excursions form {len(runs)} stretches of consecutive lots; the longest are {runs[:5].tolist()}, "
          f"and stretches of 20+ lots hold {runs[runs >= 20].sum()} of them")
    print("chance of an excursion given one `lag` lots earlier, relative to the base rate:",
          {k: round(v, 1) for k, v in burstiness(bad).items()})

    strengths = [0.05, 0.1, 0.2, 0.4]
    opts = dict(n_runs=args.runs, seed=args.seed)
    null = false_alarms(bad, [1, 5, 20, 100], **opts)
    null.to_csv(out / "false_alarms.csv", index=False)
    found = pd.concat([power(bad, strengths, [250, 1000, None], run_length=r, **opts) for r in (1, 20)])
    found.to_csv(out / "power.csv", index=False)
    # Dedicated paths between two steps; independent routing so that Fisher is valid and the effect is visible.
    tied = pd.concat([power(bad, [0.2], [None], run_length=1, coupling=c, **opts) for c in (0.0, 0.5, 0.9)])
    tied.to_csv(out / "coupling.csv", index=False)
    # How the stretch length of the time-stratified test trades false alarms against detection.
    sweep = pd.concat([false_alarms(bad, [20], window=w, **opts).merge(
        power(bad, [0.2, 0.4], [None], run_length=20, window=w, **opts), on=["run_length", "window", "runs"])
        for w in (10, 20, 50, 200)])
    sweep.to_csv(out / "windows.csv", index=False)

    from .plots import commonality_figure
    commonality_figure(null, found[found["run_length"] == 20], out / "commonality.png")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        for frame in (null, found, tied, sweep):
            print()
            print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
