"""Yield report: spatial features, loss Pareto, lot excursions.

    python -m wafermap.report --data data/synthetic.pkl --out reports
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import analysis, plots
from .data import labeled, load_dataset
from .features import extract_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--max-wafers", type=int, default=None,
                    help="randomly sample this many labeled wafers (the full WM-811K takes a while)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = labeled(load_dataset(args.data))
    if args.max_wafers and len(df) > args.max_wafers:
        df = df.sample(args.max_wafers, random_state=0).reset_index(drop=True)
    feats = extract_features(df)

    breakdown = analysis.loss_breakdown(feats)
    pareto = analysis.pattern_pareto(df, feats)
    lots = analysis.lot_summary(df, feats)
    slots = analysis.slot_yield(df, feats)

    feats.assign(lotName=df["lotName"], waferIndex=df["waferIndex"],
                 failureType=df["failureType"]).to_csv(out / "features.csv", index=False)
    pareto.to_csv(out / "pattern_pareto.csv")
    lots.to_csv(out / "lot_summary.csv")
    slots.to_csv(out / "slot_yield.csv")
    plots.gallery(df, out / "gallery.png")
    plots.pareto(pareto, out / "pattern_pareto.png")
    plots.lot_trend(lots, out / "lot_trend.png")

    print(f"{len(df)} labeled wafers, {len(lots)} lots")
    print(f"overall yield      {breakdown['yield']:.1%}")
    print(f"systematic loss    {breakdown['systematic_loss']:.1%}  (fails in groups of 5+ dies)")
    print(f"random loss        {breakdown['random_loss']:.1%}")
    print(f"excursion lots     {int(lots['excursion'].sum())} of {len(lots)}")
    print("\nYield loss Pareto:")
    print(pareto.round(3).to_string())
    print(f"\nwrote CSVs and figures to {out}/")


if __name__ == "__main__":
    main()
