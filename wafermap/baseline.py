"""Baseline classifier: random forest on the hand-built spatial features.

    python -m wafermap.baseline --data data/synthetic.pkl --out reports
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .data import CLASS_TO_IDX, load_labeled
from .evaluate import score, split_indices
from .features import FEATURE_COLUMNS, extract_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--name", default="baseline_rf", help="prefix for the output files")
    ap.add_argument("--max-wafers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-records", action="store_true",
                    help="keep repeat/retest records of the same wafers (dropped by default)")
    args = ap.parse_args()

    df = load_labeled(args.data, args.all_records)
    if args.max_wafers and len(df) > args.max_wafers:
        df = df.sample(args.max_wafers, random_state=args.seed).reset_index(drop=True)
    X = extract_features(df)[FEATURE_COLUMNS].to_numpy()
    y = df["failureType"].map(CLASS_TO_IDX).to_numpy()
    train, test = split_indices(y, df["lotName"].to_numpy(), seed=args.seed)

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=args.seed, n_jobs=-1)
    model.fit(X[train], y[train])
    score(y[test], model.predict(X[test]), args.out, args.name, {"seed": args.seed})

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print("\nTop features:")
    print(importance.head(8).round(3).to_string())


if __name__ == "__main__":
    main()
