import numpy as np
import pytest

from wafermap import analysis
from wafermap.data import CLASSES, resize, to_arrays
from wafermap.features import FEATURE_COLUMNS, extract_features, wafer_features
from wafermap.synthetic import make_dataset, make_wafer


def mean_features(pattern, n_wafers=20):
    rng = np.random.default_rng(1)
    feats = [wafer_features(make_wafer(pattern, rng, 40)) for _ in range(n_wafers)]
    return {k: float(np.mean([f[k] for f in feats])) for k in feats[0]}


def test_hand_built_wafer():
    wmap = np.ones((10, 10), dtype=np.uint8)
    wmap[0, :] = 0          # 10 cells without a die
    wmap[4:6, 4:7] = 2      # one group of 6 fails
    wmap[9, 0] = 2          # one isolated fail
    f = wafer_features(wmap)
    assert f["n_die"] == 90 and f["n_fail"] == 7
    assert f["yield"] == pytest.approx(1 - 7 / 90)
    assert f["n_clusters"] == 1 and f["n_fail_clustered"] == 6
    assert f["clustered_frac"] == pytest.approx(6 / 7)


def test_center_and_edge_patterns_land_in_their_zones():
    center, ring = mean_features("Center"), mean_features("Edge-Ring")
    assert center["fail_center"] > 0.3 and center["edge_minus_center"] < -0.2
    assert ring["fail_edge"] > 0.3 and ring["edge_minus_center"] > 0.2


def test_scratch_is_more_elongated_than_a_blob():
    assert mean_features("Scratch")["largest_cluster_elongation"] > 2 * mean_features("Loc")["largest_cluster_elongation"]


def test_pattern_free_wafer_has_little_clustered_loss():
    assert mean_features("none")["clustered_frac"] < 0.2 < mean_features("Center")["clustered_frac"]


def test_resize_keeps_values_and_die_fraction():
    wmap = make_wafer("Edge-Ring", np.random.default_rng(0), 33)
    out = resize(wmap, 64)
    assert out.shape == (64, 64) and set(np.unique(out)) <= {0, 1, 2}
    assert (out > 0).mean() == pytest.approx((wmap > 0).mean(), abs=0.03)


def test_dataset_analysis_end_to_end():
    df = make_dataset(n_lots=40, wafers_per_lot=25, seed=3)
    feats = extract_features(df)
    assert list(feats.index) == list(df.index) and not feats[FEATURE_COLUMNS].isna().any().any()

    X, y = to_arrays(df, 32)
    assert X.shape == (1000, 32, 32) and y.min() >= 0 and y.max() < len(CLASSES)

    breakdown = analysis.loss_breakdown(feats)
    assert breakdown["yield"] + breakdown["systematic_loss"] + breakdown["random_loss"] == pytest.approx(1)

    pareto = analysis.pattern_pareto(df, feats)
    assert pareto["loss_share"].sum() == pytest.approx(1)

    lots = analysis.lot_summary(df, feats)
    assert len(lots) == 40
    # the generator plants excursions in roughly a quarter of the lots
    assert 3 <= lots["excursion"].sum() <= 20
    assert lots.loc[lots["excursion"], "mean_yield"].mean() < lots.loc[~lots["excursion"], "mean_yield"].mean()


# ------------------------------------------------- repeat / retest records

def _lot(name, maps):
    import pandas as pd
    return pd.DataFrame({"waferMap": maps, "lotName": name, "waferIndex": range(1, len(maps) + 1),
                         "failureType": "none"})


def test_repeat_and_retest_records_are_tagged():
    import pandas as pd
    from wafermap.records import classify_lots

    rng = np.random.default_rng(0)
    first = [make_wafer("Edge-Ring", rng, 30) for _ in range(5)]
    repeat, retest = [], []
    for wmap in first:
        again = wmap.copy()                               # cumulative record: old fails plus one more
        passing = np.argwhere(wmap == 1)
        again[tuple(passing[0])] = 2
        repeat.append(again)
        delta = np.where(wmap > 0, 1, 0).astype(np.uint8)  # only the die that newly failed
        delta[tuple(passing[1])] = 2
        retest.append(delta)
    other = [make_wafer("Edge-Ring", rng, 30) for _ in range(5)]   # same pattern, different wafers
    df = pd.concat([_lot("lot1", first), _lot("lot2", repeat), _lot("lot3", retest),
                    _lot("lot4", other), _lot("lot9", [m.copy() for m in other])], ignore_index=True)
    record = classify_lots(df)
    assert record.groupby(df["lotName"]).first().to_dict() == {
        "lot1": "first", "lot2": "repeat", "lot3": "retest",
        "lot4": "first",      # an unrelated lot, even with the same defect pattern
        "lot9": "first",      # identical maps, but not the next lot number
    }


def test_synthetic_lots_are_all_first_records():
    from wafermap.records import classify_lots
    df = make_dataset(n_lots=60, wafers_per_lot=25, seed=5)
    assert (classify_lots(df) == "first").all()
