import numpy as np
import pandas as pd
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


def test_record_thresholds_can_be_varied_without_rereading_maps():
    from wafermap.records import decide, tag_lots
    import pandas as pd

    # 85% of slots shared, current fails are 15% of the previous lot's, none on a die that already failed
    stats = {"shared": 0.85, "n_die": 1000, "fails_prev": 100, "fails_cur": 15, "overlap": 0}
    assert decide(stats) == "retest"
    assert decide(stats, min_shared=0.9) == "first"      # not enough shared slots any more
    assert decide(stats, few_fails=0.1) == "first"       # 15% is no longer "few"
    tags = tag_lots(pd.DataFrame([stats, {**stats, "fails_cur": 95, "overlap": 93}], index=["a", "b"]))
    assert tags.to_dict() == {"a": "retest", "b": "repeat"}


def test_excursion_sweep_counts_follow_the_rule():
    from wafermap.sensitivity import excursions, lot_table

    df = make_dataset(n_lots=30, seed=3)
    lots = lot_table(df)
    assert lots["n_wafers"].sum() == len(df)
    base = excursions(lots, 0.3, 5)
    feats = extract_features(df)
    assert base["excursions"] == int(analysis.lot_summary(df, feats)["excursion"].sum())
    assert excursions(lots, 0.2, 5)["excursions"] >= base["excursions"] >= excursions(lots, 0.5, 5)["excursions"]


def test_commonality_counts_and_correction():
    from wafermap.commonality import benjamini_hochberg, fisher_scan

    routes = np.array([[0], [0], [0], [1], [1], [1], [1], [1]])
    bad = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    table = fisher_scan(routes, bad, n_tools=2).set_index("tool")
    assert table.loc[0, ["lots", "bad_lots"]].tolist() == [3, 3]
    assert table.loc[0, "p"] == pytest.approx(1 / 56)  # one way out of C(8, 3) to pick exactly those three
    assert table.loc[1, "p"] == pytest.approx(1.0)
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.5]))
    assert q == pytest.approx([0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])


def test_commonality_finds_a_planted_tool_and_only_that_one():
    from wafermap.commonality import flagged, make_routes, plant, shift_scan

    rng = np.random.default_rng(0)
    bad = rng.random(3000) < 0.1
    routes = make_routes(len(bad), rng)
    assert not flagged(shift_scan(routes, bad, rng), "shift")
    table = shift_scan(plant(routes, bad, step=3, tool=1, strength=0.5, rng=rng), bad, rng)
    for method in ("fisher", "shift", "local"):
        assert flagged(table, method) == {(3, 1)}


def test_time_stratified_test_ignores_a_tool_that_only_ran_during_a_bad_stretch():
    from wafermap.commonality import bad_runs, couple, make_routes, shift_scan

    rng = np.random.default_rng(1)
    bad = np.zeros(2000, dtype=bool)
    bad[400:600] = True
    routes = make_routes(len(bad), rng, n_steps=2, n_tools=3)
    routes[:, 0] = np.where(bad, 2, routes[:, 0] % 2)  # tool 2 of step 0 ran exactly during the bad stretch
    table = shift_scan(routes, bad, rng, n_tools=3).set_index(["step", "tool"])
    assert table.loc[(0, 2), "q"] < 1e-6        # the lot-by-lot test is certain
    assert table.loc[(0, 2), "z_local"] == 0    # inside each stretch of time there is nothing to compare
    assert bad_runs(bad).tolist() == [200]
    tied = couple(routes, 1.0, rng)
    assert (tied[:, 1] == tied[:, 0]).all() and (routes[:, 1] != routes[:, 0]).any()


def test_excursion_flags_match_the_excursion_count():
    from wafermap.commonality import excursion_flags
    from wafermap.sensitivity import excursions

    lots = pd.DataFrame({"n_labeled": [10, 10, 4, 6], "top_count": [5, 2, 4, 2],
                         "top_pattern": ["Center", "Loc", "Center", "Loc"], "mean_yield": [0.8, 0.9, 0.7, 0.9]},
                        index=["a", "b", "c", "d"])
    flags = excursion_flags(lots)
    assert flags.to_dict() == {"a": True, "b": False, "d": True}
    assert flags.sum() == excursions(lots)["excursions"]


def _round_wafer(size=30):
    yy, xx = np.mgrid[0:size, 0:size]
    return (np.hypot(xx - (size - 1) / 2, yy - (size - 1) / 2) < size / 2).astype(np.uint8)


def test_join_count_on_a_hand_built_wafer():
    from wafermap.spatial import join_count

    wmap = np.ones((3, 3), dtype=np.uint8)
    wmap[0, 0] = wmap[0, 1] = wmap[2, 2] = 2
    out = join_count(wmap)
    assert (out["joins"], out["fail_joins"]) == (12, 1)
    assert out["expected"] == pytest.approx(12 * (3 * 2) / (9 * 8))  # each join is fail-fail with chance 3*2/(9*8)


def test_join_count_z_is_standard_normal_for_random_fails_and_large_for_a_blob():
    from wafermap.spatial import Z_CRIT, join_count, radial_z

    rng = np.random.default_rng(0)
    base = _round_wafer()
    dies = np.flatnonzero(base)
    z = []
    for _ in range(1500):
        wmap = base.copy()
        wmap.flat[rng.choice(dies, 60, replace=False)] = 2
        z.append(join_count(wmap)["z"])
    z = np.array(z)
    assert abs(z.mean()) < 0.1 and 0.9 < z.std() < 1.1
    assert 0.02 < (z > Z_CRIT).mean() < 0.08

    blob = base.copy()
    blob[10:17, 10:18] = 2  # the same number of fails in one block
    assert join_count(blob)["z"] > 20
    assert radial_z(blob, rng) > 5


def test_ring_shuffle_does_not_flag_a_plain_radial_gradient():
    from wafermap.features import polar_grid
    from wafermap.spatial import join_count, radial_z

    rng = np.random.default_rng(1)
    base = _round_wafer(40)
    r, _ = polar_grid(base.shape)
    plain, ringwise = [], []
    for _ in range(60):
        wmap = base.copy()
        wmap[(base > 0) & (rng.random(base.shape) < 0.6 * r ** 4)] = 2  # fail rate rises towards the edge
        plain.append(join_count(wmap)["z"])
        ringwise.append(radial_z(wmap, rng, n_shuffles=100))
    assert np.mean(plain) > 3          # uniform null: looks clustered
    assert abs(np.mean(ringwise)) < 0.7  # null that knows the radial profile: nothing left


def test_auc_counts_ties_as_half():
    from wafermap.spatial import auc

    assert auc(np.array([1, 2, 3, 4]), np.array([False, False, True, True])) == 1.0
    assert auc(np.array([1, 1, 1, 1]), np.array([False, True, False, True])) == 0.5
