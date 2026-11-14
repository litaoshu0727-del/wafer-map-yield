import json

import numpy as np
import pandas as pd
import pytest

from wafermap.data import CLASSES
from wafermap.routing import coverage_curve, pick_threshold, route
from wafermap.seeds import load_runs, paired


def probs(conf, pred, n_classes=len(CLASSES)):
    """Probability rows whose top class is `pred` with probability `conf`."""
    p = np.full((len(conf), n_classes), 0.0)
    for i, (c, k) in enumerate(zip(conf, pred)):
        p[i] = (1 - c) / (n_classes - 1)
        p[i, k] = c
    return p


def test_routing_sends_the_least_confident_wafers_to_review():
    # the two least confident predictions are the wrong ones
    conf = np.array([0.99, 0.95, 0.9, 0.6, 0.5])
    y = np.array([0, 1, 2, 3, 4])
    proba = probs(conf, [0, 1, 2, 0, 0])
    curve = coverage_curve(proba, y)
    assert curve["accuracy"].tolist() == pytest.approx([1, 1, 1, 0.75, 0.6])
    t = pick_threshold(proba, y, 1.0)
    assert t == pytest.approx(0.9)
    r = route(proba, y, t)
    assert r["coverage"] == pytest.approx(0.6) and r["auto_accuracy"] == 1.0
    assert r["review_share"][CLASSES[3]] == 1.0 and r["review_share"][CLASSES[0]] == 0.0


def test_unreachable_target_routes_everything():
    proba = probs(np.array([0.9, 0.8]), [1, 1])
    assert route(proba, np.array([0, 0]), pick_threshold(proba, np.array([0, 0]), 0.99))["coverage"] == 0


def test_seed_summary_pairs_runs_by_seed(tmp_path):
    for config, scores in {"a": [0.80, 0.82, 0.81], "b": [0.83, 0.82, 0.85]}.items():
        for seed, f in enumerate(scores):
            m = {"macro_f1": f, "macro_precision": f, "macro_recall": f, "accuracy": 0.9,
                 "f1": {c: f for c in CLASSES}}
            (tmp_path / f"{config}_s{seed}_metrics.json").write_text(json.dumps(m))
    runs = load_runs(tmp_path)
    assert len(runs) == 6
    p = paired(runs, "a", "b")
    assert p["seeds"] == 3 and p["mean_diff"] == pytest.approx(0.07 / 3)
    assert p["diffs"] == pytest.approx([0.03, 0.0, 0.04]) and p["b_wins"] == 2


def test_lot_subsampling_keeps_whole_lots():
    torch = pytest.importorskip("torch")  # noqa: F841
    from wafermap.cnn import subsample_lots

    lots = np.repeat(np.arange(20), 5)
    idx = np.arange(100)
    kept = subsample_lots(idx, lots, 0.25, seed=0)
    assert len(np.unique(lots[kept])) == 5 and len(kept) == 25
    assert np.array_equal(subsample_lots(idx, lots, 1.0, seed=0), idx)


def test_contrastive_views_keep_the_die_layout():
    torch = pytest.importorskip("torch")
    from wafermap.pretrain import nt_xent, random_view

    x = torch.ones((8, 16, 16), dtype=torch.uint8)
    x[:, :2, :] = 0           # no-die band on one side
    x[:, 6:10, 6:10] = 2      # failing block
    v = random_view(x, zoom=(1.0, 1.0), max_shift=0.0)
    assert set(v.unique().tolist()) <= {0, 1, 2}
    assert ((v == 0).sum(dim=(1, 2)) == 32).all()           # rotation/flip only moves the band
    zoomed = random_view(x.repeat(16, 1, 1), p_add=0, p_drop=0)
    assert set(zoomed.unique().tolist()) <= {0, 1, 2}
    assert len(set((zoomed == 0).sum(dim=(1, 2)).tolist())) > 1  # zoom/shift changes the outline
    z = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    assert nt_xent(z, z) < nt_xent(z, torch.roll(z, 1, 0))  # matching pairs give a lower loss


def test_fail_lift_rewards_attention_on_failing_dies():
    pytest.importorskip("torch")
    from wafermap.diagnose import fail_lift

    wafer = np.ones((10, 10), dtype=np.uint8)
    wafer[:2, :2] = 2                       # 4 of 100 dies fail
    on = np.zeros((10, 10)); on[:2, :2] = 1
    assert fail_lift(on, wafer) == pytest.approx(25)
    assert fail_lift(np.ones((10, 10)), wafer) == pytest.approx(1)


def test_budget_table_pairs_starts_by_seed(tmp_path):
    from wafermap.budget import load, table

    scores = {("random", 0.1): [0.60, 0.64], ("pretrained", 0.1): [0.70, 0.66],
              ("random", 1.0): [0.85, 0.86], ("pretrained", 1.0): [0.86, 0.86]}
    for (start, frac), f1s in scores.items():
        for seed, f in enumerate(f1s):
            m = {"seed": seed, "train_fraction": frac, "init": "random" if start == "random" else "reports/pre/enc.pt",
                 "n_train": int(1000 * frac), "macro_f1": f, "macro_recall": f, "macro_precision": f}
            (tmp_path / f"{start}_{frac}_s{seed}_metrics.json").write_text(json.dumps(m))
    t = table(load(tmp_path))
    gain = "gain: pre, encoder lr x1"
    assert t.loc[0.1, (gain, "mean")] == pytest.approx(0.06)
    assert t.loc[0.1, (gain, "seeds_up")] == "2/2" and t.loc[1.0, (gain, "seeds_up")] == "1/2"
    assert t.loc[0.1, ("f1_mean", "random")] == pytest.approx(0.62)


def test_pseudo_label_selection_keeps_confident_patterns_and_caps_none():
    pytest.importorskip("torch")
    from wafermap.pseudo import NONE, select_pseudo

    pattern = probs(np.array([0.95, 0.95, 0.5]), [0, 3, 0])       # two confident patterns, one unsure
    none = probs(np.full(20, 0.99), [NONE] * 20)
    keep = select_pseudo(np.vstack([pattern, none]), confidence=0.9, none_ratio=3)
    assert {0, 1} <= set(keep) and 2 not in keep
    assert len(keep) == 2 + 6                                      # none capped at 3 per pattern wafer


def test_mixed_loader_drops_duplicates_and_scores_combinations(tmp_path):
    pytest.importorskip("torch")
    from wafermap.mixed import LABELS, combo_names, combo_table, evaluate, load_mixed, split

    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, (40, 52, 52)).astype(np.int32)
    X[0, 0, 0] = 3                                   # stray value becomes a failing die
    y = np.zeros((40, 8), dtype=np.int32)
    y[:20, 0] = 1                                    # Center
    y[10:30, 6] = 1                                  # Scratch, overlapping -> Center+Scratch
    np.savez(tmp_path / "m.npz", np.concatenate([X, X[:5]]), np.concatenate([y, y[:5]]))
    Xu, yu = load_mixed(tmp_path / "m.npz")
    assert len(Xu) == 40 and Xu.max() == 2 and Xu[0, 0, 0] == 2
    assert set(combo_names(yu)) == {"Center", "Center+Scratch", "Scratch", "none"}
    train, val, test = split(yu, seed=0)
    assert len(set(train) | set(val) | set(test)) == 40 and not set(train) & set(test)

    proba = yu.astype(float)
    proba[10:20, LABELS.index("Scratch")] = 0.1       # miss the scratch on every Center+Scratch wafer
    scores = evaluate(yu, proba)
    assert scores["exact_match"] == pytest.approx(0.75) and scores["exact_by_patterns"][2] == 0.0
    worst = combo_table(yu, proba).iloc[0]
    assert worst["combo"] == "Center+Scratch" and worst["most_common_error"] == "Center"
