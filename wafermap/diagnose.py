"""Look inside a trained CNN: confidence routing, confident errors, Grad-CAM.

    python -m wafermap.diagnose --data data/wm811k/LSWMD.pkl \\
        --run reports/wm811k_seeds/cnn_w0.5_s0 --out reports/wm811k_diag

`--run` is the prefix of a run from wafermap.cnn: it needs `<run>.pt` and
`<run>_pred.npz`, and `--seed` must match the run so the split is the same.

Three questions:

1. Routing. If wafers the model is unsure about go to an engineer, what
   share can be decided automatically at a given error rate? The threshold
   is chosen on the validation lots and then checked on the test lots.
2. Confident errors. Wrong predictions made with high confidence are either
   real model failures or wrong labels; the gallery is for reviewing them.
3. Grad-CAM. Does the model look at the failing dies that define a pattern?
   Measured as the share of attention on failing dies divided by their share
   of the wafer area (lift 1 = no better than spread evenly).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from .cnn import WaferCNN, one_hot, pick_device, splits
from .data import CLASSES, load_labeled, to_arrays
from .plots import WAFER_CMAP, _save
from .routing import coverage_curve, pick_threshold, route

LINE = "#2a78d6"
MARK = "#eb6834"
PATTERNS = [c for c in CLASSES if c not in ("none", "Near-full")]


# ---------------------------------------------------------------- routing
def plot_routing(curve: pd.DataFrame, chosen: list[dict], path: Path) -> Path:
    """Error rate of the auto-decided wafers against the share decided automatically."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["coverage"] * 100, (1 - curve["accuracy"]) * 100, color=LINE, lw=2)
    notes = []
    for r in chosen:
        x, yv = r["coverage"] * 100, (1 - r["auto_accuracy"]) * 100
        ax.plot(x, yv, "o", color=MARK, ms=8, mec="white", mew=2, zorder=3)
        notes.append(f"threshold {r['threshold']:.2f} (set on validation lots): "
                     f"{x:.0f}% automatic, {yv:.2f}% wrong")
    ax.text(0.02, 0.97, "\n".join(notes), transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xlabel("Wafers decided automatically (%)")
    ax.set_ylabel("Error rate on those wafers (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Test lots: send the least confident wafers to manual review")
    return _save(fig, path)


# ---------------------------------------------------------------- Grad-CAM
def grad_cam(model: WaferCNN, x: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """Grad-CAM maps (N, H, W) in [0, 1] for class `target`, at input resolution.

    Taken at the last convolution (8x8 for a 64x64 input), so the map is
    coarse: it shows the region, not single dies.
    """
    layer = model.features[-1][2]  # ReLU of the last conv block, before pooling
    store = {}
    handle = layer.register_forward_hook(lambda m, i, o: store.update(act=o))
    model.eval()
    logits = model(x)
    handle.remove()
    act = store["act"]
    grads = torch.autograd.grad(logits.gather(1, target[:, None]).sum(), act)[0]
    cam = torch.relu((grads.mean(dim=(2, 3), keepdim=True) * act).sum(1, keepdim=True))
    cam = nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / cam.flatten(1).amax(1).clamp_min(1e-8)[:, None, None]
    return cam.detach().cpu().numpy()


def fail_lift(cam: np.ndarray, wafer: np.ndarray) -> float:
    """Share of attention on failing dies / share of the die area that fails."""
    die, fail = wafer >= 1, wafer == 2
    if not fail.any() or fail.sum() == die.sum():
        return float("nan")
    on_die = cam[die].sum()
    if on_die <= 0:
        return float("nan")
    return float((cam[fail].sum() / on_die) / (fail.sum() / die.sum()))


def cams(model: WaferCNN, X: torch.Tensor, target: np.ndarray, device: str, batch: int = 256) -> np.ndarray:
    out = [grad_cam(model, one_hot(X[i:i + batch].to(device)),
                    torch.from_numpy(target[i:i + batch]).to(device)) for i in range(0, len(X), batch)]
    return np.concatenate(out)


def plot_cam_gallery(X: np.ndarray, cam: np.ndarray, titles: list[str], path: Path) -> Path:
    """Top row: wafer map (red = failing die). Bottom row: where the model looked."""
    n = len(titles)
    fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 4.0), gridspec_kw={"hspace": 0.05})
    for j in range(n):
        axes[0, j].imshow(X[j], cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
        axes[0, j].set_title(titles[j], fontsize=9)
        shown = np.ma.masked_where(X[j] == 0, cam[j])
        axes[1, j].imshow(shown, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        for ax in axes[:, j]:
            ax.axis("off")
    axes[0, 0].text(-0.15, 0.5, "wafer map", transform=axes[0, 0].transAxes, rotation=90,
                    va="center", ha="right", fontsize=9)
    axes[1, 0].text(-0.15, 0.5, "Grad-CAM", transform=axes[1, 0].transAxes, rotation=90,
                    va="center", ha="right", fontsize=9)
    return _save(fig, path)


def plot_error_gallery(X: np.ndarray, titles: list[str], path: Path, cols: int = 6) -> Path:
    rows = max(1, int(np.ceil(len(titles) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(1.9 * cols, 2.2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, img, t in zip(axes.flat, X, titles):
        ax.imshow(img, cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(t, fontsize=8)
    return _save(fig, path)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="run prefix, e.g. reports/wm811k_seeds/cnn_w0.5_s0")
    ap.add_argument("--seed", type=int, default=0, help="seed the run was trained with")
    ap.add_argument("--out", default="reports/diagnose")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--targets", default="0.99,0.995", help="auto-decided accuracies to aim for")
    ap.add_argument("--confident", type=float, default=0.9, help="confidence that counts as sure")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_labeled(args.data)
    X_np, y_np = to_arrays(df, args.size)
    _, val, test = splits(y_np, df["lotName"].to_numpy(), args.seed)
    pred = np.load(f"{args.run}_pred.npz")
    if not (np.array_equal(pred["val"], val) and np.array_equal(pred["test"], test)):
        raise SystemExit("split does not match the run: check --seed and --data")
    pv, pt = pred["val_proba"], pred["test_proba"]
    yv, yt, Xt = y_np[val], y_np[test], X_np[test]
    result = {"run": args.run}

    # 1. routing: thresholds from validation lots, scored on test lots
    chosen = [route(pt, yt, pick_threshold(pv, yv, float(t))) | {"target": float(t)}
              for t in args.targets.split(",")]
    result["all_auto"] = route(pt, yt, 0.0)
    result["routing"] = chosen
    plot_routing(coverage_curve(pt, yt), chosen, out / "routing.png")

    # 2. confident errors
    conf, guess = pt.max(1), pt.argmax(1)
    wrong = np.flatnonzero((guess != yt) & (conf >= args.confident))
    wrong = wrong[np.argsort(-conf[wrong])]
    pairs = (pd.DataFrame({"true": np.array(CLASSES)[yt[wrong]], "pred": np.array(CLASSES)[guess[wrong]]})
             .value_counts().rename("wafers").reset_index())
    pairs.to_csv(out / "confident_errors.csv", index=False)
    result["confident_errors"] = {"threshold": args.confident, "wafers": int(len(wrong)),
                                  "share_of_errors": float(len(wrong) / max((guess != yt).sum(), 1)),
                                  "top_pairs": pairs.head(8).to_dict("records")}
    shown = wrong[:24]
    plot_error_gallery(Xt[shown],
                       [f"label {CLASSES[yt[i]]}\nmodel {CLASSES[guess[i]]} {conf[i]:.2f}" for i in shown],
                       out / "confident_errors.png")
    df.iloc[test[shown]][["lotName", "waferIndex", "failureType"]].assign(
        predicted=np.array(CLASSES)[guess[shown]], confidence=conf[shown].round(3)
    ).to_csv(out / "confident_errors_shown.csv", index=False)

    # 3. Grad-CAM on correctly classified pattern wafers
    device = pick_device()
    model = WaferCNN().to(device)
    model.load_state_dict(torch.load(f"{args.run}.pt", map_location=device))
    X_test = torch.from_numpy(Xt)
    rng = np.random.default_rng(0)
    lifts, examples = {}, []
    for c in PATTERNS:
        i = CLASSES.index(c)
        idx = np.flatnonzero((yt == i) & (guess == i))
        if not len(idx):
            continue
        cam = cams(model, X_test[idx], np.full(len(idx), i), device)
        lift = np.array([fail_lift(cam[k], Xt[idx[k]]) for k in range(len(idx))])
        lifts[c] = {"wafers": int(len(idx)), "median_lift": float(np.nanmedian(lift)),
                    "share_lift_above_1": float(np.nanmean(lift > 1))}
        examples.append((c, int(rng.choice(len(idx))), idx, cam))
    result["grad_cam"] = lifts
    plot_cam_gallery(np.stack([Xt[idx[k]] for _, k, idx, _ in examples]),
                     np.stack([cam[k] for _, k, _, cam in examples]),
                     [c for c, *_ in examples], out / "gradcam.png")

    (out / "diagnose.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
