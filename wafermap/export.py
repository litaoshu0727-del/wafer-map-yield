"""Export a trained CNN as a self-contained file for use elsewhere (torch.export).

    python -m wafermap.export --run reports/wm811k_seeds/cnn_w0.5_s0 --out models/wafer_cnn

Writes `<out>.pt2` and `<out>.json`. Load it with
`torch.export.load(path).module()`. The exported program takes a batch of
wafer maps already resized to `size` x `size` as uint8 {0 no die, 1 pass,
2 fail} and returns class probabilities, so the caller needs neither this
package nor the training code. The JSON records the class order, the input
size and how maps must be resized.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .cnn import WaferCNN, one_hot
from .data import CLASSES


class Packaged(nn.Module):
    """uint8 maps (N, H, W) in, class probabilities (N, n_classes) out."""

    def __init__(self, model: WaferCNN):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(one_hot(x)).softmax(1)


def export(run: str, out: str, size: int = 64, note: str = "") -> Path:
    model = WaferCNN()
    model.load_state_dict(torch.load(f"{run}.pt", map_location="cpu"))
    packaged = Packaged(model).eval()
    example = torch.randint(0, 3, (4, size, size), dtype=torch.uint8)
    program = torch.export.export(packaged, (example,), dynamic_shapes={"x": {0: torch.export.Dim.AUTO}})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(program, out.with_suffix(".pt2"))
    meta = {"classes": CLASSES, "size": size, "resize": "nearest neighbour, cell centres",
            "input": "uint8 (N, size, size): 0 no die, 1 pass, 2 fail", "source_run": Path(run).name,
            "note": note}
    metrics = Path(f"{run}_metrics.json")
    if metrics.exists():
        m = json.loads(metrics.read_text())
        meta["test_macro_f1"] = round(m["macro_f1"], 4)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    reloaded = torch.export.load(out.with_suffix(".pt2")).module()
    with torch.no_grad():
        for n in (1, 7):  # batch size must stay free
            x = torch.randint(0, 3, (n, size, size), dtype=torch.uint8)
            assert torch.allclose(reloaded(x), packaged(x), atol=1e-5)
    return out.with_suffix(".pt2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run prefix from wafermap.cnn (needs <run>.pt)")
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    print(export(args.run, args.out, args.size, args.note))


if __name__ == "__main__":
    main()
