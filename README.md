# Wafer Map Yield Analysis (wafer-map-yield)

[![tests](https://github.com/litaoshu0727-del/wafer-map-yield/actions/workflows/tests.yml/badge.svg)](https://github.com/litaoshu0727-del/wafer-map-yield/actions/workflows/tests.yml)

**English** | [中文](README.zh-CN.md)

**In one sentence:** using 811k real wafer bin maps (WM-811K), find where yield is lost, flag excursion lots, and compare two ways of recognising defect patterns.

| What | Result |
|---|---|
| Split the 10.1% yield loss into systematic and random | 3.6% systematic and 6.5% random by a connected-cluster rule; a join-count test shows that 1.1 points of the 3.6% cannot be told apart from chance. Edge-Ring is the largest identifiable source of loss |
| Flag lots in which many wafers share a defect pattern | 746 of 7,251 lots; mean yield 80.9% against 87.1% for the other lots |
| Classify 9 defect patterns, split by lot, 5 random seeds | Random forest macro-F1 0.804, CNN 0.854 |
| Send low-confidence wafers to manual review | 90% decided automatically at 0.45% error (2.8% if every wafer is decided automatically) |
| Test tool commonality methods on synthetic tool routes (the real data has no tool history) | 88% of excursions sit in 3 runs of consecutive lots; when tools process lots in campaigns, a lot-by-lot Fisher test raises a false alarm in 100% of runs, while circular-shift calibration keeps false alarms at or below 7% |
| Compare wafer maps one by one to check data quality | About 26% of lots are repeat records of the same wafers, which the dataset documentation does not mention |

There are also two negative results: neither self-supervised pretraining nor pseudo-labelling beat a supervised model trained long enough on the labeled data. The headline results follow; detailed experiments are in `docs/en/` (see [More analyses](#more-analyses)).

![One real wafer for each defect pattern; failing dies in red](docs/gallery.png)

The bin maps are used to answer three questions:

1. **Where is yield lost?** Locate failing dies by zone (centre, middle, edge) and by radial profile.
2. **Systematic or random?** Separate clustered fails (systematic) from isolated fails (random) with connected components, and build a defect pattern Pareto.
3. **Which lots went wrong?** A lot in which several wafers show the same defect pattern is flagged as an excursion.

Defect patterns are also classified in two ways and compared: hand-crafted spatial features with a random forest (interpretable), and a CNN. The CNN work also covers multi-seed statistics, Grad-CAM, confidence-based routing to manual review, a review of confident errors, two attempts to use unlabeled wafers (self-supervised pretraining and pseudo-labels; neither beat a well-trained supervised model), and multi-label classification of mixed defects. The trained CNN is exported and used in the [yield-dashboard](https://github.com/litaoshu0727-del/yield-dashboard) project to give every wafer a defect pattern.

## Results on real data (WM-811K)

Data: 811,457 wafers in 46,293 lots, of which 172,950 carry an engineer's label. The analysis uses only the labeled wafers, with repeat records removed (see [data quality](docs/en/data-quality.md)), which leaves 170,684.

**Yield loss**

| Metric | Value |
|---|---|
| Overall yield | 89.9% |
| Systematic loss (5 or more connected failing dies) | 3.6% (overstated, see the [join-count test](docs/en/spatial-test.md)) |
| Random loss (isolated fails) | 6.5% |

- 67.8% of failing dies are on wafers labeled `none` (no clear pattern): most yield loss comes from background defects, not from recognisable patterns.
- Among patterned wafers, Edge-Ring costs the most, 16.2% of all failing dies, followed by Edge-Loc at 5.1% and Loc and Center at 2.8% each.
- Mean yield by slot (1–25) stays between 88.2% and 88.5%: there is no slot effect.

![Pareto chart of yield loss by defect pattern](docs/pattern_pareto.png)

The connected-cluster rule has no error rate. Checked with a join-count spatial test, it fires on 71% of `none` wafers, while the test is significant on only 25% of them; 1.1 points of the 3.6% cannot be told apart from chance. Details in [docs/en/spatial-test.md](docs/en/spatial-test.md).

**Lot excursions**

- 7,251 lots have at least 5 labeled wafers; 746 of them (10.3%) are flagged as excursions.
- Dominant pattern: Edge-Ring 404, Center 158, Edge-Loc 78, Loc 42.
- Mean yield is 80.9% in excursion lots and 87.1% in the others.

The number of excursions depends on the thresholds (anywhere from 473 to 1,105), but the dominant pattern and the yield gap do not; see [data quality](docs/en/data-quality.md). The excursions are strongly clustered in time: 653 of the 746 fall in 3 runs of consecutive lots. What this means for tool commonality analysis is in [docs/en/commonality.md](docs/en/commonality.md).

**Defect pattern classification**

80/20 split grouped by lot (no lot has wafers on both sides), about 34,000 test wafers. Each model is run with 5 random seeds; the seed sets both the lot split and the model initialisation. Mean ± standard deviation:

| Model | macro-F1 | Macro precision | Macro recall | Accuracy |
|---|---|---|---|---|
| Random forest (17 spatial features) | 0.804 ± 0.005 | 0.77 | 0.86 | 0.947 |
| CNN (class weight power 0.5, 15 epochs) | 0.854 ± 0.005 | 0.84 | 0.87 | 0.972 |

- With the same seed both models see the same test lots, so they can be compared seed by seed: the CNN is higher on all 5 seeds, by 0.050 ± 0.004 on average.
- The random forest's weakest classes are Loc (F1 0.60), Scratch (0.63) and Edge-Loc (0.67). Its most important feature is the radial position of the largest fail cluster.
- The CNN improves all three: Loc 0.72, Scratch 0.73, Edge-Loc 0.80. Loc and Scratch are still its weakest.

Confusion matrices, normalised by row: each row shows how one true class is predicted (seed 0).

| Random forest | CNN |
|---|---|
| ![Confusion matrix of the random forest](docs/confusion_rf.png) | ![Confusion matrix of the CNN](docs/confusion_cnn.png) |

**Sending low-confidence wafers to manual review**

A fab does not need the model to decide every wafer; the uncertain ones can go to an engineer. The confidence threshold is picked on validation lots and then checked on the test set:

![Share of wafers decided automatically against their error rate](docs/routing.png)

| Approach | Decided automatically | Error rate on those wafers | macro-F1 |
|---|---|---|---|
| Decide every wafer | 100% | 2.8% | 0.854 ± 0.005 |
| Threshold for ≤ 1% error on validation (about 0.77) | 94.4% ± 0.7% | 0.91% ± 0.12% | 0.938 ± 0.008 |
| Threshold for ≤ 0.5% error on validation (about 0.88) | 90.5% ± 1.1% | 0.45% ± 0.08% | 0.960 ± 0.007 |

Mean ± standard deviation over 5 seeds. The targets set on validation lots largely hold on the test set. The figure is seed 0.

The class weight comparison, Grad-CAM, which classes go to review and the confident errors are in [docs/en/model-diagnostics.md](docs/en/model-diagnostics.md).

## More analyses

| File | Contents |
|---|---|
| [docs/en/model-diagnostics.md](docs/en/model-diagnostics.md) | Effect of class weights (5 seeds), Grad-CAM, which classes go to manual review, review of confident errors |
| [docs/en/unlabeled-data.md](docs/en/unlabeled-data.md) | Self-supervised pretraining and pseudo-labels; neither beat training long enough on the labeled data (negative results) |
| [docs/en/mixed-defects.md](docs/en/mixed-defects.md) | Multi-label classification of mixed defects (MixedWM38) |
| [docs/en/spatial-test.md](docs/en/spatial-test.md) | Join-count spatial test: checking the rule behind "systematic loss" |
| [docs/en/data-quality.md](docs/en/data-quality.md) | Repeat records of the same wafers, and threshold sensitivity |
| [docs/en/commonality.md](docs/en/commonality.md) | Checking tool commonality methods (real lot outcomes, synthetic tool routes) |

The same files in Chinese are in [`docs/`](docs/).

## Quick start

```bash
pip install -e ".[dev]"          # core dependencies
pip install -e ".[cnn]"          # optional: PyTorch, needed only for the CNN
pytest                            # 27 tests, 5 of which need PyTorch
python -m wafermap.synthetic --out data/synthetic.pkl --n-lots 80
python -m wafermap.report   --data data/synthetic.pkl --out reports
python -m wafermap.baseline --data data/synthetic.pkl --out reports
python -m wafermap.cnn      --data data/synthetic.pkl --out reports --epochs 30
```

Multi-seed comparison, model diagnostics and export (on the real data):

```bash
# 5 seeds per configuration, files named <config>_s<seed>, then summarise
python -m wafermap.cnn      --data data/wm811k/LSWMD.pkl --out reports/wm811k_seeds --name cnn_w0.5_s1 --seed 1 --epochs 15
python -m wafermap.baseline --data data/wm811k/LSWMD.pkl --out reports/wm811k_seeds --name rf_s1 --seed 1
python -m wafermap.seeds    --runs reports/wm811k_seeds
# review threshold, confident errors, Grad-CAM (--seed must match training)
python -m wafermap.diagnose --data data/wm811k/LSWMD.pkl --run reports/wm811k_seeds/cnn_w0.5_s0 --seed 0 --out reports/wm811k_diag
# threshold sensitivity
python -m wafermap.sensitivity --data data/wm811k/LSWMD.pkl --out reports/wm811k_sensitivity
# join-count test for spatial clustering
python -m wafermap.spatial --data data/wm811k/LSWMD.pkl --out reports/wm811k_spatial
# tool commonality method check (the tool routes are synthetic)
python -m wafermap.commonality --data data/wm811k/LSWMD.pkl --out reports/wm811k_commonality
# pseudo-labels, and mixed-defect multi-label classification (MixedWM38 unpacked to data/mixedwm38/)
python -m wafermap.pseudo   --data data/wm811k/LSWMD.pkl --out reports/wm811k_pseudo --name f0.1_s0 --seed 0 --train-fraction 0.1 --epochs 40
python -m wafermap.mixed    --data data/mixedwm38/Wafer_Map_Datasets.npz --out reports/mixed --seed 0
# export a standalone model file for the dashboard project
python -m wafermap.export   --run reports/wm811k_seeds/cnn_w0.5_s0 --out models/wafer_cnn
```

## Data sources

The datasets are not included in the repository and have to be downloaded:

| Dataset | How to get it | Source |
|---|---|---|
| WM-811K | Search Kaggle for `wm811k-wafer-map`, unpack `LSWMD.pkl` (about 2 GB) to `data/wm811k/LSWMD.pkl` | M.-J. Wu, J.-S. R. Jang, J.-L. Chen, "Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets," *IEEE Transactions on Semiconductor Manufacturing*, vol. 28, no. 1, pp. 1–12, 2015. [doi:10.1109/TSM.2014.2364237](https://doi.org/10.1109/TSM.2014.2364237) |
| MixedWM38 | Kaggle or [the authors' repository](https://github.com/Junliangwangdhu/WaferMap), unpacked to `data/mixedwm38/` | J. Wang, C. Xu, Z. Yang, J. Zhang, X. Li, "Deformable Convolutional Networks for Efficient Mixed-Type Wafer Defect Pattern Recognition," *IEEE Transactions on Semiconductor Manufacturing*, 2020. [doi:10.1109/TSM.2020.3020985](https://doi.org/10.1109/TSM.2020.3020985) |

To run the commands above on WM-811K, pass `--data data/wm811k/LSWMD.pkl`. For a quick first run, add `--max-wafers 20000`.

Notes:

- `LSWMD.pkl` is a pickle written by a Python 2-era pandas that current pandas cannot read. `wafermap/data.py` has a dedicated reader that only lets the file construct numpy objects.
- Only 172,950 of the 811,457 wafers are labeled; this project uses the labeled part.
- `none` dominates the real data and `Near-full` has only 149 wafers, so the metric is macro-F1, not accuracy.
- The real data has no tool or route information, so the excursion analysis stops at "a pattern shared within a lot". Tool commonality is only checked as a method on synthetic tool routes (see [docs/en/commonality.md](docs/en/commonality.md)); it does not and cannot say anything about real tools.

## Layout

| File | Purpose |
|---|---|
| `wafermap/data.py` | Load data, unify fields, nearest-neighbour resizing |
| `wafermap/synthetic.py` | Synthetic wafers with the same structure as WM-811K |
| `wafermap/records.py` | Detect repeat and retest records of the same wafers |
| `wafermap/sensitivity.py` | Threshold sensitivity: sweep the repeat-record and excursion thresholds |
| `wafermap/commonality.py` | Tool commonality: real lot outcomes with synthetic tool routes, false alarms and detection of three tests |
| `wafermap/features.py` | Per-wafer spatial features (zone fail rates, radial and sector profiles, connected clusters) |
| `wafermap/spatial.py` | Join-count test: are a wafer's fails more clustered than random placement, plus a ring-by-ring version |
| `wafermap/analysis.py` | Loss split, pattern Pareto, lot excursions, slot yield |
| `wafermap/baseline.py` | Random forest baseline and feature importance |
| `wafermap/cnn.py` | CNN classifier (class weighting, rotation and flip augmentation, fewer labels by lot, starting from pretrained weights) |
| `wafermap/evaluate.py` | Split and scoring shared by both models |
| `wafermap/seeds.py` | Multi-seed summary: mean ± standard deviation, paired comparison by seed |
| `wafermap/routing.py` | Split wafers by confidence into automatic decisions and manual review |
| `wafermap/diagnose.py` | Review curve, confident errors, Grad-CAM |
| `wafermap/pretrain.py` | SimCLR self-supervised pretraining on unlabeled wafers |
| `wafermap/budget.py` | Label budget curve: random initialisation against pretraining at different label fractions |
| `wafermap/pseudo.py` | Pseudo-labels: a teacher labels unlabeled wafers, a student retrains on both |
| `wafermap/mixed.py` | Multi-label classification of MixedWM38 mixed defects |
| `wafermap/export.py` | Export the model as a `.pt2` file that does not need this project's code |
| `wafermap/report.py` | Write all CSVs and figures in one go |

## Synthetic data

The wafers from `wafermap/synthetic.py` are only for tests and for running the pipeline without the real data. Their patterns are very regular, and scores on them say nothing about real performance.

## Related projects

- [yield-dashboard](https://github.com/litaoshu0727-del/yield-dashboard): yield monitoring dashboard; uses the CNN exported here to give 811k wafers a defect pattern
- [secom-root-cause](https://github.com/litaoshu0727-del/secom-root-cause): root-cause screening on the SECOM fab data

## Next steps

- Systematically relabel the confident errors and estimate how much label noise affects macro-F1

## License

The code is under the MIT license, see [LICENSE](LICENSE). Follow each dataset's own terms and cite the original papers above.
