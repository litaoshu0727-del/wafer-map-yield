# Two attempts to use unlabeled wafers: self-supervised pretraining and pseudo-labels

[← Back to README](../../README.md) · [中文](../unlabeled-data.md)

Neither method beat training long enough on the labeled data.

## Self-supervised pretraining: an attempt that did not work

Only about a fifth of the 811k wafers are labeled. I tried pretraining the convolutional part of the CNN with SimCLR on unlabeled wafers and then fine-tuning on a few labels, to see whether that reduces the need for labels.

- Pretraining data: 393k unlabeled first-test wafers from lots with no labeled wafer at all. Whatever the seed's split, no validation or test lot appears in pretraining.
- Contrastive method: two random transformations of the same wafer should have close features, different wafers distant ones.
- Evaluation: only 1%, 5% or 10% of the training lots keep their labels, compared with random initialisation. With the same seed both use the same lots.

Test macro-F1:

| Labels | Random initialisation | Pretraining v1 | Pretraining v2 (encoder learning rate ×0.1) |
|---|---|---|---|
| 1% (about 1,100 wafers) | 0.509 ± 0.044 (3 seeds) | 0.509 (seed 0; random initialisation on the same seed: 0.549) | 0.466 ± 0.059 (3 seeds, lower than random initialisation on each, paired difference −0.043 ± 0.016) |
| 5% | 0.745 (seed 0) | 0.725 | – |
| 10% | 0.788 (seed 0) | 0.777 | 0.747 |

Pretraining never beat random initialisation. To see why, the convolutional layers were frozen and only a logistic regression was trained on top (a linear probe), which measures how much information the features themselves carry:

| Convolutional layers from | 1% labels | 10% labels | All labels |
|---|---|---|---|
| Random initialisation, untrained | 0.387 | 0.440 | 0.465 |
| Pretraining v1 | 0.391 | 0.602 | 0.640 |
| Pretraining v2 | 0.461 | 0.588 | 0.647 |
| Supervised training (reference ceiling) | 0.759 | 0.788 | 0.794 |

- Pretraining did learn something (0.15–0.18 above random features), but stayed far from supervised features, and the small advantage was lost in fine-tuning.
- Version 1 used only rotations, flips and a little failing-die noise, so the two views of a wafer had almost the same outline and fail count. My guess is that the encoder could pair views by "outline + fail density" without learning patterns, and the contrastive loss settled at a low value. Version 2 added random zoom, shift and fail noise of random strength. The contrastive task got harder (loss rose from 2.39 to 3.36), but the linear probe improved only at 1% labels, and fine-tuning got worse.
- Other likely reasons: the encoder is small (4 convolutional blocks, 128 dimensions), while contrastive learning usually needs bigger models, bigger batches and many more epochs; and wafer maps are small grids with three values, where transformations that work on natural images may not fit.

Conclusion: in this setting self-supervised pretraining is not worth it, and the default pipeline still trains from random initialisation. The code stays in `wafermap/pretrain.py` and `wafermap/budget.py`. Both sides of the 1%–10% comparison above trained for the same number of epochs, but as the next section shows, that number is too small for little data, so the absolute values in the table are low.

## Pseudo-labels: the gain came from training longer, not from the pseudo-labels

The second way to use unlabeled wafers is pseudo-labelling: train a teacher on the labeled data, let it label the 393k unlabeled wafers, keep only predictions with confidence ≥ 0.9, and train a student from scratch on both the real labels and the pseudo-labels. Pseudo-labels are dominated by `none`, so at most 3 `none` wafers are kept per defect wafer; each run adds about 60k–80k wafers.

Comparing only teacher and student, the pseudo-labels help every time:

| Real labels | Teacher | Student (with pseudo-labels) | Student − teacher |
|---|---|---|---|
| 10% (about 11k wafers, 3 seeds) | 0.771 | 0.780 | +0.010, higher on all 3 seeds |
| 100% (about 109k wafers, 3 seeds) | 0.854 | 0.860 | +0.006, higher on all 3 seeds |

But the student's training set is larger, so for the same number of epochs it takes several times as many gradient steps as the teacher (about 6 times at 10%). So a control was added: real labels only, with the epochs raised to roughly match the student's number of steps (40 → 240 epochs at 10%, 15 → 26 at 100%).

| Real labels | Teacher | Student (with pseudo-labels) | Control (real labels only, same steps) | Student − control |
|---|---|---|---|---|
| 10% (3 seeds) | 0.771 | 0.780 | **0.801** | −0.021 ± 0.011, lower on all 3 seeds |
| 100% (3 seeds) | 0.854 | 0.860 | 0.861 | −0.001 ± 0.008, a tie (the student is higher on 1 of 3 seeds) |

- At 10% labels, simply training longer beats adding pseudo-labels by 0.021. The pseudo-labels did not just fail to help, they hurt: the teacher's own mistakes were learned as labels, and the unlabeled wafers are distributed differently from the labeled ones (the dashboard project shows this too).
- At 100% labels the two tie; the pseudo-labels add no information.
- A side finding: at 10% labels 40 epochs are far too few; 240 epochs raise the score from 0.771 to 0.801. In other words, a tenth of the labels reaches 0.80, against 0.854 with all labels.

Without this control the conclusion would have read "pseudo-labels improve macro-F1 by 0.01", and it would have been wrong. Neither way of using unlabeled data (contrastive pretraining and pseudo-labels) beat training long enough on the labeled data in this dataset.
