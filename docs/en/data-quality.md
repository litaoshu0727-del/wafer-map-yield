# Data quality: repeat records and threshold sensitivity

[← Back to README](../../README.md) · [中文](../data-quality.md)

## Repeat records of the same wafers

In WM-811K, consecutive lot numbers are often the same wafers recorded again at a later test step: the same slots, exactly the same die positions, and a fail map that is either almost identical (a few more fails) or reduced to a handful of scattered fails. The dataset documentation does not mention this; it is inferred by comparing wafer maps one by one, and the rules are in `wafermap/records.py`.

- 187,504 of all 811,457 wafers are such records, but almost none of them are labeled: only 2,266 (1.3%), 1,541 of which are Center.
- These records are dropped by default; otherwise the same wafer is counted twice and can end up in both the training and the test set. `--all-records` keeps them.
- Before and after removing them: excursion lots drop from 913 to 746 (Center-dominated ones from 291 to 158); everything else holds, with the random forest going from 0.815 to 0.811 and the CNN from 0.857 to 0.861.

## Do the conclusions depend on thresholds I picked?

Two sets of thresholds are judgement calls with no ground truth: the four that decide repeat records, and the two in the excursion rule. Each threshold was varied over a range on its own, with the others at their defaults, and the headline numbers were recomputed (`wafermap/sensitivity.py`).

The repeat-record decision barely depends on its thresholds:

| Threshold (default) | Range | Repeat + retest lots | Excursion lots |
|---|---|---|---|
| Share of shared slots (0.8) | 0.6 – 1.0 | 11,948 – 11,950 | 746, unchanged |
| Containment of the fail map (0.9) | 0.8 – 1.0 | 11,618 – 11,953 | 745 – 758 |
| Size similarity required for "repeat" (0.8) | 0.6 – 0.9 | 11,655 – 12,099 | 737 – 751 |
| Fail-count ratio required for "retest" (0.2) | 0.1 – 0.4 | 11,895 – 11,969 | 746, unchanged |

At the defaults there are 11,949 such lots (26%). Across the whole range the count moves by no more than 3%, and the labeled wafers removed stay between 2,065 and 2,410. The reason is that neighbouring lots either match almost perfectly or not at all, with very little in between, so where the threshold sits makes little difference.

The excursion rule is more sensitive, mainly to the minimum number of wafers:

| Minimum labeled wafers | Same-pattern share ≥ 20% | ≥ 30% | ≥ 40% | ≥ 50% |
|---|---|---|---|---|
| 3 | 1,105 | 1,013 | 991 | 974 |
| 5 | 838 | **746** | 730 | 713 |
| 10 | 594 | 504 | 489 | 473 |

- Raising the share threshold from 30% to 50% removes only 4%: a lot either has most of its wafers in one pattern or very few, so this threshold is not sensitive.
- The minimum number of wafers matters: going from 5 to 10 wafers takes the count from 746 to 504. So the absolute number "746" depends on the definition and should not be quoted on its own.
- What does not change with the thresholds: excursions are 7%–15% of eligible lots; the dominant pattern is always Edge-Ring (42%–76%); and among eligible lots, excursion lots have a mean yield 5.5 to 8.5 points below the others (8.0 at the defaults).
