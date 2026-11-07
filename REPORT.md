# Why microimpute's QRF MOOP draws fall 3.6% short

PolicyEngine's MOOP imputation experiment ([`experiments/moop_spm`](https://github.com/PolicyEngine/microimpute/tree/e4cbe025dcea8a8b2733d9562f2fa04e50719d5b/experiments/moop_spm) in microimpute) imputes medical out-of-pocket spending (MOOP) to each Supplemental Poverty Measure (SPM) unit in the 2026 Current Population Survey Annual Social and Economic Supplement (CPS ASEC). The draws of its quantile regression forest (QRF) total 3.6% less than the reported values. We measure each candidate cause's share of that shortfall, on the experiment's folds and on a synthetic process with a known conditional distribution. We also run microcosm-fit's `RegimeGatedQRF`, which Microcosm uses to build PolicyEngine's microdata, through the same tests.

## Summary

Pooled over the experiment's five folds, microimpute's QRF draws total $828.4B against $858.9B reported, 3.55% short. Our rerun reproduces the published draws exactly, and the draws' exact expectation falls 3.48% short.

The forest itself doesn't cause the shortfall: its own conditional mean falls 0.18% short of the reported total, inside the folds' noise. Two steps that turn the forest into draws take the other 3.30 points (fold SE 0.05):

| Part of the shortfall (CPS, five folds pooled) | Points | Fold SE |
|---|---|---|
| microimpute's grid: each draw reads quantile (round(100u) + 1)/102, never above 0.990 | −1.76 | 0.03 |
| quantile-forest's linear quantile: each row's largest pooled value gets half weight | −1.54 | 0.02 |
| the forest's own mean against the reported total | −0.18 | 0.87 |
| **microimpute as run, expected draw** | **−3.48** | 0.83 |

Two other candidates don't cause it:

- **One sample per leaf** loses nothing here. Fully grown trees leave a single distinct MOOP value in 99.98% of leaves (fold 0), so the one stored sample carries the leaf's whole value. Setting `max_samples_leaf=None` keeps every in-bag sample, and quantile-forest then pools them by bootstrap count, so a leaf of several zero-MOOP donors outweighs a leaf holding one positive donor. Zero draws nearly double (12.2% against 6.4% reported), and the total falls a further 1.65 points.
- **Weights** barely move the total. Resampling donors in proportion to weight raises it by 0.52 points (fold SE 0.54); passing weights to the split criterion, as microimpute does when given weights, lowers it by 0.25 (SE 0.18).

[PR 219](https://github.com/PolicyEngine/microimpute/pull/219), a draft, removes the shortfall on CPS when we run its own code (+0.3% unweighted, +0.5% weighted). It drops the grid, weights leaves per tree, and stops trees at leaves of 20 donors, so each row's pool holds about 4,300 values and the linear quantile's loss shrinks to 0.13 points. Microcosm's `RegimeGatedQRF` falls 1.2% short in expectation and 1.0% short as drawn, the figure [PR 223](https://github.com/PolicyEngine/microimpute/pull/223) reports. Its linear interpolation across 203 knots costs 0.44 points of that, and the remaining 0.7% falls inside the folds' noise.

On a synthetic process with a known conditional mean, the same two draw mechanisms cost more (grid 4.5 points, linear quantile 4.0). But microimpute's forest mean there runs 7.7% high, so its expected draws net out 0.8% short. The excess comes from the adaptive splits of fully grown trees. They give donors with large positive noise about a fifth more catchment than average, which adds 11 to 13 points, and filling the same trees with donors the splits never saw removes it. PR 219's forests, with leaves of 20 or more donors, show no such excess (forest mean −0.1% where sampling ignores the noise). Microcosm's amount forest, which also grows its trees fully, carries the same selection there (10.9 points), and reading its amounts as the pool mean puts the total 9 to 11% high; honest fills of forests we build its way remove the excess. On CPS MOOP the same-tree test finds no selection in either forest: −0.42 points (SE 0.61) for microimpute's and −0.81 (SE 0.49) for Microcosm's construction.

## The candidates, one at a time

All CPS numbers pool the five folds: every SPM unit receives an imputation once, from a forest fit on the other four folds, and we compare against the SPM-unit-weighted reported total. Differences between draw mechanisms come from the same recipients and the same trees, so their fold SEs run small. A level against the reported total also carries the recipients' own sampling noise, which the fold SE of a level (about 0.85 points) measures.

### Grid: 1.76 points

microimpute draws `u ~ Beta(1, 1)`, rounds `100u` to an index `k` and reads the forest's quantile at `(k + 1)/102` (`microimpute/models/qrf.py:273-295`). No draw reads above the 99.0th percentile of its row's conditional distribution, and every draw's quantile shrinks toward the median (`q ≈ 0.0098 + 0.98u`). For right-skewed spending the lost top percentile outweighs the lost bottom one. Reading one exact quantile per row instead raises the total by 2.07 points at microimpute's linear quantile, or 1.45 points under the inverse ECDF; the 1.76 averages the two orders of removal.

### Leaf samples: not a cause

`_QRFModel.fit` never sets `max_samples_leaf` (`microimpute/models/qrf.py:205-212`), so quantile-forest keeps its default of 1 (`quantile_forest/_quantile_forest.py:1333`) and stores one in-bag sample per leaf. That could thin each row's distribution, but on this data it doesn't. The one-sample pools average to the forest's per-tree leaf means within 0.002% on every fold and forest, because microimpute keeps quantile-forest's growth defaults (`max_depth=None`, `min_samples_leaf=1`; `get_params` confirms them). Those defaults expand nodes until every leaf holds a single value (`_quantile_forest.py:1096-1099`). On fold 0's forest, 0.02% of leaves hold more than one distinct MOOP value, and those leaves hold 0.04% of in-bag samples (`harness/recheck.py`).

Setting `max_samples_leaf=None`, as PR 219 and Microcosm both do, keeps every in-bag sample. quantile-forest's `predict` then pools them across trees by count unless the caller passes `weighted_leaves=True`, as PR 219 does (`_quantile_forest.py:623`). Leaves that hold several donors get proportionally more weight than leaves that hold one, and on CPS MOOP the zero leaves hold more donors. On fold 0's forest a zero leaf holds 2.84 in-bag samples on average and a positive leaf 1.61. Count pooling therefore gives zeros 12.1% of a recipient's pool where per-tree weights give them 6.9%, and 6.6% of the fold's recipients report zero. Across the five folds:

| CPS, five folds | Share of zero draws | Total vs reported |
|---|---|---|
| reported | 6.40% | — |
| microimpute as run (one sample per leaf) | 6.19% | −3.48% |
| one sample per leaf, exact u, inverse ECDF | 6.99% | −0.18% |
| all samples, count-pooled, exact u, inverse ECDF | 12.24% | −2.35% |

At microimpute's own draw, `max_samples_leaf=None` costs 1.65 points. Count pooling takes 2.17; the larger pool (a median of 169 values instead of 100) gives back 0.50 by shrinking the linear quantile's loss, and the grid's interaction gives back 0.02.

### Weights: not material

The experiment fits without weights (`experiments/moop_spm/methods.py:324-330`). With weights, microimpute passes them to quantile-forest, and under scikit-learn 1.8 they weight only the split criterion; leaf values stay unweighted and the bootstrap ignores them (`_quantile_forest.py:267-275, 358-373`). We measured both that and Microcosm's approach, which resamples donors in proportion to weight before growing each forest:

| CPS, five folds: change in the total against the unweighted forest, points | At microimpute's draw | Forest mean |
|---|---|---|
| split-criterion weights (microimpute with `weight_col`) | −0.25 (SE 0.18) | −0.08 |
| weighted bootstrap (Microcosm) | +0.52 (SE 0.54) | +0.26 (SE 0.62) |

Neither moves the total by more than its fold SE. The weighted-bootstrap forest's own mean exceeds the reported total by 0.08%.

### Linear quantile: 1.54 points

quantile-forest reads quantiles with numpy's linear (type 7) rule at rank `(N − 1)q + 1` in each row's pooled sample (`_quantile_forest.py:738-773`). Integrated over `u`, that rule averages to `(Σz − (z_min + z_max)/2)/(N − 1)`, so the largest pooled value gets half weight and the draws' mean falls short of the pool's mean by about `(z_max/2 − mean)/(N − 1)`. With `N = 100` values per row and a right-skewed pool, that costs 1.54 points. Reading the pool's inverse ECDF instead keeps the mean exactly.

### The forest's own mean: −0.18%

When we read it without the grid and with an inverse ECDF, microimpute's forest averages 0.18% short of the reported total (fold SE 0.87). To test whether its adaptive splits bias that mean, we grew forests on random halves of each fold's donors. We then filled the same trees two ways: with the half that grew them (adaptive) and with the other half, whose values the splits never saw (honest). On CPS the two fills agree:

| CPS, five folds pooled | Per-tree weights | Count-pooled |
|---|---|---|
| half-data trees, adaptive fill | −0.39% | −2.64% |
| same trees, honest fill | +0.03% | −0.00% |
| adaptive minus honest (fold SE) | −0.42 (0.61) | −2.63 (0.55) |

With per-tree weights, which microimpute's one-sample leaves reproduce, adaptivity costs nothing measurable. With count pooling it costs 2.6 points, and only under adaptive fills: under the honest fill, count pooling moves the total by 0.03 points. The adaptive splits leave in-bag zeros in multi-donor leaves (2.84 in-bag samples per zero leaf against 1.61 per positive leaf on fold 0), which count pooling rewards. The full forests also hold the donors that grew them, so their 2.17-point loss from count pooling comes from the same effect.

## How the shortfall grows up the distribution

The draws' shortfall widens up the distribution. The grid and the linear quantile explain the tail, and the forest explains the median's 2% gap:

| Realized draws vs reported, five folds | total | median | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|---|
| microimpute as run (published draws) | −3.5 | −1.9 | −2.6 | −3.5 | −4.4 | −6.8 |
| exact u, linear quantile | −1.4 | −2.1 | −1.7 | −1.1 | −0.8 | −1.8 |
| exact u, inverse ECDF | +0.1 | −2.0 | −0.9 | +0.1 | +1.2 | +0.2 |

Removing the grid lifts p99 from −6.8% to −1.8%, and removing the linear quantile lifts it to +0.2%. The median stays 2% low under every reading of the same forest, so the forest's conditional distributions, rather than the draw, set it.

## PR 219

[PR 219](https://github.com/PolicyEngine/microimpute/pull/219), a draft at head `296e47d`, changes four things the shortfall depends on:

1. It reads one exact quantile per row, with no grid (`microimpute/models/qrf.py:362` at `296e47d`).
2. It sets `max_samples_leaf=None` but reads leaves with `weighted_leaves=True` (`qrf.py:290-301`), so each tree's leaf carries equal weight and zeros don't inflate.
3. It sets `min_samples_leaf=20` (`config.py:52`, which `qrf.py:215` applies as a runtime default), so its trees stop at leaves of 20 or more donors. Each row's pool then holds about 4,300 values, and the linear quantile's half weight on the largest costs 0.13 points.
4. Given weights, it inverts a survey-weighted leaf ECDF without interpolation (`qrf.py:232-288`), which keeps the forest's mean exactly.

We ran PR 219's own code from a worktree on the same folds and seeds (`harness/pr219.py`). For the unweighted draw's exact expectation we integrated its quantile function over 4,000 levels; 16,000 levels change the total by 5e-7.

| CPS, five folds pooled | Expected | Realized | Forest mean | Zero draws |
|---|---|---|---|---|
| reported | — | — | — | 6.40% |
| microimpute main | −3.48% | −3.55% | −0.18% | 6.19% |
| PR 219, unweighted | +0.30% | +0.60% | +0.43% | 6.06% |
| PR 219, weighted | +0.49% | +0.86% | +0.49% | 6.33% |

PR 219 removes the shortfall on CPS. Its expected totals land inside the fold SE (0.66) of the reported total, and its unweighted draws track the reported distribution from the median (+1.2%) through p99 (−0.4%), where main's draws run 1.9% and 6.8% short. Its forest parameters differ from main's only in `min_samples_leaf` and `max_samples_leaf` (`get_params` confirms it). Its forest's own mean exceeds main's by 0.61 points (SE 0.33), and its draws close the median gap that main's fully grown trees leave under every reading.

On the synthetic process PR 219 lands within its replication noise of the truth where sampling ignores the noise (x_only: −0.35% unweighted, +0.30% weighted, SEs 0.3). Under informative sampling its unweighted draws run 1.85% low (SE 0.54), of which the sampling itself accounts for 1.34 points, and its survey-weighted draws remove that (+0.03%, SE 0.56). Its unweighted forest means land at −0.13% (x_only) and −1.62% (informative, where the sampling costs microimpute's forests 1.34 points), against +10.4% and +7.7% for main's fully grown trees.

## Microcosm's RegimeGatedQRF

Microcosm avoids the grid, keeps every leaf sample of amounts with many distinct values, like MOOP, and resamples donors by weight. It gates zeros with a gradient-boosting classifier and draws amounts from a forest it grows on positive donors only (`microcosm/fit/qrf.py:2146-2155`). It reads that forest's linear quantiles at 203 knots and interpolates at each row's own uniform (`qrf.py:729-760, 885-892, 1018-1036, 1612-1625`).

| CPS, five folds | Expected | Realized |
|---|---|---|
| Microcosm, weighted (PR 223's configuration) | −1.17% | −1.03% |
| same gate and forest, amount read as the pool mean | −0.73% | — |
| Microcosm, unweighted | −2.00% | −1.99% |
| same gate and forest, amount read as the pool mean | −1.29% | — |

The gate gets zeros right (6.31% expected against 6.40% reported). Linear interpolation of the amount costs 0.44 points weighted and 0.71 unweighted. Without it, the weighted total falls 0.73% short of the reported total, inside the folds' noise. Microcosm keeps every leaf sample, as microimpute does with `max_samples_leaf=None`, but its zeros don't inflate. The gate alone decides them, and the amount forest grows on positive donors only, so its leaves hold no zeros for count pooling to over-weight.

Microcosm grows its amount forest's trees fully, as microimpute does, and on the synthetic process that forest carries the same selection. There Microcosm's draws run 8.0% high (weighted, informative design) and 9.3% high (x_only), its pool-mean reading runs 9.2% and 10.6% high, and the exact split puts 10.9 points on selection. On CPS MOOP it doesn't run high. When we read its amounts as the pool mean, the total falls 0.73% short of the reported total, and the same-tree test finds no selection in forests we build its way (adaptive minus honest −0.81 points, SE 0.49).

## The synthetic process

`harness/synth.py` builds a population of 1.5 million units: about 3% spend nothing, and the rest spend lognormal amounts whose location and spread rise with the predictors. It draws a Poisson sample of about 50,000, and one fifth of the sample receives imputations from forests we fit on the rest. We score each total against the recipients' weighted population conditional mean, which the process fixes, so each level carries only replication noise. In the informative design, inclusion also falls slightly with each unit's own noise draw, which lowers the sample's conditional mean by about 1.3%; in the x_only design, inclusion depends on the predictors alone.

| Synthetic, % vs truth, mean (SE) over replications | informative (8) | x_only (4) |
|---|---|---|
| microimpute as run, expected | −0.76 (0.44) | +1.82 (0.61) |
| grid's contribution | −4.47 (0.08) | −4.53 (0.07) |
| linear quantile's contribution | −4.00 (0.08) | −4.03 (0.06) |
| the forest's own mean | +7.71 (0.54) | +10.38 (0.62) |
| PR 219, unweighted, expected | −1.85 (0.54) | −0.35 (0.27) |
| PR 219, weighted, expected | +0.03 (0.56) | +0.30 (0.29) |
| Microcosm, weighted, expected | +7.98 (0.70) | +9.30 (0.79) |
| Microcosm, weighted, amount read as the pool mean | +9.17 (0.70) | +10.55 (0.80) |

The grid and the linear quantile pull the same way as on CPS, and harder. But here the forest's own mean runs 7.7% high (10.4% in x_only), so microimpute's two losses nearly cancel it, and Microcosm, which has no grid, lands 8% high. Reading Microcosm's amounts as the pool mean would push it to 9%.

### Where the synthetic forest's excess comes from

On the synthetic process we know each donor's conditional mean, so a forest's aggregate splits exactly into three parts (`harness/forest_bias.py`). Each recipient's estimate weights donor values, so the aggregate equals Σ c_j y_j, where donor j's catchment c_j sums its weights across recipients. Selection, Σ c_j (y_j − m*_j), weights each donor's own noise by its catchment, with m*_j the conditional mean among sampled units; it averages zero unless catchments depend on that noise. Informative sampling, Σ c_j (m*_j − m_j), measures the sample's tilt. Smoothing, Σ c_j m_j minus the truth, measures what averaging conditional means over neighbors costs. The identity holds to 2e-16.

| Synthetic, points vs truth, mean (SE) | Total | Selection | Informative sampling | Smoothing | Catchment, top 10% noise |
|---|---|---|---|---|---|
| informative: microimpute's forest | +7.71 (0.54) | +11.17 (0.47) | −1.34 | −2.12 (0.19) | 1.21x |
| informative: half-data trees, adaptive fill | +7.57 (0.64) | +11.63 (0.54) | −1.33 | −2.73 (0.21) | 1.22x |
| informative: same trees, honest fill | −3.73 (0.54) | −0.26 (0.42) | −1.34 | −2.13 (0.20) | 1.00x |
| x_only: microimpute's forest | +10.38 (0.62) | +12.57 (0.80) | — | −2.18 (0.35) | 1.20x |
| x_only: half-data trees, adaptive fill | +9.38 (0.68) | +12.23 (0.63) | — | −2.84 (0.38) | 1.21x |
| x_only: same trees, honest fill | −2.38 (0.53) | −0.12 (0.37) | — | −2.26 (0.33) | 1.00x |

The last column divides the mean catchment of positive donors in the top tenth of noise by that of all positive donors. In fully grown trees, the adaptive splits give those donors 20 to 21% more catchment than average (31% for the top hundredth), and selection adds 11 to 13 points. When the other half's donors fill the same trees, the top tenth of noise gets average catchment, and selection falls to zero within its SE, leaving smoothing's −2.1 to −2.3 points and the informative sampling. Count pooling, as `max_samples_leaf=None` reads these forests, lowers each total by about 0.8 points, nearly all of it through selection.

We don't know CPS's conditional mean, so there the same-tree comparison stands in for the split. Adaptive minus honest comes to −0.42 points per tree (SE 0.61), so CPS MOOP shows no selection. We haven't isolated why the two data sets differ.

### Microcosm's amount forest carries the same selection

Microcosm's gate decides zeros, so we split the error of its amount forest alone, weighting each recipient's amount by its true positive probability (`harness/microcosm_actual.py`). When we refit it with the same donors, weights and seed, Microcosm reproduces its saved pool-mean totals exactly, and the harness's count-pooled estimates match the forest's own `predict(quantiles="mean")` to 2e-15. quantile-forest keeps each forest's training targets and their sort order (`_quantile_forest.py:193-200`), and no two synthetic amounts share a value, so each training row maps back to one donor and the catchments come straight from Microcosm's trees. For the same-tree test we grew forests the way Microcosm does on a weighted bootstrap of half the positive donors and filled them with the other half's donors, each carrying its survey weight (`harness/microcosm_bias.py`).

| Synthetic, points vs truth, mean (SE), amounts times true p | Total | Selection | Informative sampling | Smoothing | Catchment, top 10% noise |
|---|---|---|---|---|---|
| informative: Microcosm's own forest | +9.46 (0.69) | +10.92 (0.75) | −1.37 | −0.09 (0.17) | 1.21x |
| informative: Microcosm-style half-data trees, adaptive fill | +9.54 (0.72) | +11.02 (0.67) | −1.37 | −0.11 (0.20) | 1.20x |
| informative: same trees, honest fill | −0.18 (0.63) | +0.97 (0.53) | −1.37 | +0.22 (0.21) | 1.02x |
| x_only: Microcosm's own forest | +10.81 (0.71) | +10.90 (0.59) | — | −0.10 (0.34) | 1.17x |
| x_only: Microcosm-style half-data trees, adaptive fill | +8.96 (0.31) | +9.15 (0.44) | — | −0.19 (0.32) | 1.17x |
| x_only: same trees, honest fill | −0.29 (0.50) | −0.22 (0.41) | — | −0.07 (0.37) | 1.00x |

Selection makes up all of Microcosm's excess; its smoothing comes to −0.1 points. The gate's own probabilities take about 0.3 points off the true-p totals (its positive share averages 96.7%, as the truth's does). With honest fills, the same trees land within their SEs of the truth. In the informative design the survey weights carry each unit's noise: they undo the sampling's −1.4 and put about a point into each selection entry (the honest fill's +1.0). The adaptive splits add the other 10 points (adaptive minus honest selection +10.05, SE 0.32; x_only +9.36, SE 0.61).

On CPS MOOP the same-tree test finds no such selection in Microcosm's construction (`harness/microcosm_cps.py`, amounts weighted by Microcosm's own gate):

| CPS, five folds pooled | Count-pooled (Microcosm's reading) | Per-tree |
|---|---|---|
| Microcosm-style half-data trees, adaptive fill | −1.17% | −0.49% |
| same trees, honest fill | −0.35% | −0.15% |
| adaptive minus honest (fold SE) | −0.81 (0.49) | −0.34 (0.45) |

Microcosm's own forest lands at −0.73% under the same reading. The selection that adds 10 points on the synthetic process doesn't show on CPS MOOP, for Microcosm's forest as for microimpute's. We didn't test Microcosm's other targets.

## Mechanisms in the code

We read this code on 2026-11-07: microimpute `main` at [`e4cbe02`](https://github.com/PolicyEngine/microimpute/tree/e4cbe025dcea8a8b2733d9562f2fa04e50719d5b), PR 219 at [`296e47d`](https://github.com/PolicyEngine/microimpute/tree/296e47dc1d49b8f8be940dd915a24efc2a7b091d), quantile-forest 1.4.2 with scikit-learn 1.8.0, and microcosm-fit at PolicyEngine/microcosm [`7080ba8`](https://github.com/PolicyEngine/microcosm/tree/7080ba8a3ff4ba498575ec1a624a9064a6f619c8) (the commit PR 223 pins).

1. **Grid.** `qrf.py:273` draws `u`; `qrf.py:280-290` builds `grid_size = max(count_samples, 101)` levels at `(k + 1)/102` and picks `k = rint(100u)`.
2. **Linear quantile.** With one sample per leaf, quantile-forest calls `np.quantile(..., method="linear")` (`_quantile_forest.py:738-773`). With all samples, it calls `calc_weighted_quantile` with `C = 1` (`_quantile_forest_fast.pyx:261-295`), which reproduces type 7 on the count-expanded pool.
3. **Leaf weighting.** quantile-forest defaults to `max_samples_leaf=1` (`_quantile_forest.py:1333`) and to `predict(weighted_leaves=False)` (`_quantile_forest.py:623`), which pools samples by count. With `weighted_leaves=True`, it gives each tree's leaf equal total weight (`_quantile_forest_fast.pyx:710-711`).
4. **Weights.** microimpute passes weights to quantile-forest's `fit` (`qrf.py:205-212`); quantile-forest uses them in the split criterion and drops zero-weight rows from leaves (`_quantile_forest.py:267-275`), and its bootstrap ignores them before scikit-learn 1.9 (`_quantile_forest.py:358-373`).
5. **PR 219.** `config.py:52-54` sets `min_samples_leaf=20` and `max_samples_leaf=None`, which `qrf.py:215` applies as runtime defaults; the unweighted draw reads `predict(..., weighted_leaves=True)` at one uniform quantile per row (`qrf.py:290-301, 362`); the weighted draw inverts a survey-weighted leaf ECDF without interpolation (`qrf.py:232-288`).
6. **Microcosm.** microcosm-fit grows the amount forest on positive donors only (`qrf.py:2146-2155`). It takes a weighted bootstrap of them (`qrf.py:265-294, 1018`) and fits a quantile forest with every growth parameter at its default (`qrf.py:1028-1036`), so its trees grow until every leaf holds a single value. It keeps every leaf sample (`max_samples_leaf=None`) unless that forest's donors hold between 2 and 32 distinct values, when it keeps 4,096 samples per leaf (`qrf.py:917, 925, 1019-1027`); MOOP and the synthetic amounts hold far more distinct values, and no run here triggered the bound. It reads that forest with `predict(..., quantiles=grid)` and `weighted_leaves` at its default (`qrf.py:855`), which pools leaf samples by count. Draws interpolate linear quantiles at 203 knots (`qrf.py:729-760, 885-892`), and the gate and the amount use separate uniforms (`qrf.py:1612-1625`).

## Method and validation

The harness (`harness/`) computes each recipient's exact expected draw under each mechanism from the fitted trees, so mechanism differences carry no Monte Carlo noise. It also draws ten realizations per cell for the distribution statistics.

- **Differential tests** in `harness/tests/test_qrfdecomp.py` (9 pass) check that:
  - pooled quantiles match quantile-forest's `predict` for one-sample and all-sample leaves, with and without weights;
  - the per-tree mean matches `predict(quantiles="mean", weighted_leaves=True)`;
  - the type-7 closed form matches numerical integration;
  - replayed microimpute draws match its `predict` to 1e-12;
  - Microcosm's exact expectation matches 600 of its draws within 5 SE;
  - pool quantiles match numpy on the expanded sample (a Hypothesis property).
- **Catchment properties** in `harness/tests/test_catchments.py` (3 pass) check, for per-tree and count-pooled reads and for in-bag and honest fills, that catchments sum to the recipients' weight and that Σ c_j y_j equals the directly computed total. The file also checks that count-pooled estimates match `predict(quantiles="mean")`, and that the harness recovers a weighted-bootstrap forest's training rows from `forest_.y_train` and `sorter_`.
- **Reproduction.** Each fold's microimpute forest reproduces the published cached draws exactly (first draw, max abs diff 0.0), and replays of its draws from the pools match to 8e-11.
- **Realized against exact.** Across 160 comparisons the z-scores average −0.12 with SD 1.20. The three largest gaps fell 4.2, 3.5 and 5.8 SEs short of their expectations. Those ten-draw samples ran low in both mean and SD: their SDs came to 39-67% of what 400 draws give, the pattern of right-skewed totals (skew 0.3-0.5) when a sample misses the upper tail. Against the 400-draw SD, their gaps shrink to 2.3 to 2.4 SEs, and 400-draw reruns of the same cells land at z −1.26, −0.25 and +0.86 (`harness/recheck.py`).
- **Parameters.** The weighted forests use the baseline forest's hyperparameters and `random_state` (`get_params` confirms them).
- **Microcosm refits.** With the same donors, weights and seed, `RegimeGatedQRF` reproduces the saved pool-mean totals on all 12 synthetic replications and all 5 CPS folds (largest relative difference 2e-16). On the synthetic process the harness's count-pooled estimates from the recovered training rows match the forest's own `predict(quantiles="mean")` to 2e-15, and every catchment identity holds to 3e-16.

## Reproduce

```bash
scripts/setup.sh
cd harness
export OMP_NUM_THREADS=4
../.venv/bin/python -m pytest tests -q -p no:cacheprovider
../.venv/bin/python cps.py --folds 0 1 2 3 4 --out ../results/cps.json
../.venv/bin/python synth.py --reps 8 --design informative --out ../results/synth_informative.json
../.venv/bin/python synth.py --reps 4 --design x_only --out ../results/synth_x_only.json
../.venv/bin/python export_cps.py
../.venv/bin/python pr219.py --cps --out ../results/pr219_cps.json
../.venv/bin/python pr219.py --design x_only --reps 4 --out ../results/pr219_x_only.json
../.venv/bin/python pr219.py --design informative --reps 8 --out ../results/pr219_informative.json
../.venv/bin/python forest_bias.py --cps --out ../results/forest_bias_cps.json
../.venv/bin/python forest_bias.py --design x_only --reps 4 --out ../results/forest_bias_x_only.json
../.venv/bin/python forest_bias.py --design informative --reps 8 --out ../results/forest_bias_informative.json
../.venv/bin/python microcosm_bias.py --design x_only --reps 4 --out ../results/microcosm_bias_x_only.json
../.venv/bin/python microcosm_bias.py --design informative --reps 8 --out ../results/microcosm_bias_informative.json
../.venv/bin/python microcosm_actual.py --design x_only --reps 4 --out ../results/microcosm_actual_x_only.json
../.venv/bin/python microcosm_actual.py --design informative --reps 8 --out ../results/microcosm_actual_informative.json
../.venv/bin/python microcosm_cps.py --out ../results/microcosm_cps.json
../.venv/bin/python recheck.py
../.venv/bin/python analyze.py && ../.venv/bin/python analyze2.py
```

`scripts/setup.sh` uses uv to build a Python 3.13 venv from `requirements.lock.txt`. It checks out microimpute `main` at `e4cbe02` into `microimpute/`, installs it, and checks out PR 219's head `296e47d` into `microimpute-pr219/`. `cps.py` loads the 2026 CPS ASEC through the experiment's own loader, which downloads Census's public-use file into `data/cps-asec-2026/` (or the folder `QRF_SHORTFALL_DATA` names) and checks its SHA-256. `export_cps.py` writes the design matrix, MOOP, weights and folds to `results/cps_inputs.npz` for the other CPS scripts; that file holds record-level survey data, so it stays out of git. To compare with the experiment's published draws, first run `python -m experiments.moop_spm.run --data-dir ../data/cps-asec-2026 --output ../data/moop-spm` in `microimpute/`; `cps.py` reads the per-fold draws it caches in `data/moop-spm/cache/` (or the folder `QRF_SHORTFALL_CACHE` names) and skips the comparison without them. Every script caps its forests at four workers or fewer.
