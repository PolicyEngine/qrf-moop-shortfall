# qrf-moop-shortfall

PolicyEngine's MOOP imputation experiment fits microimpute's quantile regression forest (QRF) to the 2026 CPS ASEC, and its draws of medical out-of-pocket spending (MOOP) total 3.6% less than the reported values. We measure each cause's share of that shortfall and run microcosm-fit's `RegimeGatedQRF`, which Microcosm uses to build PolicyEngine's microdata, through the same tests. [REPORT.md](REPORT.md) gives the results, the code behind each mechanism and the validation.

- microimpute's quantile grid costs 1.76 points and quantile-forest's linear quantile 1.54 points. The forest's own mean falls 0.18% short, inside the folds' noise.
- One stored sample per leaf and unweighted fitting don't cause the shortfall. Setting `max_samples_leaf=None` lowers the total a further 1.65 points, because quantile-forest then pools leaf samples by count.
- [PR 219](https://github.com/PolicyEngine/microimpute/pull/219)'s own code removes the shortfall (+0.3% unweighted, +0.5% weighted). Microcosm's `RegimeGatedQRF` falls 1.2% short in expectation, 0.44 points of it from interpolating between 203 quantile knots.
- On a synthetic process with a known conditional mean, the fully grown trees of microimpute's forest and Microcosm's amount forest run 8 to 11% high, because their adaptive splits favor donors with large positive noise. Filling the same trees with donors the splits never saw removes the excess, and CPS MOOP shows no such selection.

## Reproduce

`scripts/setup.sh` builds the pinned environment, and [REPORT.md](REPORT.md#reproduce) lists the commands that regenerate every file in `results/`. The scripts download Census's public-use CPS ASEC 2026 file themselves, and `results/cps_inputs.npz`, which holds record-level survey data, stays out of git.
