"""Run microimpute PR 219's own QRF on the same splits.

PR 219 (head 296e47d) changes more than the draw: its ``config.py`` makes
``min_samples_leaf=20`` and ``max_samples_leaf=None`` the QRF's runtime
defaults, so its trees stop at leaves of 20 or more samples instead of
growing until pure. Its unweighted draw reads quantile-forest's
``predict(weighted_leaves=True)`` at one uniform quantile per row; given
weights, it inverts a survey-weighted leaf ECDF without interpolation.

For each split this records, for PR 219 fit without and with weights:

- realized draws from PR 219's own ``predict`` (``n_draws`` per split);
- ``forest_mean``: the forest's per-tree leaf mean (survey-weighted for
  the weighted fit), exact from the fitted trees;
- ``expected``: each row's expected draw, summed. Weighted: exact, equal
  to ``forest_mean``, since the inverse ECDF preserves the mean.
  Unweighted: numerically integrated, quantile-forest's weighted quantile
  function integrated over u by the midpoint rule on ``K`` levels, from
  the forest's own ``predict``. A rerun at 4K levels on the first 1,000
  rows of the first split records how far their subtotal moves; it checks
  convergence on those rows and bounds nothing beyond them.

This process imports microimpute from the PR 219 worktree, so the CPS
inputs come from ``results/cps_inputs.npz`` (``export_cps.py``).

Usage:
  python pr219.py --cps --out results/pr219_cps.json
  python pr219.py --design x_only --reps 4 --out results/pr219_x_only.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PR219_ROOT = HERE.parent / "microimpute-pr219"
sys.path.insert(0, str(PR219_ROOT))
sys.path.insert(1, str(HERE))

import microimpute.models.qrf as pr219_qrf  # noqa: E402

assert Path(pr219_qrf.__file__).resolve().is_relative_to(PR219_ROOT.resolve()), (
    pr219_qrf.__file__
)
from microimpute.models.qrf import QRF  # noqa: E402

import qrfdecomp as qd  # noqa: E402
from cells import draw_stats  # noqa: E402

N_JOBS = 2
K_LEVELS = 4000


def _frame(prediction):
    if isinstance(prediction, dict):
        return next(iter(prediction.values()))
    return prediction


def expected_unweighted_draw(forest, Xr: pd.DataFrame, k: int, chunk: int = 400):
    """Midpoint-rule integral over u of predict(u, weighted_leaves=True)."""
    mids = list((np.arange(k) + 0.5) / k)
    out = np.empty(len(Xr))
    for s in range(0, len(Xr), chunk):
        q = np.asarray(forest.predict(Xr.iloc[s:s + chunk], quantiles=mids,
                                      weighted_leaves=True))
        out[s:s + chunk] = q.mean(axis=1)
    return out


def pool_sizes(forest, Xd, Xr):
    counts = qd.bootstrap_counts(forest, len(Xd))
    lf, lr = forest.apply(Xd), forest.apply(Xr)
    n = np.zeros(len(Xr))
    for t, est in enumerate(forest.estimators_):
        s0 = np.bincount(lf[:, t], weights=counts[t], minlength=est.tree_.node_count)
        n += s0[lr[:, t]]
    return n


def analyse_split(X_donor, y_donor, w_donor, X_rec, w_rec, *, seed, n_draws,
                  check_convergence):
    predictors = list(X_donor.columns)
    train = X_donor.copy()
    train["moop"] = np.asarray(y_donor, dtype=float)
    out: dict = {}
    for label, weights in (("unweighted", None), ("weighted", w_donor)):
        t0 = time.time()
        kwargs = {} if weights is None else {"weight_col": np.asarray(weights, float)}
        fitted = QRF(seed=seed, log_level="ERROR").fit(
            train, predictors=predictors, imputed_variables=["moop"],
            not_numeric_categorical=predictors, n_jobs=N_JOBS, **kwargs)
        inner = fitted.models["moop"]
        forest = inner.qrf
        Xd = inner._align_features(X_donor)
        Xr = inner._align_features(X_rec)
        draws = np.stack([_frame(fitted.predict(X_rec))["moop"].to_numpy()
                          for _ in range(n_draws)])
        res = {"realized": draw_stats(draws, w_rec)}
        params = forest.get_params()
        res["forest_params"] = {k: params[k] for k in (
            "n_estimators", "min_samples_leaf", "max_samples_leaf", "max_features",
            "max_depth", "bootstrap", "random_state")}
        mein = qd.meinshausen_mean(forest, Xd, y_donor, Xr,
                                   survey_weight=None if weights is None else weights)
        res["forest_mean"] = float((mein * w_rec).sum())
        if weights is None:
            exp = expected_unweighted_draw(forest, Xr, K_LEVELS)
            res["expected"] = float((exp * w_rec).sum())
            res["expected_levels"] = K_LEVELS
            if check_convergence:
                sub = slice(0, min(1000, len(Xr)))
                fine = expected_unweighted_draw(forest, Xr.iloc[sub], 4 * K_LEVELS)
                res["grid_check"] = {
                    "rows": int(sub.stop),
                    "levels_coarse": K_LEVELS,
                    "levels_fine": 4 * K_LEVELS,
                    "rel_diff_total": float(
                        (exp[sub] * w_rec[sub]).sum() / (fine * w_rec[sub]).sum() - 1),
                }
            n = pool_sizes(forest, Xd, Xr)
            res["pool_size"] = {"mean": float(n.mean()), "median": float(np.median(n)),
                                "p10": float(np.quantile(n, 0.1))}
        else:
            res["expected"] = res["forest_mean"]
        out[label] = res
        print(f"  pr219 {label} {time.time() - t0:.0f}s", flush=True)
    return out


def run_cps(folds, out_path, n_draws):
    data = np.load(HERE.parent / "results" / "cps_inputs.npz", allow_pickle=False)
    X = pd.DataFrame(data["X"], columns=[str(c) for c in data["columns"]])
    moop, weight, fold_of = data["moop"], data["weight"], data["folds"]
    from cps import SEED

    results = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["fold"] for r in results}
    for fold in folds:
        if fold in done:
            continue
        t0 = time.time()
        donor, rec = fold_of != fold, fold_of == fold
        out = analyse_split(
            X.loc[donor].reset_index(drop=True), moop[donor], weight[donor],
            X.loc[rec].reset_index(drop=True), weight[rec], seed=SEED + fold,
            n_draws=n_draws, check_convergence=(fold == folds[0]))
        out["truth"] = {"reported_total": float((moop[rec] * weight[rec]).sum()),
                        "reported_stats": draw_stats(moop[rec][None, :], weight[rec])}
        out["fold"] = fold
        results.append(out)
        out_path.write_text(json.dumps(results, indent=1))
        print(f"fold {fold} done in {time.time() - t0:.0f}s", flush=True)


def run_synth(design, reps, out_path, n_draws):
    import synth

    results = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["rep"] for r in results}
    for rep in range(reps):
        if rep in done:
            continue
        t0 = time.time()
        rng = np.random.default_rng([20261107, rep, 0 if design == "x_only" else 1])
        X, y, true_mean, eps = synth.population(rng)
        idx, w = synth.sample(rng, X, eps, synth.GAMMA[design])
        Xs, ys, ms = X.iloc[idx].reset_index(drop=True), y[idx], true_mean[idx]
        rec = rng.random(len(idx)) < 0.2
        out = analyse_split(
            Xs.loc[~rec].reset_index(drop=True), ys[~rec], w[~rec],
            Xs.loc[rec].reset_index(drop=True), w[rec], seed=1000 + rep,
            n_draws=n_draws, check_convergence=(rep == 0))
        out["truth"] = {"conditional_mean_total": float((ms[rec] * w[rec]).sum()),
                        "realized_stats": draw_stats(ys[rec][None, :], w[rec])}
        out["rep"] = rep
        out["design"] = design
        results.append(out)
        out_path.write_text(json.dumps(results, indent=1))
        print(f"rep {rep} ({design}) done in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cps", action="store_true")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--design", choices=["x_only", "informative"], default="x_only")
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.cps:
        run_cps(args.folds, args.out, args.draws)
    else:
        run_synth(args.design, args.reps, args.out, args.draws)


if __name__ == "__main__":
    main()
