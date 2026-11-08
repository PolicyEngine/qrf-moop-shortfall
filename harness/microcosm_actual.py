"""Decompose Microcosm's own amount forest on the synthetic process.

``microcosm_bias.py`` rebuilds the amount forest the way Microcosm does
and compares adaptive with honest fills. This script checks that the
rebuilt forest stands for the real one. For each replication it refits
``RegimeGatedQRF`` exactly as ``cells.decompose`` did (same donors, weights
and seed), then:

1. reproduces ``synth_<design>.json``'s Microcosm pool-mean total (gate
   probability times the amount forest's ``predict(quantiles="mean")``);
2. swaps the gate's probability for the true p(x), which isolates the
   amount forest from the gate;
3. recovers the forest's training rows from ``forest_.y_train`` and
   ``sorter_`` (quantile-forest keeps both), maps each row back to its
   positive donor by its value (the synthetic amounts are continuous, so
   values identify donors; asserted), replays each tree's bootstrap, and
   splits the true-p total into selection, informative sampling and
   smoothing exactly as ``microcosm_bias.py`` does. The count-pooled
   recipient estimates must equal the forest's own ``predict(quantiles=
   "mean")``.

Usage:
  python microcosm_actual.py --design x_only --reps 4 --out results/microcosm_actual_x_only.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qrfdecomp as qd  # noqa: E402
import synth  # noqa: E402
from forest_bias import fill_catchments, recipient_estimates  # noqa: E402
from microcosm_bias import structural  # noqa: E402

N_JOBS = 4


def analyse_split(X_donor, y_donor, w_donor, X_rec, w_rec, *, seed, gamma, eps_donor):
    from microcosm.fit import RegimeGatedQRF

    # microcosm-fit at 7080ba8 reads its thread caps from these variables
    # (microcosm/fit/qrf.py:905-910).
    os.environ["POPULACE_FIT_N_JOBS"] = str(N_JOBS)
    os.environ["POPULACE_FIT_PREDICT_WORKERS"] = str(N_JOBS)
    predictors = list(X_donor.columns)
    train = X_donor.copy()
    train["moop"] = y_donor
    estimator = RegimeGatedQRF(seed=seed)
    model = estimator.fit(train, predictors, ["moop"], weights=w_donor)
    target = model._target_models["moop"]
    forest = target.positive.model
    cols = list(target.positive.columns)
    Xr = X_rec[cols].to_numpy(dtype=np.float64)
    amount_mean = np.asarray(forest.predict(Xr, quantiles="mean"), dtype=np.float64)
    classes = list(target.gate.classes_)
    p_gate = np.asarray(target.gate.predict_proba(
        X_rec[list(target.columns)].to_numpy(dtype=np.float64)))[:, classes.index(1)]
    p_true, mplus_rec, _ = structural(X_rec)

    # The amount forest's training rows: a weighted bootstrap of the
    # positive donors, recovered from the stored, sorted targets.
    pos = y_donor > estimator.zero_atol
    y_pos = np.asarray(y_donor[pos], dtype=np.float64)
    X_pos = X_donor.loc[pos, cols].to_numpy(dtype=np.float64)
    eps_pos = eps_donor[pos]
    _, mplus, sigma = structural(X_donor.loc[pos].reset_index(drop=True))
    mstar = mplus * np.exp(-gamma * sigma)
    y_sorted = np.asarray(forest.forest_.y_train)[0]
    sorter = np.asarray(forest.sorter_)[:, 0]
    y_fit = np.empty_like(y_sorted)
    y_fit[sorter] = y_sorted
    order = np.argsort(y_pos, kind="stable")
    assert np.all(np.diff(y_pos[order]) > 0), "positive donor values are not unique"
    loc = np.searchsorted(y_pos[order], y_fit)
    rows_to_donor = order[loc]
    assert np.array_equal(y_pos[rows_to_donor], y_fit), "a training row matches no donor"
    x_fit = X_pos[rows_to_donor]
    n_fit, n_pos = len(y_fit), len(y_pos)

    counts = qd.bootstrap_counts(forest, n_fit)
    nodes = [e.tree_.node_count for e in forest.estimators_]
    lf, lr = forest.apply(x_fit), forest.apply(Xr)

    scale = w_rec * p_true
    truth = float((scale * mplus_rec).sum())
    out = {
        "truth": truth,
        "n_positive_donors": int(n_pos),
        "n_distinct_donors_in_forest": int(len(np.unique(rows_to_donor))),
        "total_gate_pool_mean": float((w_rec * p_gate * amount_mean).sum()),
        "total_truep_pool_mean": float((scale * amount_mean).sum()),
        "gate_positive_share": float(np.average(p_gate, weights=w_rec)),
        "true_positive_share": float(np.average(p_true, weights=w_rec)),
        "totals": {}, "split": {}, "catchment_profile": {},
    }
    q90, q99 = np.quantile(eps_pos, [0.9, 0.99])
    for mode in ("pooled", "tree"):
        c_rows = fill_catchments(lf, counts, lr, scale, nodes, mode)
        c = np.bincount(rows_to_donor, weights=c_rows, minlength=n_pos)
        est = recipient_estimates(lf, counts, y_fit, lr, nodes, mode)
        if mode == "pooled":
            out["pooled_estimate_vs_predict_mean_max_rel_diff"] = float(
                np.max(np.abs(est / amount_mean - 1)))
        total = float((c * y_pos).sum())
        out[f"catchment_identity_rel_err_{mode}"] = abs(total / float((est * scale).sum()) - 1)
        out["totals"][mode] = total
        out["split"][mode] = {
            "selection": float((c * (y_pos - mstar)).sum()),
            "informative": float((c * (mstar - mplus)).sum()),
            "smoothing": float((c * mplus).sum() - truth),
        }
        cbar = c.mean()
        out["catchment_profile"][mode] = {
            "top10_noise": float(c[eps_pos > q90].mean() / cbar),
            "top1_noise": float(c[eps_pos > q99].mean() / cbar),
            "bottom50_noise": float(c[eps_pos < np.median(eps_pos)].mean() / cbar),
        }
    return out


def run(design, reps, out_path):
    gamma = synth.GAMMA[design]
    saved = {r["rep"]: r for r in json.loads(
        (HERE.parent / "results" / f"synth_{design}.json").read_text())}
    results = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["rep"] for r in results}
    for rep in range(reps):
        if rep in done:
            continue
        t0 = time.time()
        # Same stream and split as synth.run_rep.
        rng = np.random.default_rng([20261107, rep, 0 if design == "x_only" else 1])
        X, y, true_mean, eps = synth.population(rng)
        idx, w = synth.sample(rng, X, eps, gamma)
        Xs, ys, es = X.iloc[idx].reset_index(drop=True), y[idx], eps[idx]
        rec = rng.random(len(idx)) < 0.2
        out = analyse_split(
            Xs.loc[~rec].reset_index(drop=True), ys[~rec], w[~rec],
            Xs.loc[rec].reset_index(drop=True), w[rec], seed=1000 + rep, gamma=gamma,
            eps_donor=es[~rec])
        ref = saved[rep]["microcosm"]["weighted"]["expected_total_pool_mean"]
        out["synth_json_pool_mean_total"] = ref
        out["reproduces_synth_json_rel_diff"] = out["total_gate_pool_mean"] / ref - 1
        out["truth_overall"] = float((true_mean[idx][rec] * w[rec]).sum())
        out["rep"], out["design"] = rep, design
        results.append(out)
        out_path.write_text(json.dumps(results, indent=1))
        print(f"rep {rep} ({design}) done in {time.time() - t0:.0f}s; reproduces synth json "
              f"to {out['reproduces_synth_json_rel_diff']:.1e}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", choices=["x_only", "informative"], default="x_only")
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run(args.design, args.reps, args.out)


if __name__ == "__main__":
    main()
