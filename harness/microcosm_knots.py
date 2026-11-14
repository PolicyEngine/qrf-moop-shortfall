"""Split Microcosm's draw loss on CPS into the linear quantile and the knots.

``cells.py`` compares Microcosm's expected draw, linear quantiles read at
203 knots and interpolated at each row's uniform, with the amount forest's
pool mean. That gap combines two steps: the linear (type 7) quantile of
each row's finite pool, which under an exact uniform integrates to
``(sum z - (z_min + z_max)/2)/(N - 1)``, and the interpolation between
knots that stands in for that quantile function. For each fold and both
weightings this refits ``RegimeGatedQRF`` exactly as ``cells.decompose``
did (same donors, weights and seed), rebuilds each recipient's
count-expanded pool from the amount forest's stored leaf samples
(``forest_.y_train_leaves`` holds every in-bag sample's index into the
sorted ``forest_.y_train``, bootstrap repeats included), and computes per
row:

- the pool mean, which must match ``predict(quantiles="mean")``;
- the exact type-7 expectation (``qrfdecomp.type7_integral``);
- the 203-knot expectation from the pool, which must match the same
  expectation from ``predict(quantiles=knots)``, both integrated with
  ``qrfdecomp.interp_weights`` before the comparison.

``pool_size_median`` records each fold's median pool size.

Each total weights the amount by the gate's positive probability and the
recipient's weight, as ``cells.py`` does, and must reproduce
``cps.json``'s Microcosm totals.

Usage:
  python microcosm_knots.py --out ../results/microcosm_knots_cps.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qrfdecomp as qd  # noqa: E402

N_JOBS = 4


def stored_pools(forest, X_rec: np.ndarray) -> qd.RowPools:
    """Each recipient's count-expanded pool from the forest's stored leaves."""
    leaves = forest.apply(X_rec)
    stored = np.asarray(forest.forest_.y_train_leaves)
    y_sorted = np.asarray(forest.forest_.y_train)[0]
    rows_parts, value_parts = [], []
    for t in range(forest.n_estimators):
        idx = stored[t, leaves[:, t], 0, :]
        r, c = np.nonzero(idx)
        rows_parts.append(r)
        value_parts.append(y_sorted[idx[r, c] - 1])
    rows = np.concatenate(rows_parts)
    values = np.concatenate(value_parts)
    order = np.lexsort((values, rows))
    rows, values = rows[order], values[order]
    m = len(X_rec)
    offset = np.searchsorted(rows, np.arange(m + 1), side="left")
    counts = np.ones(len(values), dtype=np.int64)
    n = np.diff(offset)
    if (n == 0).any():
        raise RuntimeError("a recipient's pool is empty")
    return qd.RowPools(values, counts, offset, np.cumsum(counts),
                       offset[:-1].astype(np.int64), n)


def analyse_fold(X_donor, y_donor, w_donor, X_rec, w_rec, *, seed, knots):
    from microcosm.fit import NO_WEIGHTS, RegimeGatedQRF

    # microcosm-fit at 7080ba8 reads its thread caps from these variables
    # (microcosm/fit/qrf.py:905-910).
    os.environ["POPULACE_FIT_N_JOBS"] = str(N_JOBS)
    os.environ["POPULACE_FIT_PREDICT_WORKERS"] = str(N_JOBS)
    predictors = list(X_donor.columns)
    train = X_donor.copy()
    train["moop"] = y_donor
    iw = qd.interp_weights(knots)
    out = {}
    for label, weights in (("weighted", w_donor), ("unweighted", NO_WEIGHTS)):
        model = RegimeGatedQRF(seed=seed).fit(train, predictors, ["moop"], weights=weights)
        target = model._target_models["moop"]
        forest = target.positive.model
        Xrc = X_rec[list(target.positive.columns)].to_numpy(dtype=np.float64)
        classes = list(target.gate.classes_)
        p_pos = np.asarray(target.gate.predict_proba(
            X_rec[list(target.columns)].to_numpy(dtype=np.float64)))[:, classes.index(1)]
        scale = p_pos * w_rec

        pools = stored_pools(forest, Xrc)
        mean = pools.mean()
        type7 = qd.type7_integral(pools)
        # Integrated knot expectations, one per row: the check below compares
        # these, not the 203 knots one by one.
        knots_pool = pools.quantiles(knots, "linear") @ iw
        mean_predict = np.asarray(forest.predict(Xrc, quantiles="mean"), dtype=np.float64)
        knots_predict = np.asarray(forest.predict(Xrc, quantiles=list(knots))) @ iw
        out[label] = {
            "expected_positive_share": float(np.average(p_pos, weights=w_rec)),
            "total_pool_mean": float((scale * mean).sum()),
            "total_type7_exact": float((scale * type7).sum()),
            "total_knots": float((scale * knots_pool).sum()),
            "pool_size_median": float(np.median(pools.n)),
            "pool_mean_vs_predict_max_rel_diff": float(np.max(np.abs(mean / mean_predict - 1))),
            "knots_vs_predict_max_rel_diff": float(
                np.max(np.abs(knots_pool - knots_predict) / np.abs(knots_predict))),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from cps import SEED

    knots = qd.microcosm_grid()
    data = np.load(HERE.parent / "results" / "cps_inputs.npz", allow_pickle=False)
    X = pd.DataFrame(data["X"], columns=[str(c) for c in data["columns"]])
    moop, weight, fold_of = data["moop"], data["weight"], data["folds"]
    cps = {r["fold"]: r for r in json.loads((HERE.parent / "results" / "cps.json").read_text())}
    results = json.loads(args.out.read_text()) if args.out.exists() else []
    done = {r["fold"] for r in results}
    for fold in args.folds:
        if fold in done:
            continue
        t0 = time.time()
        donor, rec = fold_of != fold, fold_of == fold
        out = analyse_fold(X.loc[donor].reset_index(drop=True), moop[donor], weight[donor],
                           X.loc[rec].reset_index(drop=True), weight[rec], seed=SEED + fold,
                           knots=knots)
        for label in ("weighted", "unweighted"):
            ref = cps[fold]["microcosm"][label]
            out[label]["reproduces_cps_json_knots_rel_diff"] = (
                out[label]["total_knots"] / ref["expected_total"] - 1)
            out[label]["reproduces_cps_json_pool_mean_rel_diff"] = (
                out[label]["total_pool_mean"] / ref["expected_total_pool_mean"] - 1)
        out["reported_total"] = float((moop[rec] * weight[rec]).sum())
        out["weight_total"] = float(weight[rec].sum())
        out["fold"] = fold
        results.append(out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
        print(f"fold {fold} done in {time.time() - t0:.0f}s; reproduces cps.json to "
              f"{out['weighted']['reproduces_cps_json_knots_rel_diff']:.1e} (knots), "
              f"{out['weighted']['reproduces_cps_json_pool_mean_rel_diff']:.1e} (pool mean)",
              flush=True)


if __name__ == "__main__":
    main()
