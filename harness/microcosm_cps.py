"""Same-tree selection test of Microcosm's amount forest on CPS MOOP.

On the synthetic process Microcosm's amount forest carries about +11
points of selection (``microcosm_actual.py``). CPS has no known
conditional mean, so the exact split isn't available; the same-tree
comparison is. For each fold this:

1. refits ``RegimeGatedQRF`` exactly as ``cells.decompose`` did (same donors,
   weights and seed) and reproduces ``cps.json``'s Microcosm pool-mean total
   (gate probability times the amount forest's ``predict(quantiles="mean")``);
2. grows forests the way Microcosm grows its amount forest
   (``microcosm_bias.forest``) on a weighted bootstrap of one random half of
   the positive donors, and fills the same trees two ways: with their own
   in-bag rows (adaptive) and with the other half's donors, each carrying
   its survey weight (honest); averaged over the two halves.

Every total weights each recipient's amount by the fitted gate's positive
probability, the same in both fills, so adaptive minus honest isolates
what the splits' sight of the donors' own values does to the total.

Usage:
  python microcosm_cps.py --out results/microcosm_cps.json
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
from microcosm_bias import fill, forest  # noqa: E402

N_JOBS = 4


def analyse_fold(X_donor, y_donor, w_donor, X_rec, w_rec, *, seed):
    from microcosm.fit import RegimeGatedQRF

    os.environ["POPULACE_FIT_N_JOBS"] = str(N_JOBS)
    os.environ["POPULACE_FIT_PREDICT_WORKERS"] = str(N_JOBS)
    predictors = list(X_donor.columns)
    train = X_donor.copy()
    train["moop"] = y_donor
    estimator = RegimeGatedQRF(seed=seed)
    model = estimator.fit(train, predictors, ["moop"], weights=w_donor)
    target = model._target_models["moop"]
    own = target.positive.model
    cols = list(target.positive.columns)
    Xr = X_rec[cols].to_numpy(dtype=np.float64)
    classes = list(target.gate.classes_)
    p_gate = np.asarray(target.gate.predict_proba(
        X_rec[list(target.columns)].to_numpy(dtype=np.float64)))[:, classes.index(1)]
    amount_mean = np.asarray(own.predict(Xr, quantiles="mean"), dtype=np.float64)
    scale = w_rec * p_gate

    pos = y_donor > estimator.zero_atol
    X_pos = X_donor.loc[pos, cols].to_numpy(dtype=np.float64)
    y_pos = np.asarray(y_donor[pos], dtype=np.float64)
    w_pos = np.asarray(w_donor[pos], dtype=np.float64)
    n = len(y_pos)
    totals = {k: 0.0 for k in ("ahalf/tree", "ahalf/pooled", "honest/tree", "honest/pooled")}
    half = np.random.default_rng([seed, 31]).random(n) < 0.5
    for k, grow in enumerate((half, ~half)):
        idx_g, idx_f = np.flatnonzero(grow), np.flatnonzero(~grow)
        pick_g = idx_g[qd.weighted_bootstrap(len(idx_g), w_pos[idx_g],
                                             np.random.default_rng([seed, 7, k]))]
        model_h = forest(X_pos[pick_g], y_pos[pick_g], seed)
        cnt_g = qd.bootstrap_counts(model_h, len(pick_g))
        adaptive = fill(model_h, X_pos[pick_g], cnt_g, y_pos[pick_g], Xr, scale, pick_g, n)
        mass_f = np.broadcast_to(w_pos[idx_f] / w_pos[idx_f].mean(),
                                 (model_h.n_estimators, len(idx_f)))
        honest = fill(model_h, X_pos[idx_f], mass_f, y_pos[idx_f], Xr, scale, idx_f, n)
        for mode in ("tree", "pooled"):
            totals[f"ahalf/{mode}"] += 0.5 * adaptive[mode][1]
            totals[f"honest/{mode}"] += 0.5 * honest[mode][1]
    return {
        "own_gate_pool_mean": float((scale * amount_mean).sum()),
        "gate_positive_share": float(np.average(p_gate, weights=w_rec)),
        "n_positive_donors": int(n),
        "totals": totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from cps import SEED

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
                           X.loc[rec].reset_index(drop=True), weight[rec], seed=SEED + fold)
        ref = cps[fold]["microcosm"]["weighted"]["expected_total_pool_mean"]
        out["cps_json_pool_mean_total"] = ref
        out["reproduces_cps_json_rel_diff"] = out["own_gate_pool_mean"] / ref - 1
        out["reported_total"] = float((moop[rec] * weight[rec]).sum())
        out["fold"] = fold
        results.append(out)
        args.out.write_text(json.dumps(results, indent=1))
        print(f"fold {fold} done in {time.time() - t0:.0f}s; reproduces cps.json to "
              f"{out['reproduces_cps_json_rel_diff']:.1e}", flush=True)


if __name__ == "__main__":
    main()
