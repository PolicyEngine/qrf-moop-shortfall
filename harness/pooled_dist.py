"""Distribution statistics of the CPS draws, pooled over the five folds.

``cps.json`` and ``pr219_cps.json`` record each fold's draw statistics, so
an average of their percentage gaps is a mean across folds, not the gap
in the pooled distribution. This script imputes every SPM unit once, from
the forest fit on the other four folds, and computes each statistic on the
pooled draws (unit-weighted, with the experiment's weighted quantile),
averaged over ``n_draws`` realizations, against the same statistic of the
reported values. It writes aggregates only.

``--code main`` (microimpute at e4cbe02):

- ``microimpute``: microimpute's own ``predict``, ten calls per fold. They
  must reproduce the experiment's cached draws when those exist.
- ``published``: the cached draws of the experiment run that
  ``cps.cached_run`` names, the published run unless a fresh run's
  results.json sits beside the cache. ``checks.experiment_run`` records
  which run.
- ``type7_exact``, ``type1_exact``: one exact uniform per row, read from the
  same one-sample-per-leaf pools with the linear quantile or the inverse
  ECDF. Both read the same uniforms.
- ``leafall_type1_exact``: the inverse ECDF at the same uniforms, on pools
  of every in-bag sample pooled by count (``max_samples_leaf=None``).

``--code pr219`` (PR 219's head 296e47d): PR 219's own ``predict``, fit
without and with weights.

Usage:
  python pooled_dist.py --code main --out ../results/pooled_dist_main.json
  python pooled_dist.py --code pr219 --out ../results/pooled_dist_pr219.json
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
ROOT = HERE.parent
PR219_ROOT = ROOT / "microimpute-pr219"

N_JOBS = 4
STATS = {"median": 0.5, "p75": 0.75, "p90": 0.9, "p95": 0.95, "p99": 0.99}


def pooled_stats(draws: np.ndarray, weight: np.ndarray) -> dict:
    """Mean over realizations of each pooled statistic, plus the totals' SD."""
    import qrfdecomp as qd

    out = {k: [] for k in ("total", "share_zero", *STATS)}
    for d in draws:
        out["total"].append(float((d * weight).sum()))
        out["share_zero"].append(float(np.average(d == 0, weights=weight)))
        for name, q in STATS.items():
            out[name].append(qd.weighted_quantile(d, weight, q))
    stats = {k: float(np.mean(v)) for k, v in out.items()}
    stats["total_sd"] = float(np.std(out["total"], ddof=1)) if len(draws) > 1 else 0.0
    stats["n_draws"] = int(len(draws))
    return stats


def frame(prediction):
    if isinstance(prediction, dict):
        return next(iter(prediction.values()))
    return prediction


def fit_qrf(QRF, X_donor, y_donor, *, seed, weights=None):
    predictors = list(X_donor.columns)
    train = X_donor.copy()
    train["moop"] = np.asarray(y_donor, dtype=float)
    kwargs = {} if weights is None else {"weight_col": np.asarray(weights, float)}
    return QRF(seed=seed, log_level="ERROR").fit(
        train, predictors=predictors, imputed_variables=["moop"],
        not_numeric_categorical=predictors, n_jobs=N_JOBS, **kwargs)


def main_folds(QRF, X, moop, fold_of, weight, seed0, n_draws, log):
    """Per-unit draws for every main cell, filled fold by fold."""
    import qrfdecomp as qd

    cps = {r["fold"]: r for r in json.loads((ROOT / "results" / "cps.json").read_text())}
    from cps import CACHE, cached_run

    run = cached_run(n_draws)
    tag = run["tag"]
    n = len(moop)
    cells = {k: np.full((n_draws, n), np.nan) for k in (
        "microimpute", "type7_exact", "type1_exact", "leafall_type1_exact")}
    published = np.full((n_draws, n), np.nan)
    have_published = True
    checks = {"experiment_run": run, "forest_rel_diff_vs_cps_json": [],
              "rerun_vs_published_max_abs_diff": []}
    for fold in range(5):
        t0 = time.time()
        donor, rec = fold_of != fold, fold_of == fold
        seed = seed0 + fold
        Xd_raw = X.loc[donor].reset_index(drop=True)
        Xr_raw = X.loc[rec].reset_index(drop=True)
        fitted = fit_qrf(QRF, Xd_raw, moop[donor], seed=seed)
        cells["microimpute"][:, rec] = np.stack(
            [frame(fitted.predict(Xr_raw))["moop"].to_numpy() for _ in range(n_draws)])
        inner = fitted.models["moop"]
        base = inner.qrf
        Xd = inner._align_features(Xd_raw)
        Xr = inner._align_features(Xr_raw)
        p1 = qd.pools_from_leaf_values(qd.leaf_values_single(base, Xr))
        pn = qd.pools_all_samples(base, Xd, np.asarray(moop[donor], float), Xr)
        # The refit must be microimpute's published forest: its exact
        # expected draw has to match cps.json.
        exact = float((qd.expectations(p1).type7_grid * weight[rec]).sum())
        checks["forest_rel_diff_vs_cps_json"].append(
            exact / cps[fold]["totals"]["none/leaf1"]["type7_grid"] - 1)
        rng = np.random.default_rng([seed, 77])
        for d in range(n_draws):
            u = rng.random(p1.n_rows)
            cells["type7_exact"][d, rec] = p1.quantile(u, "linear")
            cells["type1_exact"][d, rec] = p1.quantile(u, "inverted_cdf")
            cells["leafall_type1_exact"][d, rec] = pn.quantile(u, "inverted_cdf")
        path = CACHE / f"{tag}_fold{fold}.npz"
        if path.exists():
            pub = np.load(path, allow_pickle=False)["qrf"]
            if pub.shape[0] < n_draws or pub.shape[1] != int(rec.sum()):
                raise ValueError(f"{path} holds draws of shape {pub.shape}, "
                                 f"need at least ({n_draws}, {int(rec.sum())})")
            published[:, rec] = pub[:n_draws]
            checks["rerun_vs_published_max_abs_diff"].append(
                float(np.max(np.abs(pub[:n_draws] - cells["microimpute"][:, rec]))))
        else:
            have_published = False
        log(f"fold {fold} done in {time.time() - t0:.0f}s")
    if have_published:
        cells["published"] = published
    return cells, checks


def pr219_folds(QRF, X, moop, fold_of, weight, seed0, n_draws, log):
    """Per-unit draws from PR 219's own predict, without and with weights."""
    ref = {r["fold"]: r for r in json.loads(
        (ROOT / "results" / "pr219_cps.json").read_text())}
    n = len(moop)
    cells = {k: np.full((n_draws, n), np.nan) for k in ("unweighted", "weighted")}
    checks = {"realized_total_rel_diff_vs_pr219_cps_json": []}
    for fold in range(5):
        t0 = time.time()
        donor, rec = fold_of != fold, fold_of == fold
        Xd_raw = X.loc[donor].reset_index(drop=True)
        Xr_raw = X.loc[rec].reset_index(drop=True)
        for label, weights in (("unweighted", None), ("weighted", weight[donor])):
            fitted = fit_qrf(QRF, Xd_raw, moop[donor], seed=seed0 + fold, weights=weights)
            draws = np.stack([frame(fitted.predict(Xr_raw))["moop"].to_numpy()
                              for _ in range(n_draws)])
            cells[label][:, rec] = draws
            total = float(np.mean((draws * weight[rec]).sum(axis=1)))
            checks["realized_total_rel_diff_vs_pr219_cps_json"].append(
                total / ref[fold][label]["realized"]["total"] - 1)
        log(f"fold {fold} done in {time.time() - t0:.0f}s")
    return cells, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", choices=["main", "pr219"], required=True)
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(HERE))
    if args.code == "pr219":
        sys.path.insert(0, str(PR219_ROOT))
    else:
        sys.path.insert(0, str(ROOT / "microimpute"))
    import microimpute.models.qrf as qrf_module

    expected_root = PR219_ROOT if args.code == "pr219" else ROOT / "microimpute"
    assert Path(qrf_module.__file__).resolve().is_relative_to(expected_root.resolve()), (
        qrf_module.__file__)
    from cps import SEED

    data = np.load(ROOT / "results" / "cps_inputs.npz", allow_pickle=False)
    X = pd.DataFrame(data["X"], columns=[str(c) for c in data["columns"]])
    moop, weight, fold_of = data["moop"], data["weight"], data["folds"]
    log = lambda msg: print(msg, flush=True)  # noqa: E731
    run = main_folds if args.code == "main" else pr219_folds
    cells, checks = run(qrf_module.QRF, X, moop, fold_of, weight, SEED, args.draws, log)
    for name, draws in cells.items():
        assert not np.isnan(draws).any(), f"{name} left a unit without a draw"
    out = {
        "code": args.code,
        "n_units": int(len(moop)),
        "reported": pooled_stats(moop[None, :], weight),
        "cells": {name: pooled_stats(draws, weight) for name, draws in cells.items()},
        "checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
