"""Run every decomposition cell for one donor/recipient split.

The baseline cell is microimpute's own fitted forest (``QRF.fit`` with the
experiment's arguments), so its grid/type-7/one-sample-leaf cell is the
published model. The weighted forests reuse that forest's ``random_state``:

- ``split``: ``sample_weight`` passed to quantile-forest. Under
  scikit-learn 1.8 that weights the split criterion only; leaf values stay
  unweighted (what microimpute ``main`` does with ``weight_col``).
- ``boot``: rows resampled n-of-n in proportion to weight, then an
  unweighted forest (what Microcosm's ``RegimeGatedQRF`` does).
- ``ecdf``: the ``split`` forest read through survey-weighted leaf ECDFs
  normalized per tree and inverted without interpolation (what
  microimpute PR 219 does when given weights). Its expected draw is
  :func:`qrfdecomp.meinshausen_mean` with the survey weights.

For each forest and leaf setting, every draw mechanism's expected total is
exact given the fitted trees. Realized draws (``n_draws`` per cell) give
the distribution statistics the experiment reports.
"""

from __future__ import annotations

import copy
import os
import time

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor

import qrfdecomp as qd

MECHANISMS = ["type7_grid", "type1_grid", "type7_exact", "type1_exact", "type7_knots"]
QUANTILES = {"median": 0.5, "p75": 0.75, "p90": 0.9, "p95": 0.95, "p99": 0.99}


def _as_frame(prediction):
    if isinstance(prediction, dict):
        return next(iter(prediction.values()))
    return prediction


def draw_stats(draws: np.ndarray, weight: np.ndarray) -> dict:
    """The experiment's distribution statistics, averaged over draws.

    ``total_sd`` is the spread of the total across draws, so the Monte Carlo
    SE of ``total`` is ``total_sd / sqrt(n_draws)``.
    """
    out = {"total": [], "share_zero": []}
    for name in QUANTILES:
        out[name] = []
    for d in draws:
        out["total"].append(float((d * weight).sum()))
        out["share_zero"].append(float(np.average(d == 0, weights=weight)))
        for name, q in QUANTILES.items():
            out[name].append(qd.weighted_quantile(d, weight, q))
    stats = {k: float(np.mean(v)) for k, v in out.items()}
    stats["total_sd"] = float(np.std(out["total"], ddof=1)) if len(draws) > 1 else 0.0
    stats["n_draws"] = int(len(draws))
    return stats


def honest_mean(
    X: pd.DataFrame,
    y: np.ndarray,
    X_rec: pd.DataFrame,
    *,
    params: dict,
    seed: int,
    survey_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Cross-fitted honest forest: splits from one half, leaf values from the other.

    Each half grows a forest (bootstrap as usual); the other half's donors
    are routed through its trees and fill the leaves. A tree whose leaf for a
    recipient holds no donor from the filling half is skipped for that
    recipient. The two halves' tree-averaged leaf means are averaged. With
    ``survey_weight``, leaf means are survey-weighted. The partition never
    sees the filling half's targets, so the splits can't tie a donor's
    weight in the recipients' means to its own noise draw. Survey weights
    still can: in the informative synthetic design, inclusion depends on
    each unit's noise (``synth.py``), so the weights carry it.
    """
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng([seed, 31])
    half = rng.random(len(y)) < 0.5
    sw = np.ones(len(y)) if survey_weight is None else np.asarray(survey_weight, float)
    means = []
    for grow in (half, ~half):
        fill = ~grow
        model = RandomForestQuantileRegressor(**{**params, "max_samples_leaf": 1})
        model.fit(X.loc[grow], y[grow])
        leaves_fill = model.apply(X.loc[fill])
        leaves_rec = model.apply(X_rec)
        total = np.zeros(len(X_rec))
        used = np.zeros(len(X_rec))
        for t in range(model.n_estimators):
            n_nodes = model.estimators_[t].tree_.node_count
            s0 = np.bincount(leaves_fill[:, t], weights=sw[fill], minlength=n_nodes)
            s1 = np.bincount(leaves_fill[:, t], weights=sw[fill] * y[fill],
                             minlength=n_nodes)
            lr = leaves_rec[:, t]
            ok = s0[lr] > 0
            total[ok] += s1[lr][ok] / s0[lr][ok]
            used[ok] += 1
        if (used == 0).any():
            raise RuntimeError("a recipient found no filling donor in any tree")
        means.append(total / used)
    return 0.5 * (means[0] + means[1])


def _cell(pools, weight, rng, n_draws, knots):
    exp = qd.expectations(pools, knots).as_dict()
    totals = {m: float((v * weight).sum()) for m, v in exp.items()}
    realized = {}
    for m in MECHANISMS:
        draws = np.stack([qd.draw(pools, rng.random(pools.n_rows), m, knots)
                          for _ in range(n_draws)])
        realized[m] = draw_stats(draws, weight)
    return totals, realized, exp


def decompose(
    X_donor: pd.DataFrame,
    y_donor: np.ndarray,
    w_donor: np.ndarray,
    X_rec: pd.DataFrame,
    w_rec: np.ndarray,
    *,
    seed: int,
    n_draws: int = 10,
    n_jobs: int = 4,
    microcosm: bool = True,
    log=print,
) -> dict:
    from microimpute.models.qrf import QRF

    knots = qd.microcosm_grid()
    predictors = list(X_donor.columns)
    rng = np.random.default_rng([seed, 99])
    out: dict = {"n_donor": len(X_donor), "n_rec": len(X_rec)}

    # Baseline: microimpute's own forest, fit exactly as the experiment does.
    t0 = time.time()
    train = X_donor.copy()
    train["moop"] = y_donor
    fitted = QRF(seed=seed, log_level="ERROR").fit(
        train,
        predictors=predictors,
        imputed_variables=["moop"],
        not_numeric_categorical=predictors,
        n_jobs=n_jobs,
    )
    inner = fitted.models["moop"]
    base = inner.qrf
    Xd = inner._align_features(X_donor)
    Xr = inner._align_features(X_rec)
    rng_copy = copy.deepcopy(inner._rng)
    mi_draws = np.stack(
        [_as_frame(fitted.predict(X_rec))["moop"].to_numpy() for _ in range(n_draws)]
    )
    out["microimpute_realized"] = draw_stats(mi_draws, w_rec)
    out["microimpute_draws_first"] = mi_draws[0]
    log(f"  microimpute fit+draws {time.time() - t0:.0f}s")

    # Replay microimpute's draws from the pools: must match its predict.
    pools1 = qd.pools_from_leaf_values(qd.leaf_values_single(base, Xr))
    u = rng_copy.beta(1.0, 1.0, size=len(Xr))
    replay = qd.draw(pools1, u, "type7_grid")
    out["replay_max_abs_diff"] = float(np.max(np.abs(replay - mi_draws[0])))

    params = dict(n_estimators=base.n_estimators, random_state=base.random_state,
                  n_jobs=n_jobs, max_samples_leaf=1)
    forests = {"none": (base, Xd, np.asarray(y_donor, float))}

    t0 = time.time()
    split = RandomForestQuantileRegressor(**params).fit(Xd, y_donor, sample_weight=w_donor)
    forests["split"] = (split, Xd, np.asarray(y_donor, float))
    picked = qd.weighted_bootstrap(len(Xd), w_donor, np.random.default_rng([seed, 7]))
    Xb = Xd.iloc[picked].reset_index(drop=True)
    yb = np.asarray(y_donor, float)[picked]
    boot = RandomForestQuantileRegressor(**params).fit(Xb, yb)
    forests["boot"] = (boot, Xb, yb)
    log(f"  weighted forests {time.time() - t0:.0f}s")

    out["totals"] = {}
    out["realized"] = {}
    out["pool_size"] = {}
    for name, (model, Xt, yt) in forests.items():
        t0 = time.time()
        p1 = qd.pools_from_leaf_values(qd.leaf_values_single(model, Xr))
        pn = qd.pools_all_samples(model, Xt, yt, Xr)
        for leaf, pools in (("leaf1", p1), ("leafall", pn)):
            totals, realized, _ = _cell(pools, w_rec, rng, n_draws, knots)
            out["totals"][f"{name}/{leaf}"] = totals
            out["realized"][f"{name}/{leaf}"] = realized
            out["pool_size"][f"{name}/{leaf}"] = {
                "mean": float(np.mean(pools.n)),
                "median": float(np.median(pools.n)),
                "p10": float(np.quantile(pools.n, 0.1)),
            }
        # Tree-average (Meinshausen) means: the expected mean of a one-sample
        # leaf forest, and with survey weights PR 219's weighted ECDF.
        mein = qd.meinshausen_mean(model, Xt, yt, Xr)
        out["totals"][f"{name}/meinshausen"] = float((mein * w_rec).sum())
        if name == "split":
            ecdf = qd.meinshausen_mean(model, Xt, yt, Xr, survey_weight=w_donor)
            out["totals"]["ecdf/type1_exact"] = float((ecdf * w_rec).sum())
        log(f"  cells for {name} {time.time() - t0:.0f}s")

    # Honest forests: is the forest's own mean biased by catchment selection?
    t0 = time.time()
    honest_params = {k: v for k, v in params.items() if k != "max_samples_leaf"}
    h0 = honest_mean(Xd, y_donor, Xr, params=honest_params, seed=seed)
    hw = honest_mean(Xd, y_donor, Xr, params=honest_params, seed=seed,
                     survey_weight=w_donor)
    out["totals"]["honest/meinshausen"] = float((h0 * w_rec).sum())
    out["totals"]["honest_weighted/meinshausen"] = float((hw * w_rec).sum())
    log(f"  honest forests {time.time() - t0:.0f}s")

    if microcosm:
        out["microcosm"] = {}
        from microcosm.fit import NO_WEIGHTS, RegimeGatedQRF

        # microcosm-fit at 7080ba8 reads its thread caps from these variables
        # (microcosm/fit/qrf.py:905-910).
        os.environ["POPULACE_FIT_N_JOBS"] = str(n_jobs)
        os.environ["POPULACE_FIT_PREDICT_WORKERS"] = str(n_jobs)
        for label, weights in (("weighted", w_donor), ("unweighted", NO_WEIGHTS)):
            t0 = time.time()
            model = RegimeGatedQRF(seed=seed).fit(train, predictors, ["moop"],
                                                  weights=weights)
            target = model._target_models["moop"]
            forest = target.positive.model
            cols = list(target.positive.columns)
            Xrc = X_rec[cols].to_numpy(dtype=np.float64)
            q = np.asarray(forest.predict(Xrc, quantiles=list(knots)))
            amount_knots = q @ qd.interp_weights(knots)
            classes = list(target.gate.classes_)
            proba = np.asarray(target.gate.predict_proba(
                X_rec[list(target.columns)].to_numpy(dtype=np.float64)))
            p_pos = proba[:, classes.index(1)]
            amount_mean = np.asarray(forest.predict(Xrc, quantiles="mean"))
            draws = np.stack([model.predict(X_rec)["moop"].to_numpy()
                              for _ in range(n_draws)])
            out["microcosm"][label] = {
                "expected_total": float((p_pos * amount_knots * w_rec).sum()),
                # Same gate, same forest, amount read as the pool mean
                # (an inverse-ECDF draw) instead of type 7 on the knots.
                "expected_total_pool_mean": float((p_pos * amount_mean * w_rec).sum()),
                "expected_positive_share": float(np.average(p_pos, weights=w_rec)),
                "realized": draw_stats(draws, w_rec),
            }
            log(f"  microcosm {label} {time.time() - t0:.0f}s")
    return out
