"""Does Microcosm's amount forest carry the synthetic selection bias?

On the synthetic process, Microcosm's RegimeGatedQRF lands 8-9% above the
truth, close to microimpute's fully grown forest, whose excess
``forest_bias.py`` traces to selection: adaptive splits give donors with
large positive noise more catchment. This script runs the same exact split
on a forest built the way Microcosm builds its amount forest
(``microcosm/fit/qrf.py:1018-1036``):

- positive donors only (the gate handles zeros);
- a weighted bootstrap: n rows drawn with replacement in proportion to
  weight (``_weighted_bootstrap``, ``qrf.py:265-294``; ``qd.weighted_bootstrap``
  draws the same way);
- ``RandomForestQuantileRegressor(n_estimators=100, max_samples_leaf=None)``
  with every other parameter at its default, so trees grow until leaves
  are pure;
- read by ``predict`` with ``weighted_leaves`` at its default (False,
  ``qrf.py:855``), which pools every tree's in-bag samples by count. The
  pool mean is the expected draw under an inverse ECDF; Microcosm's
  type-7 knots shave a little off it (``cells.py`` measures that).

To isolate the amount forest from the gate, each recipient's amount is
weighted by its true positive probability p(x). The aggregate then equals
sum_j c_j y_j over positive donors, with c_j = sum_i w_i p_i a_ij, and its
error against the truth sum_i w_i p_i m+(x_i) splits exactly into

    sum_j c_j (y_j - m*_j)      selection
  + sum_j c_j (m*_j - m+_j)     informative sampling
  + sum_j c_j m+_j - truth      smoothing

where m+ = exp(mu + sigma^2 / 2) is the positive amount's conditional mean
and m* = m+ exp(-gamma sigma) its mean among sampled units. In the
informative design the weights carry each unit's noise, so the weighted
bootstrap's correction of the sampling shows up as positive selection
(about +1.3 points) even under honest fills.

Estimators (``tree`` = per-tree leaf means; ``pooled`` = count pooling,
Microcosm's reading):

- full/*: the Microcosm-style forest on the whole weighted bootstrap,
  filled with its own in-bag rows.
- ahalf/*: forests grown on a weighted bootstrap of one random half of the
  positive donors, filled with their own in-bag rows; averaged over halves.
- honest/*: the same half-forests filled with the other half's donors,
  each carrying its survey weight as mass, values the splits never saw.

Usage:
  python microcosm_bias.py --design x_only --reps 4 --out results/microcosm_bias_x_only.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from quantile_forest import RandomForestQuantileRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qrfdecomp as qd  # noqa: E402
import synth  # noqa: E402
from forest_bias import fill_catchments, recipient_estimates  # noqa: E402

N_JOBS = 4
N_ESTIMATORS = 100  # microcosm.fit.qrf.DEFAULT_N_ESTIMATORS


def structural(X):
    """p(x), m+(x) and sigma(x) from synth.population's own formulas."""
    age, inc = X["age"].to_numpy(), X["inc"].to_numpy()
    size, disab = X["size"].to_numpy(), X["disab"].to_numpy()
    group = X["group"].to_numpy().astype(int)
    lin = 2.6 + 1.2 * age + 0.3 * inc + 0.25 * (size - 1) + synth.G0[group] + 0.5 * disab
    p_pos = 1.0 / (1.0 + np.exp(-lin))
    mu = 7.5 + 0.9 * age + 0.35 * inc + 0.12 * (size - 1) + synth.GM[group] + 0.4 * disab
    sigma = 0.95 + 0.3 * age + 0.1 * disab
    return p_pos, np.exp(mu + sigma**2 / 2.0), sigma


def forest(X, y, seed):
    model = RandomForestQuantileRegressor(n_estimators=N_ESTIMATORS, max_samples_leaf=None,
                                          random_state=seed, n_jobs=N_JOBS)
    return model.fit(X, y)


def fill(model, X_fill, mass, y_fill, X_rec, scale_rec, rows_to_donor, n_donor):
    """Donor catchments and direct totals for one filled forest, both modes."""
    nodes = [e.tree_.node_count for e in model.estimators_]
    lf, lr = model.apply(X_fill), model.apply(X_rec)
    out = {}
    for mode in ("tree", "pooled"):
        c_rows = fill_catchments(lf, mass, lr, scale_rec, nodes, mode)
        est = recipient_estimates(lf, mass, y_fill, lr, nodes, mode)
        out[mode] = (np.bincount(rows_to_donor, weights=c_rows, minlength=n_donor),
                     float((est * scale_rec).sum()))
    return out


def analyse_split(X_pos, y_pos, w_pos, X_rec, w_rec, *, seed, gamma, eps_pos):
    Xp = X_pos.to_numpy(dtype=np.float64)
    Xr = X_rec.to_numpy(dtype=np.float64)
    n = len(y_pos)
    p_rec, mplus_rec, _ = structural(X_rec)
    _, mplus, sigma = structural(X_pos)
    mstar = mplus * np.exp(-gamma * sigma)
    scale_rec = w_rec * p_rec
    truth = float((scale_rec * mplus_rec).sum())

    catch, direct = {}, {}
    rng = np.random.default_rng([seed, 7])
    picked = qd.weighted_bootstrap(n, w_pos, rng)
    model = forest(Xp[picked], y_pos[picked], seed)
    counts = qd.bootstrap_counts(model, n)
    for mode, (c, tot) in fill(model, Xp[picked], counts, y_pos[picked], Xr, scale_rec,
                               picked, n).items():
        catch[f"full/{mode}"], direct[f"full/{mode}"] = c, tot

    for key in ("ahalf/tree", "ahalf/pooled", "honest/tree", "honest/pooled"):
        catch[key], direct[key] = np.zeros(n), 0.0
    half = np.random.default_rng([seed, 31]).random(n) < 0.5
    for k, grow in enumerate((half, ~half)):
        idx_g, idx_f = np.flatnonzero(grow), np.flatnonzero(~grow)
        pick_g = idx_g[qd.weighted_bootstrap(len(idx_g), w_pos[idx_g],
                                             np.random.default_rng([seed, 7, k]))]
        model = forest(Xp[pick_g], y_pos[pick_g], seed)
        cnt_g = qd.bootstrap_counts(model, len(pick_g))
        adaptive = fill(model, Xp[pick_g], cnt_g, y_pos[pick_g], Xr, scale_rec, pick_g, n)
        mass_f = np.broadcast_to(w_pos[idx_f] / w_pos[idx_f].mean(),
                                 (model.n_estimators, len(idx_f)))
        honest = fill(model, Xp[idx_f], mass_f, y_pos[idx_f], Xr, scale_rec, idx_f, n)
        for mode in ("tree", "pooled"):
            catch[f"ahalf/{mode}"] += 0.5 * adaptive[mode][0]
            direct[f"ahalf/{mode}"] += 0.5 * adaptive[mode][1]
            catch[f"honest/{mode}"] += 0.5 * honest[mode][0]
            direct[f"honest/{mode}"] += 0.5 * honest[mode][1]

    q90 = np.quantile(eps_pos, 0.9)
    q99 = np.quantile(eps_pos, 0.99)
    out = {"truth": truth, "n_positive_donors": n, "totals": {}, "split": {},
           "catchment_profile": {}, "catchment_identity_max_rel_err": 0.0,
           "catchment_sum_over_scale": {}}
    for key, c in catch.items():
        total = float((c * y_pos).sum())
        out["catchment_identity_max_rel_err"] = max(
            out["catchment_identity_max_rel_err"], abs(total / direct[key] - 1))
        out["catchment_sum_over_scale"][key] = float(c.sum() / scale_rec.sum())
        out["totals"][key] = total
        out["split"][key] = {
            "selection": float((c * (y_pos - mstar)).sum()),
            "informative": float((c * (mstar - mplus)).sum()),
            "smoothing": float((c * mplus).sum() - truth),
        }
        # Every positive donor that carries catchment, by its own noise draw.
        cbar = c.mean()
        out["catchment_profile"][key] = {
            "top10_noise": float(c[eps_pos > q90].mean() / cbar),
            "top1_noise": float(c[eps_pos > q99].mean() / cbar),
            "bottom50_noise": float(c[eps_pos < np.median(eps_pos)].mean() / cbar),
        }
    return out


def run(design, reps, out_path, smoke=False):
    if smoke:
        synth.N_POP, synth.N_SAMPLE = 150_000, 6_000
    gamma = synth.GAMMA[design]
    results = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["rep"] for r in results}
    for rep in range(reps):
        if rep in done:
            continue
        t0 = time.time()
        # Same stream as synth.run_rep and forest_bias.run_synth.
        rng = np.random.default_rng([20261107, rep, 0 if design == "x_only" else 1])
        X, y, true_mean, eps = synth.population(rng)
        p_all, mplus_all, _ = structural(X)
        assert np.allclose(p_all * mplus_all, true_mean, rtol=1e-12, atol=0), "formula drift"
        idx, w = synth.sample(rng, X, eps, gamma)
        Xs, ys, es = X.iloc[idx].reset_index(drop=True), y[idx], eps[idx]
        rec = rng.random(len(idx)) < 0.2
        donor_pos = (~rec) & (ys > 0)
        out = analyse_split(
            Xs.loc[donor_pos].reset_index(drop=True), ys[donor_pos], w[donor_pos],
            Xs.loc[rec].reset_index(drop=True), w[rec], seed=1000 + rep, gamma=gamma,
            eps_pos=es[donor_pos])
        out["truth_overall"] = float((true_mean[idx][rec] * w[rec]).sum())
        out["rep"], out["design"] = rep, design
        results.append(out)
        out_path.write_text(json.dumps(results, indent=1))
        print(f"rep {rep} ({design}) done in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", choices=["x_only", "informative"], default="x_only")
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run(args.design, args.reps, args.out, smoke=args.smoke)


if __name__ == "__main__":
    main()
