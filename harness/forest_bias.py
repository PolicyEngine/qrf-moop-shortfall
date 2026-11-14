"""Is the forest's own conditional mean biased, and if so, why?

Every forest estimate here is a weighted sum of donor values: recipient i
gets sum_j a_ij y_j. Summing over recipients with their survey weights w_i
gives each donor a catchment c_j = sum_i w_i a_ij, so the aggregate is
sum_j c_j y_j exactly. On the synthetic process, where each donor's
conditional mean m_j is known, the aggregate error splits exactly into

    sum_j c_j (y_j - m*_j)    selection: zero in expectation unless the
                              catchments depend on the donors' own noise
  + sum_j c_j (m*_j - m_j)    informative sampling (informative design only)
  + sum_j c_j m_j - truth     smoothing: local averaging of m over neighbours

where m*_j = E[y | x_j, sampled] = m_j exp(-gamma sigma_j), since Poisson
inclusion proportional to exp(-gamma eps) tilts eps to N(-gamma, 1).

Estimators, all unweighted (the mechanism test needs no survey weights):

- full/tree: microimpute's own forest (refit with the experiment's seed),
  per-tree normalized leaf means (Meinshausen), the expected mean of the
  max_samples_leaf=1 pool. Must equal cells.py's ``none/meinshausen``.
- full/pooled: the same forest's in-bag samples pooled across trees by
  bootstrap count, quantile-forest's default with max_samples_leaf=None.
  Must equal cells.py's ``none/leafall/type1_exact``.
- ahalf/*: forests grown on one random half, filled with that half's own
  in-bag samples (adaptive), averaged over the two halves.
- honest/*: the same half-forests filled with the other half's donors,
  which the splits never saw. ``honest/tree`` must equal cells.py's
  ``honest/meinshausen``.

ahalf and honest share every partition, so ahalf - honest removes tree
size from the comparison; full - ahalf is the effect of growing on twice
the data. ahalf - honest remains a total contrast: on the synthetic
process it carries the two fills' different smoothing as well as
selection, and in tree mode the honest fill skips a recipient's tree
whose leaf holds no filling donor.

Usage:
  python forest_bias.py --cps --folds 0 1 2 3 4 --out results/forest_bias_cps.json
  python forest_bias.py --design x_only --reps 4 --out results/forest_bias_x_only.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "microimpute"))

import qrfdecomp as qd  # noqa: E402

N_JOBS = 4


def fill_catchments(leaves_fill, mass, leaves_rec, w_rec, node_counts, mode):
    """Catchments and recipient estimates' weights for one filled forest.

    ``leaves_fill`` (n_fill, T) and ``mass`` (T, n_fill): where each filling
    donor lands and with what mass per tree (bootstrap counts for in-bag
    fills, ones for honest fills). ``mode`` "tree" normalizes within each
    tree's leaf and averages the trees that hold any filling mass for the
    recipient; "pooled" pools every tree's leaf mass and normalizes once.
    Either mode raises if a recipient's leaves hold no filling mass in any
    tree. Returns c, one catchment per filling donor.
    """
    n_fill, n_trees = leaves_fill.shape
    M = [np.bincount(leaves_fill[:, t], weights=mass[t], minlength=node_counts[t])
         for t in range(n_trees)]
    held = np.stack([M[t][leaves_rec[:, t]] for t in range(n_trees)], axis=1)
    if mode not in ("tree", "pooled"):
        raise ValueError(mode)
    if not (held > 0).any(axis=1).all():
        raise RuntimeError("a recipient found no filling donor in any tree")
    if mode == "tree":
        scale = w_rec / (held > 0).sum(axis=1)
    else:
        scale = w_rec / held.sum(axis=1)
    c = np.zeros(n_fill)
    for t in range(n_trees):
        R = np.bincount(leaves_rec[:, t], weights=scale, minlength=node_counts[t])
        lf = leaves_fill[:, t]
        if mode == "tree":
            ok = M[t][lf] > 0
            c[ok] += mass[t][ok] * R[lf][ok] / M[t][lf][ok]
        else:
            c += mass[t] * R[lf]
    return c


def recipient_estimates(leaves_fill, mass, y_fill, leaves_rec, node_counts, mode):
    """Direct per-recipient estimates, to check the catchment identity.

    Raises, as ``fill_catchments`` does, if a recipient's leaves hold no
    filling mass in any tree.
    """
    n_trees = leaves_fill.shape[1]
    num = np.zeros(len(leaves_rec))
    den = np.zeros(len(leaves_rec))
    for t in range(n_trees):
        s0 = np.bincount(leaves_fill[:, t], weights=mass[t], minlength=node_counts[t])
        s1 = np.bincount(leaves_fill[:, t], weights=mass[t] * y_fill,
                         minlength=node_counts[t])
        lr = leaves_rec[:, t]
        if mode == "tree":
            ok = s0[lr] > 0
            num[ok] += s1[lr][ok] / s0[lr][ok]
            den[ok] += 1
        else:
            num += s1[lr]
            den += s0[lr]
    if (den == 0).any():
        raise RuntimeError("a recipient found no filling donor in any tree")
    return num / den


def analyse_split(X_donor, y_donor, X_rec, w_rec, *, seed, m_donor=None,
                  mstar_donor=None, truth_m=None, noise_donor=None):
    """Totals (and, with m, the exact error split) for every estimator."""
    from microimpute.models.qrf import QRF

    y_donor = np.asarray(y_donor, dtype=np.float64)
    n = len(y_donor)
    predictors = list(X_donor.columns)
    train = X_donor.copy()
    train["moop"] = y_donor
    fitted = QRF(seed=seed, log_level="ERROR").fit(
        train, predictors=predictors, imputed_variables=["moop"],
        not_numeric_categorical=predictors, n_jobs=N_JOBS)
    inner = fitted.models["moop"]
    base = inner.qrf
    Xd = inner._align_features(X_donor)
    Xr = inner._align_features(X_rec)

    # Catchments: one array over all donors per estimator.
    catch: dict[str, np.ndarray] = {}
    totals_direct: dict[str, float] = {}

    counts = qd.bootstrap_counts(base, n)
    lf = base.apply(Xd)
    lr = base.apply(Xr)
    nodes = [e.tree_.node_count for e in base.estimators_]
    for mode in ("tree", "pooled"):
        catch[f"full/{mode}"] = fill_catchments(lf, counts, lr, w_rec, nodes, mode)
        est = recipient_estimates(lf, counts, y_donor, lr, nodes, mode)
        totals_direct[f"full/{mode}"] = float((est * w_rec).sum())

    # Half forests exactly as cells.honest_mean grows them.
    params = dict(n_estimators=base.n_estimators, random_state=base.random_state,
                  n_jobs=N_JOBS, max_samples_leaf=1)
    rng = np.random.default_rng([seed, 31])
    half = rng.random(n) < 0.5
    for key in ("ahalf/tree", "ahalf/pooled", "honest/tree", "honest/pooled"):
        catch[key] = np.zeros(n)
        totals_direct[key] = 0.0
    for grow in (half, ~half):
        fill = ~grow
        idx_g, idx_f = np.flatnonzero(grow), np.flatnonzero(fill)
        model = RandomForestQuantileRegressor(**params)
        model.fit(Xd.iloc[idx_g], y_donor[idx_g])
        nodes_h = [e.tree_.node_count for e in model.estimators_]
        lr_h = model.apply(Xr)
        lg = model.apply(Xd.iloc[idx_g])
        lfi = model.apply(Xd.iloc[idx_f])
        cnt_g = qd.bootstrap_counts(model, len(idx_g))
        ones_f = np.ones((model.n_estimators, len(idx_f)))
        for mode in ("tree", "pooled"):
            c = fill_catchments(lg, cnt_g, lr_h, w_rec, nodes_h, mode)
            catch[f"ahalf/{mode}"][idx_g] += 0.5 * c
            est = recipient_estimates(lg, cnt_g, y_donor[idx_g], lr_h, nodes_h, mode)
            totals_direct[f"ahalf/{mode}"] += 0.5 * float((est * w_rec).sum())
            c = fill_catchments(lfi, ones_f, lr_h, w_rec, nodes_h, mode)
            catch[f"honest/{mode}"][idx_f] += 0.5 * c
            est = recipient_estimates(lfi, ones_f, y_donor[idx_f], lr_h, nodes_h, mode)
            totals_direct[f"honest/{mode}"] += 0.5 * float((est * w_rec).sum())

    out = {"totals": {}, "catchment_identity_max_rel_err": 0.0}
    for key, c in catch.items():
        total = float((c * y_donor).sum())
        rel = abs(total / totals_direct[key] - 1)
        out["catchment_identity_max_rel_err"] = max(out["catchment_identity_max_rel_err"], rel)
        out["totals"][key] = total
        # Every estimator's weights sum to one per recipient.
        out.setdefault("catchment_sum_over_wrec", {})[key] = float(c.sum() / w_rec.sum())

    if m_donor is not None:
        out["split"] = {}
        out["catchment_profile"] = {}
        positive = y_donor > 0
        q = np.quantile(noise_donor[positive], [0.9, 0.99])
        for key, c in catch.items():
            out["split"][key] = {
                "selection": float((c * (y_donor - mstar_donor)).sum()),
                "informative": float((c * (mstar_donor - m_donor)).sum()),
                "smoothing": float((c * m_donor).sum() - truth_m),
            }
            cbar = c[positive].mean()
            out["catchment_profile"][key] = {
                "top10_noise": float(c[positive & (noise_donor > q[0])].mean() / cbar),
                "top1_noise": float(c[positive & (noise_donor > q[1])].mean() / cbar),
                "bottom50_noise": float(
                    c[positive & (noise_donor < np.median(noise_donor[positive]))].mean()
                    / cbar),
            }
    return out


def run_cps(folds, out_path):
    from experiments.moop_spm.cps_asec import build_units, design_matrix, load_persons
    from experiments.moop_spm.run import assign_folds

    from cps import DATA_DIR, SEED

    persons = load_persons(DATA_DIR)
    units = build_units(persons)
    X = design_matrix(units)
    fold_of = assign_folds(units["household"], 5, SEED)
    moop = units["moop"].to_numpy(dtype=float)
    weight = units["unit_weight"].to_numpy(dtype=float)
    results = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["fold"] for r in results}
    for fold in folds:
        if fold in done:
            continue
        t0 = time.time()
        donor, rec = fold_of != fold, fold_of == fold
        out = analyse_split(X.loc[donor], moop[donor], X.loc[rec], weight[rec],
                            seed=SEED + fold)
        out["truth"] = float((moop[rec] * weight[rec]).sum())
        out["fold"] = fold
        results.append(out)
        out_path.write_text(json.dumps(results, indent=1))
        print(f"fold {fold} done in {time.time() - t0:.0f}s", flush=True)


def run_synth(design, reps, out_path, smoke=False):
    import synth

    if smoke:
        synth.N_POP, synth.N_SAMPLE = 150_000, 6_000
    results = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["rep"] for r in results}
    gamma = synth.GAMMA[design]
    for rep in range(reps):
        if rep in done:
            continue
        t0 = time.time()
        # Same stream as synth.run_rep, so the samples match the main run.
        rng = np.random.default_rng([20261107, rep, 0 if design == "x_only" else 1])
        X, y, true_mean, eps = synth.population(rng)
        idx, w = synth.sample(rng, X, eps, gamma)
        Xs, ys, ms, es = X.iloc[idx].reset_index(drop=True), y[idx], true_mean[idx], eps[idx]
        rec = rng.random(len(idx)) < 0.2
        sigma = (0.95 + 0.3 * Xs["age"] + 0.1 * Xs["disab"]).to_numpy()
        mstar = ms * np.exp(-gamma * sigma)
        truth = float((ms[rec] * w[rec]).sum())
        out = analyse_split(
            Xs.loc[~rec].reset_index(drop=True), ys[~rec],
            Xs.loc[rec].reset_index(drop=True), w[rec], seed=1000 + rep,
            m_donor=ms[~rec], mstar_donor=mstar[~rec], truth_m=truth,
            noise_donor=es[~rec])
        out["truth"] = truth
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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    global N_JOBS
    N_JOBS = args.jobs
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.cps:
        run_cps(args.folds, args.out)
    else:
        run_synth(args.design, args.reps, args.out, smoke=args.smoke)


if __name__ == "__main__":
    main()
