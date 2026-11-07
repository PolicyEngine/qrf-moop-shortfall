"""Two checks on the CPS decomposition, using microimpute main.

1. Leaf purity. Fully grown trees split until a node's targets are equal, so
   a leaf should hold one distinct MOOP value (a single donor, possibly
   repeated by the bootstrap, or several zero donors). If so, storing one
   sample per leaf loses nothing about the leaf's value, and switching to
   max_samples_leaf=None only changes how leaves are weighted: quantile-
   forest's default pools every in-bag sample by count, so a leaf holding
   many zero donors outweighs a leaf holding one positive donor.
2. Validation outliers. The three 10-draw realized totals furthest from
   their exact expectations sat 3.5, 4.2 and 5.8 Monte Carlo SEs below them
   (fold 0 boot/leaf1/type7_knots, fold 1 split/leaf1/type1_grid, fold 3
   boot/leafall/type7_grid). Rerun those cells with 400 draws to see
   whether the gaps persist.

Reads results/cps_inputs.npz and checks each refit against results/cps.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qrfdecomp as qd  # noqa: E402
from cps import SEED  # noqa: E402

N_JOBS = 4


def load():
    data = np.load(HERE.parent / "results" / "cps_inputs.npz", allow_pickle=False)
    X = pd.DataFrame(data["X"], columns=[str(c) for c in data["columns"]])
    return X, data["moop"], data["weight"], data["folds"]


def base_forest(X_donor, y_donor, seed):
    from microimpute.models.qrf import QRF

    predictors = list(X_donor.columns)
    train = X_donor.copy()
    train["moop"] = y_donor
    fitted = QRF(seed=seed, log_level="ERROR").fit(
        train, predictors=predictors, imputed_variables=["moop"],
        not_numeric_categorical=predictors, n_jobs=N_JOBS)
    inner = fitted.models["moop"]
    return inner.qrf, inner


def purity(model, Xd, y, Xr, w_rec):
    counts = qd.bootstrap_counts(model, len(y))
    lf, lr = model.apply(Xd), model.apply(Xr)
    impure_leaves = total_leaves = 0
    impure_mass = total_mass = 0.0
    zero_leaf_counts, pos_leaf_counts = [], []
    zero_tree = np.zeros(len(Xr))   # per-tree share: trees whose leaf is all zero
    zero_cnt = np.zeros(len(Xr))    # count-pooled zero mass
    all_cnt = np.zeros(len(Xr))
    for t, est in enumerate(model.estimators_):
        inbag = counts[t] > 0
        leaves = lf[inbag, t]
        ys = y[inbag]
        c = counts[t][inbag]
        frame = pd.DataFrame({"leaf": leaves, "y": ys, "c": c})
        g = frame.groupby("leaf")
        nuniq = g["y"].nunique()
        mass = g["c"].sum()
        ymax = g["y"].max()
        total_leaves += len(nuniq)
        impure_leaves += int((nuniq > 1).sum())
        total_mass += float(mass.sum())
        impure_mass += float(mass[nuniq > 1].sum())
        zero = (ymax == 0) & (nuniq == 1)
        zero_leaf_counts.append(mass[zero].to_numpy())
        pos_leaf_counts.append(mass[~zero].to_numpy())
        n_nodes = est.tree_.node_count
        is_zero = np.zeros(n_nodes, bool)
        is_zero[zero.index[zero.to_numpy()]] = True
        m = np.zeros(n_nodes)
        m[mass.index] = mass.to_numpy()
        zm = np.where(is_zero, m, 0.0)
        r = lr[:, t]
        zero_tree += is_zero[r]
        zero_cnt += zm[r]
        all_cnt += m[r]
    zc = np.concatenate(zero_leaf_counts)
    pc = np.concatenate(pos_leaf_counts)
    return {
        "share_leaves_impure": impure_leaves / total_leaves,
        "share_inbag_mass_in_impure_leaves": impure_mass / total_mass,
        "mean_count_zero_leaf": float(zc.mean()),
        "mean_count_positive_leaf": float(pc.mean()),
        "share_leaves_zero": float(len(zc) / (len(zc) + len(pc))),
        "recipient_zero_mass_per_tree": float(
            np.average(zero_tree / model.n_estimators, weights=w_rec)),
        "recipient_zero_mass_count_pooled": float(np.average(zero_cnt / all_cnt,
                                                             weights=w_rec)),
        "donor_zero_share_weighted": None,
    }


def rerun_cell(pools, w_rec, mech, n_draws, seed):
    knots = qd.microcosm_grid()
    exact = float((qd.expectations(pools, knots).as_dict()[mech] * w_rec).sum())
    rng = np.random.default_rng([seed, 4242])
    totals = np.array([(qd.draw(pools, rng.random(pools.n_rows), mech, knots) * w_rec).sum()
                       for _ in range(n_draws)])
    se = totals.std(ddof=1) / np.sqrt(n_draws)
    first10_sd = totals[:10].std(ddof=1)
    return {
        "exact": exact,
        "realized_mean": float(totals.mean()),
        "z": float((totals.mean() - exact) / se),
        "sd_all": float(totals.std(ddof=1)),
        "sd_first10": float(first10_sd),
        "skew": float(((totals - totals.mean()) ** 3).mean() / totals.std() ** 3),
        "n_draws": n_draws,
    }


def main() -> None:
    X, moop, weight, folds = load()
    cps = {r["fold"]: r for r in json.loads((HERE.parent / "results" / "cps.json").read_text())}
    out = {}
    for fold, forest_kind, leaf, mech in ((0, "boot", "leaf1", "type7_knots"),
                                          (1, "split", "leaf1", "type1_grid"),
                                          (3, "boot", "leafall", "type7_grid")):
        cell = f"{forest_kind}/{leaf}"
        seed = SEED + fold
        donor, rec = folds != fold, folds == fold
        Xd0, yd, wd = X.loc[donor].reset_index(drop=True), moop[donor], weight[donor]
        Xr0, wr = X.loc[rec].reset_index(drop=True), weight[rec]
        base, inner = base_forest(Xd0, yd, seed)
        Xd, Xr = inner._align_features(Xd0), inner._align_features(Xr0)
        res = {}
        if fold == 0:
            res["purity"] = purity(base, Xd, yd, Xr, wr)
            res["purity"]["donor_zero_share_weighted"] = float(np.average(yd == 0, weights=wd))
            res["purity"]["recipient_reported_zero_share"] = float(
                np.average(moop[rec] == 0, weights=wr))
            # The base forest must reproduce cps.json's meinshausen total.
            mein = qd.meinshausen_mean(base, Xd, yd, Xr)
            res["base_meinshausen_rel_diff_vs_cps_json"] = float(
                (mein * wr).sum() / cps[fold]["totals"]["none/meinshausen"] - 1)
        params = dict(n_estimators=base.n_estimators, random_state=base.random_state,
                      n_jobs=N_JOBS, max_samples_leaf=1)
        if forest_kind == "split":
            Xt, yt = Xd, np.asarray(yd, float)
            model = RandomForestQuantileRegressor(**params).fit(Xt, yt, sample_weight=wd)
        else:
            picked = qd.weighted_bootstrap(len(Xd), wd, np.random.default_rng([seed, 7]))
            Xt, yt = Xd.iloc[picked].reset_index(drop=True), np.asarray(yd, float)[picked]
            model = RandomForestQuantileRegressor(**params).fit(Xt, yt)
        if leaf == "leaf1":
            pools = qd.pools_from_leaf_values(qd.leaf_values_single(model, Xr))
        else:
            pools = qd.pools_all_samples(model, Xt, yt, Xr)
        rerun = rerun_cell(pools, wr, mech, 400, seed)
        rerun["exact_rel_diff_vs_cps_json"] = rerun["exact"] / cps[fold]["totals"][cell][mech] - 1
        rerun["cps_json_10draw_z"] = (
            (cps[fold]["realized"][cell][mech]["total"] - cps[fold]["totals"][cell][mech])
            / (cps[fold]["realized"][cell][mech]["total_sd"] / np.sqrt(10)))
        res[f"{cell}/{mech}"] = rerun
        out[f"fold{fold}"] = res
        print(json.dumps(res, indent=1), flush=True)
    (HERE.parent / "results" / "recheck.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
