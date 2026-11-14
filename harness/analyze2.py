"""Second-pass tables: the shortfall's path split, named configurations,
zero shares, PR 219's own code and the forest-bias decomposition.

Every percentage is 100 * (total / truth - 1). "Pooled" sums totals over
CPS folds (every unit a recipient once, truth = the full reported total).
SEs are across replications (synthetic) or across folds (CPS). A CPS fold
SE, the SD of the five folds' errors over sqrt(5), describes how much the
folds differ, not the pooled estimate's sampling error: any two folds'
forests share three of their four training folds, and it ignores the CPS
survey design. Differences between methods on the same recipients vary
little across folds; a level against the reported total varies more.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(v, t):
    return 100.0 * (v / t - 1.0)


def mse(vals):
    vals = np.asarray(vals, float)
    if len(vals) < 2:
        return float(vals.mean()), float("nan")
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(len(vals)))


def fmt(m, se=float("nan"), d=2):
    return f"{m:+.{d}f}" if math.isnan(se) else f"{m:+.{d}f} ({se:.{d}f})"


def cps_truth(r):
    return r["truth"]["reported_total"]


def syn_truth(r):
    return r["truth"]["conditional_mean_total"]


def cell(r, forest, leaf, mech):
    return r["totals"][f"{forest}/{leaf}"][mech]


# --- path split of microimpute's own configuration -------------------------

def path_values(r, truth):
    """Forest mean and the 2-factor (grid, interpolation) Shapley split."""
    t = truth(r)
    v = {
        "base": pct(cell(r, "none", "leaf1", "type7_grid"), t),
        "exact_u": pct(cell(r, "none", "leaf1", "type7_exact"), t),
        "invecdf_grid": pct(cell(r, "none", "leaf1", "type1_grid"), t),
        "forest": pct(cell(r, "none", "leaf1", "type1_exact"), t),
        "meinshausen": pct(r["totals"]["none/meinshausen"], t),
    }
    g1 = v["exact_u"] - v["base"]            # grid removed first
    g2 = v["forest"] - v["invecdf_grid"]     # grid removed second
    i1 = v["invecdf_grid"] - v["base"]       # interpolation removed first
    i2 = v["forest"] - v["exact_u"]          # interpolation removed second
    v["grid"] = -(g1 + g2) / 2               # signed contribution to the gap
    v["interp"] = -(i1 + i2) / 2
    # Each step's direct effect with the other step in place or removed.
    v["grid_given_linear"] = -g1
    v["grid_given_invecdf"] = -g2
    v["interp_given_grid"] = -i1
    v["interp_given_exact_u"] = -i2
    # Leaf: count pooling vs per-tree weighting (pool means), and pool size.
    v["leaf_weighting"] = pct(cell(r, "none", "leafall", "type1_exact"), t) - v["forest"]
    v["leaf_total_at_base"] = pct(cell(r, "none", "leafall", "type7_grid"), t) - v["base"]
    v["leaf_poolsize_on_interp"] = (
        (pct(cell(r, "none", "leafall", "type7_exact"), t)
         - pct(cell(r, "none", "leafall", "type1_exact"), t))
        - (v["exact_u"] - v["forest"]))
    v["leaf_grid_interaction"] = (v["leaf_total_at_base"] - v["leaf_weighting"]
                                  - v["leaf_poolsize_on_interp"])
    # Weights at the baseline's own draw and at the forest mean.
    v["boot_at_base"] = pct(cell(r, "boot", "leaf1", "type7_grid"), t) - v["base"]
    v["split_at_base"] = pct(cell(r, "split", "leaf1", "type7_grid"), t) - v["base"]
    v["boot_forest"] = pct(r["totals"]["boot/meinshausen"], t) - v["meinshausen"]
    v["split_forest"] = pct(r["totals"]["split/meinshausen"], t) - v["meinshausen"]
    v["honest_minus_adaptive"] = pct(r["totals"]["honest/meinshausen"], t) - v["meinshausen"]
    return v


def path_table(results, truth, pooled=None):
    rows = [path_values(r, truth) for r in results]
    keys = [
        ("base", "microimpute as run (linear, grid, 1 sample/leaf, unweighted), expected"),
        ("forest", "the same forest's own mean (exact u, inverse ECDF)"),
        ("grid", "grid's contribution (2-factor Shapley with interpolation)"),
        ("interp", "linear interpolation's contribution"),
        ("grid_given_linear", "  grid, direct effect at the linear quantile"),
        ("grid_given_invecdf", "  grid, direct effect at the inverse ECDF"),
        ("interp_given_grid", "  linear quantile, direct effect on the grid"),
        ("interp_given_exact_u", "  linear quantile, direct effect at exact u"),
        ("leaf_total_at_base", "set max_samples_leaf=None, at microimpute's draw"),
        ("leaf_weighting", "  of which count pooling (pool means)"),
        ("leaf_poolsize_on_interp", "  of which a larger pool on interpolation (at exact u)"),
        ("leaf_grid_interaction", "  of which the grid's interaction (remainder)"),
        ("boot_at_base", "weighted bootstrap, at microimpute's draw"),
        ("split_at_base", "split-criterion weights, at microimpute's draw"),
        ("boot_forest", "weighted bootstrap, forest mean"),
        ("split_forest", "split-criterion weights, forest mean"),
        ("honest_minus_adaptive", "honest minus adaptive forest mean"),
    ]
    head = "| Quantity | per-split mean (SE) |" + (" pooled |" if pooled else "")
    lines = [head, "|---|---|" + ("---|" if pooled else "")]
    pv = path_values(pooled, truth) if pooled else None
    for k, label in keys:
        line = f"| {label} | {fmt(*mse([r[k] for r in rows]))} |"
        if pv:
            line += f" {fmt(pv[k])} |"
        lines.append(line)
    lines.append("")
    lines.append(t_interval_line([r["forest"] for r in rows], "forest's own mean"))
    lines.append(t_interval_line([r["base"] for r in rows], "microimpute as run, expected"))
    return "\n".join(lines)


def t_interval_line(vals, label):
    """Per-split values, their mean and a descriptive 95% t interval.

    The splits share training data (any two CPS folds' forests share three
    of their four training folds), and the interval ignores the CPS's
    survey design, so it describes the spread across splits rather than a
    design-based confidence interval.
    """
    from scipy import stats

    vals = np.asarray(vals, float)
    m, se = mse(vals)
    t = stats.t.ppf(0.975, len(vals) - 1)
    return (f"- {label}: per-split values {', '.join(f'{v:+.2f}' for v in vals)}; mean "
            f"{m:+.3f}, SE {se:.3f}, t({len(vals) - 1}) 95% interval {m - t * se:+.2f} to "
            f"{m + t * se:+.2f}")


def zero_table(results, truth_stats):
    cells = [("none/leaf1", "type7_grid"), ("none/leaf1", "type1_exact"),
             ("none/leafall", "type1_exact"), ("boot/leafall", "type7_knots")]
    lines = ["| Draws | share zero |", "|---|---|"]
    rep = [truth_stats(r)["share_zero"] for r in results]
    lines.append(f"| reported | {np.mean(rep):.4f} |")
    for c, m in cells:
        z = [r["realized"][c][m]["share_zero"] for r in results]
        lines.append(f"| {c} {m} | {np.mean(z):.4f} |")
    for lab in ("weighted", "unweighted"):
        if "microcosm" in results[0]:
            z = [r["microcosm"][lab]["realized"]["share_zero"] for r in results]
            p = [1 - r["microcosm"][lab]["expected_positive_share"] for r in results]
            lines.append(f"| Microcosm {lab} (gate's expected zero share {np.mean(p):.4f}) "
                         f"| {np.mean(z):.4f} |")
    return "\n".join(lines)


def named_table(results, truth, pr219=None):
    """Each configuration's expected and realized totals, % vs truth.

    Each cell gives the mean over splits (SE across splits); for CPS a
    second figure pools the folds, summing every total and the truth.
    """
    pooled = "fold" in results[0]
    head = "| Configuration | expected | realized | forest mean |"
    if pooled:
        head = ("| Configuration | expected | pooled | realized | pooled "
                "| forest mean | pooled |")
    lines = [head, "|---|" + "---|" * (6 if pooled else 3)]
    T = [truth(r) for r in results]

    def cols(totals, truths):
        if totals is None:
            return ["—", "—"] if pooled else ["—"]
        per = fmt(*mse([pct(v, t) for v, t in zip(totals, truths)]))
        if not pooled:
            return [per]
        return [per, fmt(pct(sum(totals), sum(truths)))]

    def add(label, exp, real, fm, truths):
        cells = cols(exp, truths) + cols(real, truths) + cols(fm, truths)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    add("microimpute main (as run)",
        [cell(r, "none", "leaf1", "type7_grid") for r in results],
        [r["microimpute_realized"]["total"] for r in results],
        [r["totals"]["none/meinshausen"] for r in results], T)
    if pr219:
        by = {r.get("fold", r.get("rep")): r for r in pr219}
        keyname = "fold" if "fold" in results[0] else "rep"
        common = [r for r in results if r[keyname] in by]
        tc = [truth(r) for r in common]
        for lab in ("unweighted", "weighted"):
            p = [by[r[keyname]][lab] for r in common]
            add(f"microimpute PR 219, {lab} ({len(common)} splits)",
                [x["expected"] for x in p], [x["realized"]["total"] for x in p],
                [x["forest_mean"] for x in p], tc)
    if "microcosm" in results[0]:
        for lab in ("weighted", "unweighted"):
            mc = [r["microcosm"][lab] for r in results]
            add(f"Microcosm RegimeGatedQRF, {lab}",
                [x["expected_total"] for x in mc], [x["realized"]["total"] for x in mc],
                None, T)
            add(f"  same gate and forest, amount read as the pool mean ({lab})",
                [x["expected_total_pool_mean"] for x in mc], None, None, T)
    return "\n".join(lines)


def pr219_details(pr219, truth_key):
    lines = []
    for r in pr219:
        k = r.get("fold", r.get("rep"))
        t = r["truth"][truth_key]
        u, w = r["unweighted"], r["weighted"]
        z_u = (u["realized"]["total"] - u["expected"]) / (
            u["realized"]["total_sd"] / math.sqrt(u["realized"]["n_draws"]))
        z_w = (w["realized"]["total"] - w["expected"]) / (
            w["realized"]["total_sd"] / math.sqrt(w["realized"]["n_draws"]))
        extra = ""
        if "grid_check" in u:
            extra = f", grid check rel diff {u['grid_check']['rel_diff_total']:+.2e}"
        lines.append(
            f"- split {k}: unweighted expected {pct(u['expected'], t):+.2f}, forest "
            f"{pct(u['forest_mean'], t):+.2f}, zero share "
            f"{u['realized']['share_zero']:.4f}, pool size median "
            f"{u['pool_size']['median']:.0f}, z {z_u:+.2f}; weighted expected "
            f"{pct(w['expected'], t):+.2f}, zero share {w['realized']['share_zero']:.4f}, "
            f"z {z_w:+.2f}{extra}")
    p = pr219[0]["unweighted"]["forest_params"]
    lines.append(f"- forest params: {p}")
    return "\n".join(lines)


# --- forest bias -------------------------------------------------------------

FB_KEYS = ["full/tree", "full/pooled", "ahalf/tree", "ahalf/pooled", "honest/tree",
           "honest/pooled"]
FB_LABEL = {
    "full/tree": "adaptive forest, per-tree weights (microimpute's leaf pool)",
    "full/pooled": "adaptive forest, count-pooled (max_samples_leaf=None)",
    "ahalf/tree": "half-data adaptive forests, per-tree",
    "ahalf/pooled": "half-data adaptive forests, count-pooled",
    "honest/tree": "same half-data trees, honest fill, per-tree",
    "honest/pooled": "same half-data trees, honest fill, count-pooled",
}


def fb_cps_table(fb, cps_cells=None):
    lines = ["| Estimator | % vs reported, per-fold mean (SE) | pooled |", "|---|---|---|"]
    T = sum(r["truth"] for r in fb)
    for k in FB_KEYS:
        per = [pct(r["totals"][k], r["truth"]) for r in fb]
        pooled = pct(sum(r["totals"][k] for r in fb), T)
        lines.append(f"| {FB_LABEL[k]} | {fmt(*mse(per))} | {fmt(pooled)} |")
    for a, b, lab in (("ahalf/tree", "honest/tree",
                       "adaptive-minus-honest total contrast (same trees), per-tree"),
                      ("ahalf/pooled", "honest/pooled",
                       "adaptive-minus-honest total contrast, count-pooled"),
                      ("full/tree", "ahalf/tree", "full minus half-data adaptive, per-tree")):
        per = [100 * (r["totals"][a] - r["totals"][b]) / r["truth"] for r in fb]
        pooled = 100 * sum(r["totals"][a] - r["totals"][b] for r in fb) / T
        lines.append(f"| {lab} | {fmt(*mse(per))} | {fmt(pooled)} |")
    err = max(r["catchment_identity_max_rel_err"] for r in fb)
    lines.append(f"\ncatchment identity max relative error {err:.1e}")
    if cps_cells:
        by = {r["fold"]: r for r in cps_cells}
        d1 = max(abs(r["totals"]["full/tree"] / by[r["fold"]]["totals"]["none/meinshausen"] - 1)
                 for r in fb if r["fold"] in by)
        d2 = max(abs(r["totals"]["full/pooled"] / by[r["fold"]]["totals"]["none/leafall"]["type1_exact"] - 1)
                 for r in fb if r["fold"] in by)
        d3 = max(abs(r["totals"]["honest/tree"] / by[r["fold"]]["totals"]["honest/meinshausen"] - 1)
                 for r in fb if r["fold"] in by)
        lines.append(f"consistency with cps.json (max rel diff): full/tree {d1:.1e}, "
                     f"full/pooled {d2:.1e}, honest/tree {d3:.1e}")
    return "\n".join(lines)


def fb_synth_table(fb):
    lines = ["| Estimator | total | selection | informative sampling | smoothing | "
             "catchment of top-10% noise donors |",
             "|---|---|---|---|---|---|"]
    for k in FB_KEYS:
        tot = [pct(r["totals"][k], r["truth"]) for r in fb]
        sel = [100 * r["split"][k]["selection"] / r["truth"] for r in fb]
        inf = [100 * r["split"][k]["informative"] / r["truth"] for r in fb]
        smo = [100 * r["split"][k]["smoothing"] / r["truth"] for r in fb]
        top = [r["catchment_profile"][k]["top10_noise"] for r in fb]
        lines.append(f"| {FB_LABEL[k]} | {fmt(*mse(tot))} | {fmt(*mse(sel))} | "
                     f"{fmt(*mse(inf))} | {fmt(*mse(smo))} | {np.mean(top):.2f}x |")
    return "\n".join(lines)


MB_LABEL = {
    "full/tree": "Microcosm-style amount forest, per-tree",
    "full/pooled": "Microcosm-style amount forest, count-pooled (Microcosm's reading)",
    "ahalf/tree": "half-data adaptive forests, per-tree",
    "ahalf/pooled": "half-data adaptive forests, count-pooled",
    "honest/tree": "same half-data trees, honest fill, per-tree",
    "honest/pooled": "same half-data trees, honest fill, count-pooled",
}


def mb_synth_table(mb):
    """microcosm_bias.py: positive amounts weighted by the true p(x)."""
    lines = ["| Estimator | total | selection | informative sampling | smoothing | "
             "catchment of top-10% noise donors | top 1% |",
             "|---|---|---|---|---|---|---|"]
    for k in FB_KEYS:
        tot = [pct(r["totals"][k], r["truth"]) for r in mb]
        sel = [100 * r["split"][k]["selection"] / r["truth"] for r in mb]
        inf = [100 * r["split"][k]["informative"] / r["truth"] for r in mb]
        smo = [100 * r["split"][k]["smoothing"] / r["truth"] for r in mb]
        top = [r["catchment_profile"][k]["top10_noise"] for r in mb]
        top1 = [r["catchment_profile"][k]["top1_noise"] for r in mb]
        lines.append(f"| {MB_LABEL[k]} | {fmt(*mse(tot))} | {fmt(*mse(sel))} | "
                     f"{fmt(*mse(inf))} | {fmt(*mse(smo))} | {np.mean(top):.2f}x | "
                     f"{np.mean(top1):.2f}x |")
    sel_diff = [100 * (r["split"]["ahalf/pooled"]["selection"]
                       - r["split"]["honest/pooled"]["selection"]) / r["truth"] for r in mb]
    lines.append("")
    lines.append(f"- adaptive minus honest selection, count-pooled: {fmt(*mse(sel_diff))}")
    lines.append(f"- catchment identity max rel err "
                 f"{max(r['catchment_identity_max_rel_err'] for r in mb):.1e}; truth equals "
                 f"the overall truth to "
                 f"{max(abs(r['truth'] / r['truth_overall'] - 1) for r in mb):.1e}")
    return "\n".join(lines)


def ma_synth_table(ma):
    """microcosm_actual.py: Microcosm's own amount forest, refit exactly."""
    t = [r["truth"] for r in ma]
    gate = [pct(r["total_gate_pool_mean"], r["truth"]) for r in ma]
    truep = [pct(r["total_truep_pool_mean"], r["truth"]) for r in ma]
    lines = ["| Microcosm's own forest (weighted), % vs truth | mean (SE) |", "|---|---|",
             f"| gate x pool mean (reproduces synth json) | {fmt(*mse(gate))} |",
             f"| true p x pool mean | {fmt(*mse(truep))} |"]
    for mode in ("pooled", "tree"):
        for part in ("selection", "informative", "smoothing"):
            vals = [100 * r["split"][mode][part] / r["truth"] for r in ma]
            lines.append(f"| {mode}: {part} | {fmt(*mse(vals))} |")
        top = [r["catchment_profile"][mode]["top10_noise"] for r in ma]
        top1 = [r["catchment_profile"][mode]["top1_noise"] for r in ma]
        lines.append(f"| {mode}: catchment of top-10% / top-1% noise donors | "
                     f"{np.mean(top):.2f}x / {np.mean(top1):.2f}x |")
    lines.append("")
    lines.append(
        "- checks: reproduces synth json max rel diff "
        f"{max(abs(r['reproduces_synth_json_rel_diff']) for r in ma):.1e}; pooled estimates vs "
        f"predict(mean) {max(r['pooled_estimate_vs_predict_mean_max_rel_diff'] for r in ma):.1e}; "
        "catchment identity "
        f"{max(max(r['catchment_identity_rel_err_pooled'], r['catchment_identity_rel_err_tree']) for r in ma):.1e}; "
        f"distinct donors in forest {np.mean([r['n_distinct_donors_in_forest'] / r['n_positive_donors'] for r in ma]):.3f} "
        f"of positive donors; gate share {np.mean([r['gate_positive_share'] for r in ma]):.4f} vs true "
        f"{np.mean([r['true_positive_share'] for r in ma]):.4f}")
    del t
    return "\n".join(lines)


def mc_cps_table(mc):
    """microcosm_cps.py: same-tree test of Microcosm-style forests on CPS."""
    T = sum(r["reported_total"] for r in mc)
    rows = [("Microcosm's own forest, gate x pool mean (reproduces cps.json)",
             lambda r: r["own_gate_pool_mean"])]
    for k, lab in (("ahalf/pooled", "half-data trees, adaptive fill, count-pooled"),
                   ("honest/pooled", "same trees, honest fill, count-pooled"),
                   ("ahalf/tree", "half-data trees, adaptive fill, per-tree"),
                   ("honest/tree", "same trees, honest fill, per-tree")):
        rows.append((lab, lambda r, k=k: r["totals"][k]))
    lines = ["| CPS, Microcosm-style amount forest x gate | % vs reported, per-fold mean (SE) | pooled |",
             "|---|---|---|"]
    for lab, get in rows:
        per = [pct(get(r), r["reported_total"]) for r in mc]
        lines.append(f"| {lab} | {fmt(*mse(per))} | {fmt(pct(sum(get(r) for r in mc), T))} |")
    for mode in ("pooled", "tree"):
        per = [100 * (r["totals"][f"ahalf/{mode}"] - r["totals"][f"honest/{mode}"])
               / r["reported_total"] for r in mc]
        pooled = 100 * sum(r["totals"][f"ahalf/{mode}"] - r["totals"][f"honest/{mode}"]
                           for r in mc) / T
        lines.append(f"| adaptive minus honest, {mode} | {fmt(*mse(per))} | {fmt(pooled)} |")
    lines.append("")
    lines.append(f"- reproduces cps.json max rel diff "
                 f"{max(abs(r['reproduces_cps_json_rel_diff']) for r in mc):.1e}; gate share "
                 f"{np.mean([r['gate_positive_share'] for r in mc]):.4f}")
    return "\n".join(lines)


def contrast_table(fb, mb):
    """Adaptive minus honest fill of the same half-data trees, by component.

    On the synthetic process each fill's error splits exactly, so the total
    contrast (the only one available on CPS) can be compared with its
    selection component; the two differ by the fills' smoothing (and
    sampling) contrasts.
    """
    lines = ["| Adaptive minus honest, points | total | selection | informative sampling "
             "| smoothing |", "|---|---|---|---|---|"]
    for label, res, mode in (("microimpute-style trees, per-tree", fb, "tree"),
                             ("microimpute-style trees, count-pooled", fb, "pooled"),
                             ("Microcosm-style trees, count-pooled", mb, "pooled"),
                             ("Microcosm-style trees, per-tree", mb, "tree")):
        if not res:
            continue
        a, h = f"ahalf/{mode}", f"honest/{mode}"
        tot = [100 * (r["totals"][a] - r["totals"][h]) / r["truth"] for r in res]
        parts = [[100 * (r["split"][a][k] - r["split"][h][k]) / r["truth"] for r in res]
                 for k in ("selection", "informative", "smoothing")]
        lines.append(f"| {label} | {fmt(*mse(tot))} | "
                     + " | ".join(fmt(*mse(x)) for x in parts) + " |")
    return "\n".join(lines)


def mk_table(mk, cps):
    """microcosm_knots.py: Microcosm's draw loss split on CPS."""
    T = sum(r["reported_total"] for r in mk)
    W = sum(r["weight_total"] for r in mk)
    lines = ["| CPS, Microcosm, gate x amount | per-fold mean (SE) | pooled |", "|---|---|---|"]
    for lab in ("weighted", "unweighted"):
        rows = [(f"{lab}: amount read as the pool mean", lambda r: r[lab]["total_pool_mean"]),
                (f"{lab}: exact type-7 expectation", lambda r: r[lab]["total_type7_exact"]),
                (f"{lab}: 203 knots (Microcosm's draw)", lambda r: r[lab]["total_knots"])]
        for name, get in rows:
            per = [pct(get(r), r["reported_total"]) for r in mk]
            lines.append(f"| {name} | {fmt(*mse(per))} | {fmt(pct(sum(get(r) for r in mk), T))} |")
        for name, a, b in ((f"{lab}: type-7 loss (exact type 7 minus pool mean)",
                            "total_type7_exact", "total_pool_mean"),
                           (f"{lab}: knots minus exact type 7", "total_knots", "total_type7_exact"),
                           (f"{lab}: combined (knots minus pool mean)", "total_knots",
                            "total_pool_mean")):
            per = [100 * (r[lab][a] - r[lab][b]) / r["reported_total"] for r in mk]
            pooled = 100 * sum(r[lab][a] - r[lab][b] for r in mk) / T
            lines.append(f"| {name} | {fmt(*mse(per), d=3)} | {fmt(pooled, d=3)} |")
        zero = sum((1 - r[lab]["expected_positive_share"]) * r["weight_total"] for r in mk) / W
        lines.append(f"| {lab}: gate's expected zero share, pooled | | {100 * zero:.2f}% |")
    checks = max(max(abs(r[lab]["reproduces_cps_json_knots_rel_diff"]),
                     abs(r[lab]["reproduces_cps_json_pool_mean_rel_diff"]))
                 for r in mk for lab in ("weighted", "unweighted"))
    mean_chk = max(r[lab]["pool_mean_vs_predict_max_rel_diff"] for r in mk
                   for lab in ("weighted", "unweighted"))
    knot_chk = max(r[lab]["knots_vs_predict_max_rel_diff"] for r in mk
                   for lab in ("weighted", "unweighted"))
    lines.append("")
    medians = {lab: ", ".join(f"{r[lab]['pool_size_median']:.0f}" for r in mk)
               for lab in ("weighted", "unweighted")}
    lines.append(f"- checks: reproduces cps.json totals {checks:.1e}; pool means vs "
                 f"predict(mean) {mean_chk:.1e}; integrated 203-knot expectation from the "
                 f"pools vs from predict(knots) {knot_chk:.1e}")
    lines.append(f"- each fold's median pool size: weighted {medians['weighted']}; "
                 f"unweighted {medians['unweighted']}")
    del cps
    return "\n".join(lines)


PD_LABEL = {
    "published": "microimpute as run (published draws)",
    "microimpute": "microimpute's predict, rerun",
    "type7_exact": "exact u, linear quantile (one sample per leaf)",
    "type1_exact": "exact u, inverse ECDF (one sample per leaf)",
    "leafall_type1_exact": "exact u, inverse ECDF, all samples count-pooled",
    "unweighted": "PR 219, unweighted",
    "weighted": "PR 219, weighted",
}


def pd_table(*runs):
    """pooled_dist.py: pooled draw distributions, % vs the pooled reported values."""
    stats = ["total", "median", "p75", "p90", "p95", "p99"]
    lines = ["| CPS, five folds pooled, % vs reported | " + " | ".join(stats)
             + " | share zero |", "|---|" + "---|" * (len(stats) + 1)]
    rep = next(r for r in runs if r)["reported"]
    lines.append("| reported | " + " | ".join("—" for _ in stats)
                 + f" | {100 * rep['share_zero']:.2f}% |")
    checks = []
    for run in runs:
        if not run:
            continue
        for name, st in run["cells"].items():
            vals = [f"{pct(st[s], rep[s]):+.1f}" for s in stats]
            lines.append(f"| {PD_LABEL.get(name, name)} | " + " | ".join(vals)
                         + f" | {100 * st['share_zero']:.2f}% |")
        checks.append(f"{run['code']}: {json.dumps(run['checks'])}")
    lines.append("")
    lines.extend(f"- checks, {c}" for c in checks)
    return "\n".join(lines)


def combine(results):
    """Pool CPS folds: sum every total and the truth."""
    from analyze import combine_folds

    return combine_folds(results)


def main() -> None:
    cps = load("cps.json")
    pr_cps = load("pr219_cps.json")
    fb_cps = load("forest_bias_cps.json")
    if cps:
        print(f"## CPS, {len(cps)} folds\n")
        pooled = combine(cps)
        # combine_folds keeps totals only; meinshausen keys are scalars there.
        print(path_table(cps, cps_truth, pooled), "\n")
        print(zero_table(cps, lambda r: r["truth"]["reported_stats"]), "\n")
        print(named_table(cps, cps_truth, pr_cps), "\n")
        if pr_cps:
            print(pr219_details(pr_cps, "reported_total"), "\n")
        if fb_cps:
            print(fb_cps_table(fb_cps, cps), "\n")
        mc = load("microcosm_cps.json")
        if mc:
            print(mc_cps_table(mc), "\n")
        mk = load("microcosm_knots_cps.json")
        if mk:
            print(mk_table(mk, cps), "\n")
        pdm, pdp = load("pooled_dist_main.json"), load("pooled_dist_pr219.json")
        if pdm or pdp:
            print(pd_table(pdm, pdp), "\n")
    for design in ("informative", "x_only"):
        syn = load(f"synth_{design}.json")
        pr = load(f"pr219_{design}.json")
        fb = load(f"forest_bias_{design}.json")
        if not (syn or fb):
            continue
        print(f"\n## Synthetic, {design}\n")
        if syn:
            print(f"{len(syn)} replications\n")
            print(path_table(syn, syn_truth), "\n")
            print(zero_table(syn, lambda r: r["truth"]["realized_stats"]), "\n")
            print(named_table(syn, syn_truth, pr), "\n")
        if pr:
            print(pr219_details(pr, "conditional_mean_total"), "\n")
        if fb:
            print(f"forest bias, {len(fb)} replications\n")
            print(fb_synth_table(fb), "\n")
        mb = load(f"microcosm_bias_{design}.json")
        if mb:
            print(f"Microcosm-style amount forest bias, {len(mb)} replications\n")
            print(mb_synth_table(mb), "\n")
        ma = load(f"microcosm_actual_{design}.json")
        if ma:
            print(f"Microcosm's own amount forest, {len(ma)} replications\n")
            print(ma_synth_table(ma), "\n")
        if fb or mb:
            print("Adaptive minus honest fills of the same trees, by component\n")
            print(contrast_table(fb, mb), "\n")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
