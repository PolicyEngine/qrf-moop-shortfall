"""Summarize the decomposition results into tables and Shapley shares.

Reads results/cps.json and results/synth_*.json and prints markdown.
Every percentage is 100 * (imputed total / truth - 1).
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MECHS = ["type7_grid", "type1_grid", "type7_exact", "type1_exact", "type7_knots"]
MECH_LABEL = {
    "type7_grid": "linear, grid (microimpute)",
    "type1_grid": "inverse ECDF, grid",
    "type7_exact": "linear, exact u",
    "type1_exact": "inverse ECDF, exact u",
    "type7_knots": "linear, 203 knots (Microcosm)",
}
FORESTS = {"none": "unweighted", "split": "split weights", "boot": "weighted bootstrap"}
LEAVES = {"leaf1": "1 sample/leaf", "leafall": "all samples"}
FACTORS = ["G", "I", "L", "W"]
FACTOR_LABEL = {
    "G": "grid (round(100u)+1)/102 -> exact u",
    "I": "linear (type 7) -> inverse ECDF",
    "L": "max_samples_leaf 1 -> None",
    "W": "unweighted -> weighted bootstrap",
}


def cell_value(totals: dict, fixed: set[str]) -> float:
    mech = ("type1" if "I" in fixed else "type7") + ("_exact" if "G" in fixed else "_grid")
    leaf = "leafall" if "L" in fixed else "leaf1"
    forest = "boot" if "W" in fixed else "none"
    return totals[f"{forest}/{leaf}"][mech]


def shapley(value) -> dict[str, float]:
    n = len(FACTORS)
    out = {}
    for f in FACTORS:
        others = [g for g in FACTORS if g != f]
        phi = 0.0
        for k in range(n):
            for subset in itertools.combinations(others, k):
                s = set(subset)
                weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                phi += weight * (value(s | {f}) - value(s))
        out[f] = phi
    return out


def pct(total: float, truth: float) -> float:
    return 100.0 * (total / truth - 1.0)


def summarize_split(results: list[dict], truth_key) -> dict:
    """Per-result percentages for every cell, plus Shapley shares."""
    rows = []
    for r in results:
        truth = truth_key(r)
        t = r["totals"]
        row = {"truth": truth}
        for cell, value in t.items():
            if isinstance(value, dict):
                for m, v in value.items():
                    row[f"{cell}/{m}"] = pct(v, truth)
            else:
                row[cell] = pct(value, truth)
        row["microimpute_realized"] = pct(r["microimpute_realized"]["total"], truth)
        for label, mc in r.get("microcosm", {}).items():
            row[f"microcosm_{label}/expected"] = pct(mc["expected_total"], truth)
            row[f"microcosm_{label}/pool_mean"] = pct(mc["expected_total_pool_mean"], truth)
            row[f"microcosm_{label}/realized"] = pct(mc["realized"]["total"], truth)
        phi = shapley(lambda s: pct(cell_value(t, s), truth))
        for f, v in phi.items():
            row[f"shapley/{f}"] = v
        rows.append(row)
    return rows


def combine_folds(results: list[dict]) -> dict:
    """Sum totals over CPS folds into one pseudo-result (truth summed too)."""
    out = {"totals": {}, "microimpute_realized": {"total": 0.0}, "microcosm": {}}
    truth = 0.0
    for r in results:
        truth += r["truth"]["reported_total"]
        out["microimpute_realized"]["total"] += r["microimpute_realized"]["total"]
        for cell, value in r["totals"].items():
            if isinstance(value, dict):
                agg = out["totals"].setdefault(cell, {m: 0.0 for m in value})
                for m, v in value.items():
                    agg[m] += v
            else:
                out["totals"][cell] = out["totals"].get(cell, 0.0) + value
        for label, mc in r.get("microcosm", {}).items():
            agg = out["microcosm"].setdefault(
                label, {"expected_total": 0.0, "expected_total_pool_mean": 0.0,
                        "realized": {"total": 0.0}})
            agg["expected_total"] += mc["expected_total"]
            agg["expected_total_pool_mean"] += mc["expected_total_pool_mean"]
            agg["realized"]["total"] += mc["realized"]["total"]
    out["truth"] = {"reported_total": truth}
    return out


def mean_se(rows: list[dict], key: str) -> tuple[float, float]:
    vals = np.array([r[key] for r in rows if key in r])
    if len(vals) == 1:
        return float(vals[0]), float("nan")
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals)))


def fmt(m: float, se: float) -> str:
    if math.isnan(se):
        return f"{m:+.2f}"
    return f"{m:+.2f} ({se:.2f})"


def cell_table(rows: list[dict]) -> str:
    lines = ["| Forest | Leaf pool | " + " | ".join(MECH_LABEL[m] for m in MECHS) + " |",
             "|---|---|" + "---|" * len(MECHS)]
    for forest, flabel in FORESTS.items():
        for leaf, llabel in LEAVES.items():
            cells = [fmt(*mean_se(rows, f"{forest}/{leaf}/{m}")) for m in MECHS]
            lines.append(f"| {flabel} | {llabel} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def shapley_table(rows: list[dict]) -> str:
    base = mean_se(rows, "none/leaf1/type7_grid")
    full = mean_se(rows, "boot/leafall/type1_exact")
    lines = ["| Step | Shapley share (points) |", "|---|---|",
             f"| Baseline: microimpute (linear, grid, 1 sample/leaf, unweighted) | {fmt(*base)} |"]
    for f in FACTORS:
        lines.append(f"| {FACTOR_LABEL[f]} | {fmt(*mean_se(rows, f'shapley/{f}'))} |")
    lines.append(f"| All four fixed (inverse ECDF, exact u, all samples, weighted bootstrap) | {fmt(*full)} |")
    return "\n".join(lines)


def extra_table(rows: list[dict]) -> str:
    keys = [
        ("none/meinshausen", "unweighted forest, tree-averaged leaf means"),
        ("split/meinshausen", "split-weighted forest, tree-averaged leaf means"),
        ("ecdf/type1_exact", "PR 219 weighted ECDF (split weights + survey-weighted leaves)"),
        ("boot/meinshausen", "weighted-bootstrap forest, tree-averaged leaf means"),
        ("honest/meinshausen", "honest forest, unweighted"),
        ("honest_weighted/meinshausen", "honest forest, survey-weighted leaves"),
        ("microimpute_realized", "microimpute QRF, realized draws"),
        ("microcosm_weighted/expected", "Microcosm RegimeGatedQRF weighted, exact"),
        ("microcosm_weighted/realized", "Microcosm weighted, realized draws"),
        ("microcosm_weighted/pool_mean", "Microcosm weighted, amount = pool mean"),
        ("microcosm_unweighted/expected", "Microcosm unweighted, exact"),
        ("microcosm_unweighted/realized", "Microcosm unweighted, realized draws"),
        ("microcosm_unweighted/pool_mean", "Microcosm unweighted, amount = pool mean"),
    ]
    lines = ["| Quantity | % vs truth |", "|---|---|"]
    for key, label in keys:
        if any(key in r for r in rows):
            lines.append(f"| {label} | {fmt(*mean_se(rows, key))} |")
    return "\n".join(lines)


DIST_CELLS = [
    ("none/leaf1/type7_grid", "microimpute (linear, grid, 1/leaf, unweighted)"),
    ("none/leaf1/type7_exact", "exact u"),
    ("none/leaf1/type1_exact", "exact u, inverse ECDF"),
    ("none/leafall/type1_exact", "exact u, inverse ECDF, all samples"),
    ("boot/leafall/type1_exact", "all four fixed"),
    ("boot/leafall/type7_knots", "Microcosm-style draw (knots, all samples, boot)"),
]


def dist_table(results: list[dict], truth_stats_key) -> str:
    """Realized-draw statistics per cell, % vs truth, mean over splits."""
    stats = ["total", "median", "p90", "p95", "p99"]
    lines = ["| Cell | " + " | ".join(stats) + " |", "|---|" + "---|" * len(stats)]

    def row(label, getter):
        vals = []
        for s in stats:
            v = [100 * (getter(r)[s] / truth_stats_key(r)[s] - 1) for r in results]
            vals.append(f"{np.mean(v):+.1f}")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    if "published_realized" in results[0]:
        row("published draws (cache)", lambda r: r["published_realized"])
    row("microimpute draws (rerun)", lambda r: r["microimpute_realized"])
    for key, label in DIST_CELLS:
        forest, leaf, mech = key.split("/")
        row(label, lambda r, f=forest, l=leaf, m=mech: r["realized"][f"{f}/{l}"][m])
    for label in ("weighted", "unweighted"):
        if "microcosm" in results[0]:
            row(f"Microcosm RegimeGatedQRF {label}",
                lambda r, lab=label: r["microcosm"][lab]["realized"])
    return "\n".join(lines)


def validation(results: list[dict]) -> str:
    """Realized totals against exact expectations, in Monte Carlo SEs."""
    worst = []
    for r in results:
        for cell, mechs in r["realized"].items():
            for m, st in mechs.items():
                exp = r["totals"][cell][m]
                se = st["total_sd"] / np.sqrt(st["n_draws"])
                worst.append(((st["total"] - exp) / se, f"{cell}/{m}"))
        for label, mc in r.get("microcosm", {}).items():
            se = mc["realized"]["total_sd"] / np.sqrt(mc["realized"]["n_draws"])
            worst.append(((mc["realized"]["total"] - mc["expected_total"]) / se,
                          f"microcosm_{label}"))
    z = np.array([w[0] for w in worst])
    top = max(worst, key=lambda w: abs(w[0]))
    return (f"{len(z)} realized-vs-exact comparisons: mean z {z.mean():+.2f}, "
            f"sd {z.std():.2f}, max |z| {abs(top[0]):.2f} ({top[1]})")


def main() -> None:
    res = ROOT / "results"
    cps_path = res / "cps.json"
    if cps_path.exists():
        cps = json.loads(cps_path.read_text())
        print(f"## CPS ASEC 2026, folds {[r['fold'] for r in cps]}\n")
        combined = combine_folds(cps)
        rows = summarize_split([combined], lambda r: r["truth"]["reported_total"])
        print("### All folds pooled (truth = reported total)\n")
        print(cell_table(rows), "\n")
        print(shapley_table(rows), "\n")
        print(extra_table(rows), "\n")
        per_fold = summarize_split(cps, lambda r: r["truth"]["reported_total"])
        print("### Per-fold mean (SE across folds)\n")
        print(shapley_table(per_fold), "\n")
        print("### Realized draws, % vs reported (mean over folds)\n")
        print(dist_table(cps, lambda r: r["truth"]["reported_stats"]), "\n")
        print(validation(cps), "\n")
        for r in cps:
            print(f"fold {r['fold']}: replay diff {r['replay_max_abs_diff']:.2e}, "
                  f"published first-draw diff {r.get('published_first_draw_max_abs_diff')}")
    for design in ("informative", "x_only"):
        path = res / f"synth_{design}.json"
        if not path.exists():
            continue
        syn = json.loads(path.read_text())
        rows = summarize_split(syn, lambda r: r["truth"]["conditional_mean_total"])
        print(f"\n## Synthetic, {design} design, {len(syn)} replications "
              "(truth = recipients' weighted conditional means; mean (SE))\n")
        print(cell_table(rows), "\n")
        print(shapley_table(rows), "\n")
        print(extra_table(rows), "\n")
        print("Realized draws, % vs the recipients' own values (mean over reps)\n")
        print(dist_table(syn, lambda r: r["truth"]["realized_stats"]), "\n")
        print(validation(syn), "\n")


if __name__ == "__main__":
    sys.exit(main())
