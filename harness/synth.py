"""Synthetic right-skewed process with a known conditional mean.

A finite population carries MOOP-like spending: a logistic share of zeros
and lognormal amounts whose location and spread depend on the predictors.
A Poisson sample of about 50,000 units is drawn from it, with inclusion
probabilities that depend on the predictors and, in the informative
design, slightly on the amount's own noise; weights are inverse inclusion
probabilities. One fifth of the sample receives imputations from models
fit on the rest.

Truth is the weighted sum of the recipients' population conditional means,
``p_pos(x) * exp(mu(x) + sigma(x)^2 / 2)``, which the draws should match
in expectation.

Usage: python synth.py --reps 8 --design informative --out results/synth.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cells import decompose  # noqa: E402

N_POP = 1_500_000
N_SAMPLE = 50_000
G0 = np.array([0.0, -0.6, 0.3, -1.0, 0.5])  # participation shift by group
GM = np.array([0.0, -0.3, 0.2, -0.5, 0.3])  # amount shift by group
# Informative design: inclusion falls with the amount's noise. A tilt of
# gamma lowers the sample's conditional mean by about exp(-gamma * sigma),
# about 1.3% at gamma = 0.012.
GAMMA = {"x_only": 0.0, "informative": 0.012}


def population(rng: np.random.Generator):
    n = N_POP
    age = rng.uniform(0, 1, n)
    inc = rng.normal(0, 1, n)
    size = rng.integers(1, 7, n)
    group = rng.integers(0, 5, n)
    disab = (rng.random(n) < 0.12).astype(float)
    noise1 = rng.normal(0, 1, n)
    noise2 = rng.normal(0, 1, n)
    lin = 2.6 + 1.2 * age + 0.3 * inc + 0.25 * (size - 1) + G0[group] + 0.5 * disab
    p_pos = 1.0 / (1.0 + np.exp(-lin))
    mu = 7.5 + 0.9 * age + 0.35 * inc + 0.12 * (size - 1) + GM[group] + 0.4 * disab
    sigma = 0.95 + 0.3 * age + 0.1 * disab
    eps = rng.normal(0, 1, n)
    positive = rng.random(n) < p_pos
    y = np.where(positive, np.exp(mu + sigma * eps), 0.0)
    true_mean = p_pos * np.exp(mu + sigma**2 / 2.0)
    X = pd.DataFrame(
        {
            "age": age,
            "inc": inc,
            "size": size.astype(float),
            "group": group.astype(float),
            "disab": disab,
            "noise1": noise1,
            "noise2": noise2,
        }
    )
    return X, y, true_mean, eps


def sample(rng, X, eps, gamma):
    score = np.exp(-0.5 * X["inc"].to_numpy() + 0.3 * (X["group"].to_numpy() == 3)
                   - gamma * eps)
    pi = np.minimum(N_SAMPLE * score / score.sum(), 1.0)
    chosen = rng.random(len(pi)) < pi
    return np.flatnonzero(chosen), 1.0 / pi[chosen]


def run_rep(rep: int, design: str, n_draws: int, microcosm: bool) -> dict:
    rng = np.random.default_rng([20261107, rep, 0 if design == "x_only" else 1])
    X, y, true_mean, eps = population(rng)
    idx, w = sample(rng, X, eps, GAMMA[design])
    Xs = X.iloc[idx].reset_index(drop=True)
    ys = y[idx]
    ms = true_mean[idx]
    rec = rng.random(len(idx)) < 0.2
    out = decompose(
        Xs.loc[~rec].reset_index(drop=True),
        ys[~rec],
        w[~rec],
        Xs.loc[rec].reset_index(drop=True),
        w[rec],
        seed=1000 + rep,
        n_draws=n_draws,
        microcosm=microcosm,
    )
    out.pop("microimpute_draws_first", None)
    out["truth"] = {
        "conditional_mean_total": float((ms[rec] * w[rec]).sum()),
        "realized_total": float((ys[rec] * w[rec]).sum()),
        "population_total_scaled": float(y.sum() * w[rec].sum() / w.sum()),
    }
    # Truth for the distribution statistics: the recipients' own values.
    from cells import draw_stats

    out["truth"]["realized_stats"] = draw_stats(ys[rec][None, :], w[rec])
    out["rep"] = rep
    out["design"] = design
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--design", choices=list(GAMMA), default="informative")
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--no-microcosm", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    if args.out.exists():
        results = json.loads(args.out.read_text())
    done = {r["rep"] for r in results}
    for rep in range(args.start, args.start + args.reps):
        if rep in done:
            continue
        t0 = time.time()
        print(f"rep {rep} ({args.design})", flush=True)
        results.append(run_rep(rep, args.design, args.draws, not args.no_microcosm))
        args.out.write_text(json.dumps(results, indent=1))
        print(f"rep {rep} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
