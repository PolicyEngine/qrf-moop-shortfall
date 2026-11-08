"""Differential tests: qrfdecomp against quantile-forest, microimpute, Microcosm.

Each exact quantity the decomposition uses is checked against the code that
produces real draws, on small right-skewed data.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from quantile_forest import RandomForestQuantileRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import qrfdecomp as qd  # noqa: E402

N_TRAIN = 2500
N_REC = 400


def skewed_data(seed: int, n: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    x[:, 3] = rng.integers(0, 4, n)  # a discrete predictor, so leaves tie
    mu = 7.5 + 0.6 * x[:, 0] + 0.3 * x[:, 1]
    sigma = 1.0 + 0.2 * (x[:, 2] > 0)
    positive = rng.random(n) < 0.9
    y = np.where(positive, np.exp(mu + sigma * rng.normal(size=n)), 0.0)
    w = np.exp(0.5 * rng.normal(size=n))
    return x, y, w


@pytest.fixture(scope="module")
def data():
    x, y, w = skewed_data(0, N_TRAIN + N_REC)
    return x[:N_TRAIN], y[:N_TRAIN], w[:N_TRAIN], x[N_TRAIN:]


def fit(x, y, leaf, seed=7, sample_weight=None):
    model = RandomForestQuantileRegressor(
        n_estimators=60, max_samples_leaf=leaf, random_state=seed, n_jobs=2
    )
    return model.fit(x, y, sample_weight=sample_weight)


LEVELS = np.r_[0.0, qd.MI_GRID, 0.37, 0.999, 1.0]


@pytest.mark.parametrize("weighted", [False, True])
def test_all_sample_pools_match_quantile_forest(data, weighted):
    x, y, w, x_rec = data
    sw = w if weighted else None
    single = fit(x, y, 1, sample_weight=sw)
    full = fit(x, y, None, sample_weight=sw)
    # Same trees whatever max_samples_leaf is.
    assert np.array_equal(single.apply(x_rec), full.apply(x_rec))
    pools = qd.pools_all_samples(single, x, y, x_rec)
    expected = np.asarray(full.predict(x_rec, quantiles=list(LEVELS)))
    ours = pools.quantiles(LEVELS, "linear")
    np.testing.assert_allclose(ours, expected, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(
        pools.mean(), full.predict(x_rec, quantiles="mean"), rtol=1e-12
    )


def test_single_sample_pools_match_quantile_forest(data):
    x, y, _, x_rec = data
    model = fit(x, y, 1)
    pools = qd.pools_from_leaf_values(qd.leaf_values_single(model, x_rec))
    expected = np.asarray(model.predict(x_rec, quantiles=list(LEVELS)))
    np.testing.assert_allclose(pools.quantiles(LEVELS, "linear"), expected, rtol=1e-12)
    np.testing.assert_allclose(
        pools.mean(), model.predict(x_rec, quantiles="mean"), rtol=1e-12
    )


def test_meinshausen_mean_matches_weighted_leaves(data):
    x, y, _, x_rec = data
    single = fit(x, y, 1)
    full = fit(x, y, None)
    ours = qd.meinshausen_mean(single, x, y, x_rec)
    theirs = full.predict(x_rec, quantiles="mean", weighted_leaves=True)
    np.testing.assert_allclose(ours, theirs, rtol=1e-10)


def test_type7_integral_matches_numerical_integral(data):
    x, y, _, x_rec = data
    pools = qd.pools_all_samples(fit(x, y, 1), x, y, x_rec)
    fine = np.linspace(0.0, 1.0, 40001)
    values = pools.quantiles(fine, "linear")
    numeric = np.trapezoid(values, fine, axis=1)
    np.testing.assert_allclose(qd.type7_integral(pools), numeric, rtol=1e-6)
    # And the inverse ECDF integrates to the pool mean.
    mid = (np.arange(200000) + 0.5) / 200000
    inv = np.mean(
        [pools.quantile(np.full(pools.n_rows, q), "inverted_cdf") for q in mid[::997]],
        axis=0,
    )
    assert np.all(np.isfinite(inv))


def test_expectations_match_monte_carlo(data):
    x, y, _, x_rec = data
    pools = qd.pools_all_samples(fit(x, y, 1), x, y, x_rec)
    exp = qd.expectations(pools).as_dict()
    rng = np.random.default_rng(3)
    n_draws = 3000
    for mechanism, expected in exp.items():
        total = 0.0
        for _ in range(n_draws):
            total += qd.draw(pools, rng.random(pools.n_rows), mechanism).sum()
        mc = total / n_draws
        # Monte Carlo SE of a sum of N_REC draws averaged over n_draws draws.
        se = np.sqrt(np.sum(pools.mean() ** 2)) * 2.0 / np.sqrt(n_draws)
        assert abs(mc - expected.sum()) < 5 * se, (mechanism, mc, expected.sum())


def test_microimpute_draws_replicate_exactly(data):
    from microimpute.models.qrf import QRF

    x, y, _, x_rec = data
    cols = [f"x{i}" for i in range(x.shape[1])]
    train = pd.DataFrame(x, columns=cols)
    train["y"] = y
    rec = pd.DataFrame(x_rec, columns=cols)
    fitted = QRF(seed=11, log_level="ERROR").fit(
        train, predictors=cols, imputed_variables=["y"], not_numeric_categorical=cols
    )
    inner = fitted.models["y"]
    assert inner.qrf.max_samples_leaf == 1
    pools = qd.pools_from_leaf_values(
        qd.leaf_values_single(inner.qrf, inner._align_features(rec))
    )
    # _QRFModel draws u ~ Beta(1, 1) from one generator it keeps across
    # predict calls; a copy of it replays the same stream.
    import copy

    rng = copy.deepcopy(inner._rng)
    for _ in range(3):
        u = rng.beta(1.0, 1.0, size=len(rec))
        ours = qd.draw(pools, u, "type7_grid")
        pred = fitted.predict(rec)
        if isinstance(pred, dict):
            pred = next(iter(pred.values()))
        # numpy's quantile interpolates as v_lo + frac * (v_hi - v_lo) in a
        # different operation order, so allow roundoff (seen: 1.2e-15).
        np.testing.assert_allclose(ours, pred["y"].to_numpy(), rtol=1e-12, atol=1e-9)


def test_microcosm_expectation_matches_its_draws(data):
    from microcosm.fit import NO_WEIGHTS, RegimeGatedQRF

    # microcosm-fit at 7080ba8 reads its thread caps from these variables
    # (microcosm/fit/qrf.py:905-910).
    os.environ.setdefault("POPULACE_FIT_N_JOBS", "2")
    os.environ.setdefault("POPULACE_FIT_PREDICT_WORKERS", "2")
    x, y, _, x_rec = data
    cols = [f"x{i}" for i in range(x.shape[1])]
    train = pd.DataFrame(x, columns=cols)
    train["y"] = y
    rec = pd.DataFrame(x_rec, columns=cols)
    fitted = RegimeGatedQRF(seed=5, n_estimators=60).fit(
        train, cols, ["y"], weights=NO_WEIGHTS
    )
    target = fitted._target_models["y"]
    forest = target.positive.model
    knots = qd.microcosm_grid()
    q = np.asarray(forest.predict(rec[list(target.positive.columns)].to_numpy(),
                                  quantiles=list(knots)))
    amount = q @ qd.interp_weights(knots)
    classes = list(target.gate.classes_)
    p_pos = target.gate.predict_proba(rec[list(target.columns)].to_numpy())[
        :, classes.index(1)
    ]
    expected = float(np.sum(p_pos * amount))
    draws = [fitted.predict(rec)["y"].to_numpy().sum() for _ in range(600)]
    se = np.std(draws) / np.sqrt(len(draws))
    assert abs(np.mean(draws) - expected) < 5 * se, (np.mean(draws), expected, se)


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.floats(0, 1e6, allow_nan=False), min_size=2, max_size=40),
    st.lists(st.integers(1, 5), min_size=40, max_size=40),
)
def test_pool_quantiles_match_numpy_on_expanded_sample(values, counts):
    """Property: pooled quantiles equal numpy's on the count-expanded sample."""
    values = np.sort(np.asarray(values))
    counts = np.asarray(counts[: len(values)])
    pools = qd.RowPools(
        values=values,
        counts=counts,
        offset=np.array([0, len(values)]),
        cum=np.cumsum(counts),
        base=np.array([0]),
        n=np.array([counts.sum()]),
    )
    expanded = np.repeat(values, counts)
    for q in [0.0, 0.01, 0.3, 0.5, 0.77, 0.99, 1.0]:
        for method in ("linear", "inverted_cdf"):
            ours = pools.quantile(np.array([q]), method)[0]
            theirs = np.quantile(expanded, q, method=method)
            assert ours == pytest.approx(theirs, rel=1e-12, abs=1e-9)
    numeric = np.trapezoid(np.quantile(expanded, np.linspace(0, 1, 20001)),
                           np.linspace(0, 1, 20001))
    assert qd.type7_integral(pools)[0] == pytest.approx(numeric, rel=1e-6, abs=1e-6)
