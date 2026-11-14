"""Invariants of the catchment decomposition and Microcosm's row recovery.

``forest_bias.fill_catchments`` turns a filled forest into one catchment per
filling donor, and every split in the write-up rests on two identities:

1. sum_j c_j = sum_i scale_i (each recipient's weights sum to one), and
2. sum_j c_j y_j = sum_i scale_i est_i, where est_i is the recipient's
   estimate computed directly (``recipient_estimates``),

for per-tree ("tree") and count-pooled ("pooled") reads, for in-bag fills
(bootstrap counts) and honest fills (arbitrary positive masses). The
identities cover the recipients that some tree's leaf gives a filling
donor; for any other recipient both functions raise, in both modes.
``microcosm_actual.py`` also recovers a fitted forest's training rows from
``forest_.y_train`` and ``sorter_``; the recovered rows must reproduce the
training targets in order, and the count-pooled estimate from them must
equal the forest's own ``predict(quantiles="mean")``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from quantile_forest import RandomForestQuantileRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import qrfdecomp as qd  # noqa: E402
from forest_bias import fill_catchments, recipient_estimates  # noqa: E402


def small_problem(seed, n_fill, n_rec, max_samples_leaf):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_fill, 3))
    y = np.exp(7 + 0.5 * x[:, 0] + rng.normal(size=n_fill))
    y[rng.random(n_fill) < 0.1] = 0.0
    model = RandomForestQuantileRegressor(n_estimators=15, max_samples_leaf=max_samples_leaf,
                                          random_state=seed, n_jobs=1).fit(x, y)
    x_rec = rng.normal(size=(n_rec, 3))
    scale = np.exp(rng.normal(size=n_rec))
    return rng, model, x, y, x_rec, scale


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), n_fill=st.integers(30, 150), n_rec=st.integers(5, 60),
       honest=st.booleans())
def test_catchment_identities(seed, n_fill, n_rec, honest):
    rng, model, x, y, x_rec, scale = small_problem(seed, n_fill, n_rec, None)
    nodes = [e.tree_.node_count for e in model.estimators_]
    if honest:
        # Honest fill: fresh donors with arbitrary positive masses.
        x_fill = rng.normal(size=(n_fill, 3))
        y_fill = np.exp(7 + 0.5 * x_fill[:, 0] + rng.normal(size=n_fill))
        mass = np.broadcast_to(np.exp(rng.normal(size=n_fill)), (model.n_estimators, n_fill))
    else:
        x_fill, y_fill = x, y
        mass = qd.bootstrap_counts(model, n_fill).astype(float)
    lf, lr = model.apply(x_fill), model.apply(x_rec)
    held = np.stack([np.bincount(lf[:, t], weights=mass[t], minlength=nodes[t])[lr[:, t]]
                     for t in range(model.n_estimators)], axis=1)
    ok = (held > 0).any(axis=1)  # support is the same for both modes
    for mode in ("tree", "pooled"):
        if not ok.all():
            with pytest.raises(RuntimeError):
                fill_catchments(lf, mass, lr, scale, nodes, mode)
            with pytest.raises(RuntimeError):
                recipient_estimates(lf, mass, y_fill, lr, nodes, mode)
        if not ok.any():
            continue
        c = fill_catchments(lf, mass, lr[ok], scale[ok], nodes, mode)
        est = recipient_estimates(lf, mass, y_fill, lr[ok], nodes, mode)
        assert np.isfinite(est).all()
        assert np.isclose(c.sum(), scale[ok].sum(), rtol=1e-12)
        assert np.isclose((c * y_fill).sum(), (est * scale[ok]).sum(), rtol=1e-12)
        assert (c >= 0).all()


@pytest.mark.parametrize("mode", ["tree", "pooled"])
def test_recipient_without_filling_donor_raises(mode):
    """A recipient whose leaves hold no filling mass in any tree has no estimate."""
    _, model, x, y, x_rec, scale = small_problem(5, 120, 20, None)
    nodes = [e.tree_.node_count for e in model.estimators_]
    lf, lr = model.apply(x), model.apply(x_rec)
    mass = qd.bootstrap_counts(model, len(y)).astype(float)
    # Strip the filling mass from every leaf that recipient 0 lands in.
    for t in range(model.n_estimators):
        mass[t, lf[:, t] == lr[0, t]] = 0.0
    with pytest.raises(RuntimeError):
        fill_catchments(lf, mass, lr, scale, nodes, mode)
    with pytest.raises(RuntimeError):
        recipient_estimates(lf, mass, y, lr, nodes, mode)
    # The other recipients keep finite estimates where they have support.
    held = np.stack([np.bincount(lf[:, t], weights=mass[t], minlength=nodes[t])[lr[:, t]]
                     for t in range(model.n_estimators)], axis=1)
    ok = (held > 0).any(axis=1)
    assert not ok[0] and ok[1:].any()
    assert np.isfinite(recipient_estimates(lf, mass, y, lr[ok], nodes, mode)).all()


def test_pooled_estimate_matches_predict_mean_for_in_bag_fill():
    _, model, x, y, x_rec, _ = small_problem(3, 400, 80, None)
    nodes = [e.tree_.node_count for e in model.estimators_]
    counts = qd.bootstrap_counts(model, len(y))
    est = recipient_estimates(model.apply(x), counts, y, model.apply(x_rec), nodes, "pooled")
    mean = np.asarray(model.predict(x_rec, quantiles="mean"))
    np.testing.assert_allclose(est, mean, rtol=1e-12)


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(0, 10_000), n=st.integers(40, 300))
def test_training_rows_recover_from_sorted_targets(seed, n):
    """microcosm_actual.py's recovery of a weighted-bootstrap forest's rows."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = np.exp(7 + 0.5 * x[:, 0] + rng.normal(size=n))  # continuous: values are unique
    w = np.exp(rng.normal(size=n))
    picked = qd.weighted_bootstrap(n, w, rng)
    model = RandomForestQuantileRegressor(n_estimators=10, max_samples_leaf=None,
                                          random_state=seed, n_jobs=1).fit(x[picked], y[picked])
    y_sorted = np.asarray(model.forest_.y_train)[0]
    sorter = np.asarray(model.sorter_)[:, 0]
    y_fit = np.empty_like(y_sorted)
    y_fit[sorter] = y_sorted
    np.testing.assert_array_equal(y_fit, y[picked])
    order = np.argsort(y, kind="stable")
    rows = order[np.searchsorted(y[order], y_fit)]
    np.testing.assert_array_equal(y[rows], y_fit)
    x_rec = rng.normal(size=(30, 3))
    nodes = [e.tree_.node_count for e in model.estimators_]
    counts = qd.bootstrap_counts(model, len(y_fit))
    est = recipient_estimates(model.apply(x[rows]), counts, y_fit, model.apply(x_rec), nodes,
                              "pooled")
    np.testing.assert_allclose(est, np.asarray(model.predict(x_rec, quantiles="mean")),
                               rtol=1e-12)
