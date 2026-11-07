"""Exact per-row draw expectations for quantile regression forests.

A QRF imputer draws a quantile per recipient row and reads the forest's
conditional quantile function there. The expected draw for a row therefore
depends on three things this module separates:

1. The forest's per-row conditional distribution: the training values in
   the row's leaves, pooled over trees. With ``max_samples_leaf=1``,
   quantile-forest keeps one random in-bag sample per leaf, so the pool
   holds one value per tree. With ``max_samples_leaf=None`` it keeps every
   in-bag sample, counting bootstrap duplicates.
2. The quantile function read from that pool. quantile-forest returns
   numpy's linear (type 7) quantile: rank ``(N - 1) q + 1`` in the pool
   expanded by counts. Its integral over ``q`` gives the smallest and
   largest pooled values half weight, so it falls short of the pool mean
   when the largest value sits far above the mean. The inverse ECDF (type 1)
   integrates to the pool mean exactly.
3. The mapping from a uniform draw ``u`` to the quantile queried:
   microimpute 3.2.0 rounds ``100 u`` to an index ``k`` and queries
   ``(k + 1) / 102`` (``microimpute/models/qrf.py`` lines 280-288);
   Microcosm's ``RegimeGatedQRF`` interpolates the forest's quantiles on a
   203-point grid at ``u`` itself.

Everything here is computed from the fitted trees, the bootstrap counts
and the training targets, without sampling, so differences between
mechanisms carry no Monte Carlo error. ``tests/test_qrfdecomp.py`` checks
each quantity against quantile-forest's own ``predict``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble._forest import (
    _generate_sample_indices,
    _get_n_samples_bootstrap,
)

# microimpute 3.2.0, microimpute/models/qrf.py lines 280-288: grid_size =
# max(count_samples, 101) = 101, quantile_grid = linspace(1/102, 101/102,
# 101), index = clip(rint(u * 100), 0, 100). Index 0 takes u < 0.005 and
# index 100 takes u >= 0.995, so the end points carry half the mass.
MI_GRID = np.linspace(1.0 / 102.0, 101.0 / 102.0, 101)
MI_PROB = np.full(101, 0.01)
MI_PROB[[0, -1]] = 0.005


def microcosm_grid() -> np.ndarray:
    """Microcosm's draw grid (``microcosm.fit.qrf._QUANTILE_GRID``)."""
    from microcosm.fit.qrf import _QUANTILE_GRID

    return np.asarray(_QUANTILE_GRID, dtype=np.float64)


def interp_weights(grid: np.ndarray) -> np.ndarray:
    """Weights ``w`` with ``E[interp(u; grid, v)] = w @ v`` for u ~ U(0, 1).

    Linear interpolation between knots, clamped to the end values outside
    the grid, as ``np.interp`` and Microcosm's ``_interp_rows`` do.
    """
    grid = np.asarray(grid, dtype=np.float64)
    weights = np.zeros(len(grid))
    weights[0] += grid[0]
    weights[-1] += 1.0 - grid[-1]
    gaps = np.diff(grid)
    weights[:-1] += gaps / 2.0
    weights[1:] += gaps / 2.0
    return weights


def bootstrap_counts(model: RandomForestQuantileRegressor, n_train: int) -> np.ndarray:
    """In-bag counts per tree, shape ``(n_trees, n_train)``.

    Replays the bootstrap that scikit-learn (< 1.9) and quantile-forest
    1.4.2 draw from each tree's ``random_state``.
    """
    if not model.bootstrap:
        return np.ones((len(model.estimators_), n_train), dtype=np.int64)
    n_boot = _get_n_samples_bootstrap(n_train, model.max_samples)
    counts = np.empty((len(model.estimators_), n_train), dtype=np.int64)
    for t, estimator in enumerate(model.estimators_):
        drawn = _generate_sample_indices(estimator.random_state, n_train, n_boot)
        counts[t] = np.bincount(drawn, minlength=n_train)
    return counts


def _ranges(starts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Concatenate ``arange(s, s + k)`` for each (s, k), vectorized."""
    total = int(sizes.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    offsets = np.repeat(np.cumsum(sizes) - sizes, sizes)
    return np.repeat(starts, sizes) + (np.arange(total) - offsets)


@dataclass
class RowPools:
    """Each recipient row's pooled (value, count) list, sorted by value.

    Rows are stored back to back: row ``r`` owns entries
    ``offset[r]:offset[r + 1]``. ``cum`` is the running count within the
    whole array, so expanded position ``p`` (1-based) of row ``r`` sits at
    entry ``searchsorted(cum, base[r] + p)``.
    """

    values: np.ndarray
    counts: np.ndarray
    offset: np.ndarray
    cum: np.ndarray
    base: np.ndarray
    n: np.ndarray  # pooled count per row

    @property
    def n_rows(self) -> int:
        return len(self.n)

    def mean(self) -> np.ndarray:
        sums = np.add.reduceat(self.values * self.counts, self.offset[:-1])
        return sums / self.n

    def min(self) -> np.ndarray:
        return self.values[self.offset[:-1]]

    def max(self) -> np.ndarray:
        return self.values[self.offset[1:] - 1]

    def _at_position(self, position: np.ndarray) -> np.ndarray:
        """Value at 1-based expanded position, one position per row."""
        entry = np.searchsorted(self.cum, self.base + position, side="left")
        return self.values[entry]

    def quantile(self, q: np.ndarray, method: str = "linear") -> np.ndarray:
        """One quantile per row (``q`` of shape ``(n_rows,)``).

        ``linear`` is numpy's type 7 on the count-expanded pool, which is
        what quantile-forest's weighted quantile returns; ``inverted_cdf``
        is type 1, the smallest value whose ECDF reaches ``q``.
        """
        q = np.asarray(q, dtype=np.float64)
        n = self.n.astype(np.float64)
        if method == "linear":
            h = (n - 1.0) * q + 1.0
            j = np.floor(h)
            frac = h - j
            j = np.clip(j, 1, n)
            j_next = np.minimum(j + 1, n)
            lower = self._at_position(j)
            upper = self._at_position(j_next)
            return lower + frac * (upper - lower)
        if method == "inverted_cdf":
            position = np.clip(np.ceil(n * q), 1, n)
            return self._at_position(position)
        raise ValueError(method)

    def quantiles(self, levels: np.ndarray, method: str = "linear") -> np.ndarray:
        """The same quantile levels for every row, shape ``(n_rows, n_levels)``."""
        out = np.empty((self.n_rows, len(levels)))
        for i, level in enumerate(levels):
            out[:, i] = self.quantile(np.full(self.n_rows, level), method)
        return out


def pools_from_leaf_values(leaf_values: np.ndarray) -> RowPools:
    """Pools for ``max_samples_leaf=1``: one stored value per tree."""
    m, t = leaf_values.shape
    values = np.sort(leaf_values, axis=1).ravel()
    counts = np.ones(m * t, dtype=np.int64)
    offset = np.arange(0, m * t + 1, t)
    cum = np.cumsum(counts)
    base = offset[:-1].astype(np.int64)
    return RowPools(values, counts, offset, cum, base, np.full(m, t))


def leaf_values_single(model: RandomForestQuantileRegressor, X_rec) -> np.ndarray:
    """The one stored value per tree for each row, as quantile-forest reads it.

    Mirrors ``BaseForestQuantileRegressor.predict``'s
    ``max_samples_leaf == 1`` path (quantile-forest 1.4.2).
    """
    if model.max_samples_leaf != 1:
        raise ValueError("leaf_values_single needs max_samples_leaf=1")
    leaves = model.apply(X_rec)
    stored = np.asarray(model.forest_.y_train_leaves)
    y_sorted = np.asarray(model.forest_.y_train).T
    out = np.empty((len(leaves), model.n_estimators))
    for t in range(model.n_estimators):
        index = stored[t, leaves[:, t], 0, 0]
        out[:, t] = y_sorted[index - 1, 0]
    return out


def pools_all_samples(
    model: RandomForestQuantileRegressor,
    X_train,
    y_train: np.ndarray,
    X_rec,
    counts: np.ndarray | None = None,
) -> RowPools:
    """Pools for ``max_samples_leaf=None``: every in-bag sample, with counts.

    The trees do not depend on ``max_samples_leaf`` (quantile-forest grows
    them through scikit-learn first and subsamples leaves afterwards), so a
    model fitted with ``max_samples_leaf=1`` gives the same pools as one
    fitted with ``None`` and the same ``random_state``.
    """
    y_train = np.asarray(y_train, dtype=np.float64)
    n_train = len(y_train)
    if counts is None:
        counts = bootstrap_counts(model, n_train)
    leaves_train = model.apply(X_train)
    leaves_rec = model.apply(X_rec)
    m = len(leaves_rec)
    rows_parts, idx_parts, cnt_parts = [], [], []
    for t in range(model.n_estimators):
        c = counts[t]
        inbag = np.flatnonzero(c > 0)
        leaf_of = leaves_train[inbag, t]
        order = np.argsort(leaf_of, kind="stable")
        inbag = inbag[order]
        leaf_sorted = leaf_of[order]
        n_nodes = model.estimators_[t].tree_.node_count
        start = np.searchsorted(leaf_sorted, np.arange(n_nodes), side="left")
        stop = np.searchsorted(leaf_sorted, np.arange(n_nodes), side="right")
        lr = leaves_rec[:, t]
        sizes = stop[lr] - start[lr]
        if (sizes == 0).any():
            raise RuntimeError("a recipient leaf holds no in-bag sample")
        pos = _ranges(start[lr], sizes)
        rows_parts.append(np.repeat(np.arange(m), sizes))
        idx_parts.append(inbag[pos])
        cnt_parts.append(c[inbag[pos]])
    rows = np.concatenate(rows_parts)
    idx = np.concatenate(idx_parts)
    cnt = np.concatenate(cnt_parts)
    # Merge repeats of a training row across trees, then sort each row's
    # entries by value.
    key = rows.astype(np.int64) * n_train + idx
    order = np.argsort(key, kind="stable")
    key = key[order]
    cnt = cnt[order]
    first = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    merged_cnt = np.add.reduceat(cnt, first)
    merged_key = key[first]
    merged_rows = merged_key // n_train
    merged_idx = merged_key % n_train
    merged_values = y_train[merged_idx]
    order = np.lexsort((merged_values, merged_rows))
    values = merged_values[order]
    counts_out = merged_cnt[order]
    rows_sorted = merged_rows[order]
    offset = np.searchsorted(rows_sorted, np.arange(m + 1), side="left")
    cum = np.cumsum(counts_out)
    base = np.r_[0, cum[offset[1:-1] - 1]] if m > 1 else np.array([0])
    n = np.add.reduceat(counts_out, offset[:-1])
    return RowPools(values, counts_out, offset, cum, base.astype(np.int64), n)


def meinshausen_mean(
    model: RandomForestQuantileRegressor,
    X_train,
    y_train: np.ndarray,
    X_rec,
    counts: np.ndarray | None = None,
    survey_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Tree-average of within-leaf (optionally survey-weighted) means.

    This is the mean of Meinshausen's (2006) conditional distribution and of
    microimpute PR 219's survey-weighted ECDF: each tree normalizes its
    in-bag donors' ``count * weight`` within the leaf, and trees count
    equally. It is also the expected mean of a ``max_samples_leaf=1``
    forest's 100 stored values when ``survey_weight`` is None.
    """
    y_train = np.asarray(y_train, dtype=np.float64)
    n_train = len(y_train)
    if counts is None:
        counts = bootstrap_counts(model, n_train)
    sw = np.ones(n_train) if survey_weight is None else np.asarray(survey_weight, float)
    leaves_train = model.apply(X_train)
    leaves_rec = model.apply(X_rec)
    total = np.zeros(len(leaves_rec))
    for t in range(model.n_estimators):
        mass = counts[t] * sw
        n_nodes = model.estimators_[t].tree_.node_count
        s0 = np.bincount(leaves_train[:, t], weights=mass, minlength=n_nodes)
        s1 = np.bincount(leaves_train[:, t], weights=mass * y_train, minlength=n_nodes)
        lr = leaves_rec[:, t]
        total += s1[lr] / s0[lr]
    return total / model.n_estimators


def type7_integral(pools: RowPools) -> np.ndarray:
    """``E[Q_7(u)]`` for u ~ U(0, 1), in closed form.

    For an expanded pool z_1 <= ... <= z_N the linear quantile function is
    piecewise linear through (k-1)/(N-1), so its integral is the trapezoid
    sum ``(sum z - (z_1 + z_N) / 2) / (N - 1)``.
    """
    n = pools.n.astype(np.float64)
    mean = pools.mean()
    out = (n * mean - (pools.min() + pools.max()) / 2.0) / (n - 1.0)
    return np.where(n > 1, out, mean)


@dataclass
class Expectations:
    """Expected draw per row under each draw mechanism, for one pool."""

    mean: np.ndarray  # inverse ECDF at exact u (type 1): the pool mean
    type7_exact: np.ndarray  # linear quantile at exact u (PR 219 unweighted path)
    type7_grid: np.ndarray  # linear quantile on microimpute's grid (microimpute)
    type1_grid: np.ndarray  # inverse ECDF on microimpute's grid
    type7_knots: np.ndarray  # linear quantile interpolated on Microcosm's grid

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "type1_exact": self.mean,
            "type7_exact": self.type7_exact,
            "type7_grid": self.type7_grid,
            "type1_grid": self.type1_grid,
            "type7_knots": self.type7_knots,
        }


def expectations(pools: RowPools, knots: np.ndarray | None = None) -> Expectations:
    """Every mechanism's expected draw per row, from one set of pools."""
    if knots is None:
        knots = microcosm_grid()
    q7_grid = pools.quantiles(MI_GRID, "linear")
    q1_grid = pools.quantiles(MI_GRID, "inverted_cdf")
    q7_knots = pools.quantiles(knots, "linear")
    return Expectations(
        mean=pools.mean(),
        type7_exact=type7_integral(pools),
        type7_grid=q7_grid @ MI_PROB,
        type1_grid=q1_grid @ MI_PROB,
        type7_knots=q7_knots @ interp_weights(knots),
    )


def draw(
    pools: RowPools,
    u: np.ndarray,
    mechanism: str,
    knots: np.ndarray | None = None,
) -> np.ndarray:
    """Realized draws for given uniforms under one mechanism."""
    if mechanism == "type1_exact":
        return pools.quantile(u, "inverted_cdf")
    if mechanism == "type7_exact":
        return pools.quantile(u, "linear")
    if mechanism in ("type7_grid", "type1_grid"):
        index = np.clip(np.rint(u * 100).astype(int), 0, 100)
        method = "linear" if mechanism == "type7_grid" else "inverted_cdf"
        return pools.quantile(MI_GRID[index], method)
    if mechanism == "type7_knots":
        if knots is None:
            knots = microcosm_grid()
        q = pools.quantiles(knots, "linear")
        rows = np.arange(pools.n_rows)
        upper = np.clip(np.searchsorted(knots, u, side="left"), 1, len(knots) - 1)
        lower = upper - 1
        span = knots[upper] - knots[lower]
        weight = np.clip(np.where(span > 0, (u - knots[lower]) / span, 0.0), 0, 1)
        return q[rows, lower] + weight * (q[rows, upper] - q[rows, lower])
    raise ValueError(mechanism)


def weighted_bootstrap(
    n: int, weights: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Row indices resampled n-of-n in proportion to weight.

    Microcosm's ``_weighted_bootstrap`` (microcosm-fit 0.1.0) does the same
    with ``rng.choice(n, size=n, replace=True, p=w / w.sum())``.
    """
    p = np.asarray(weights, dtype=np.float64)
    return rng.choice(n, size=n, replace=True, p=p / p.sum())


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Smallest value whose weighted cumulative share reaches ``q``.

    The experiment's ``evaluate.weighted_quantile``.
    """
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    position = np.searchsorted(cumulative, q * cumulative[-1], side="left")
    return float(values[order][min(position, len(values) - 1)])
