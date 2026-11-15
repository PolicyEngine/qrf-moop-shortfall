"""How the harness picks and reads the experiment's cached draws.

``cps.cached_run`` names the run whose draws ``cps.py`` and
``pooled_dist.py`` compare with: the published run, or a fresh run whose
results.json sits beside the cache. Either must match the harness's seed
and folds and hold at least the requested draws. ``cps.cached_draws``
returns the same first draws to both callers and rejects a file whose
shape doesn't match its run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cps  # noqa: E402


@pytest.fixture
def cache(tmp_path, monkeypatch):
    folder = tmp_path / "moop-spm" / "cache"
    folder.mkdir(parents=True)
    monkeypatch.setattr(cps, "CACHE", folder)
    return folder


def fresh_run(cache, **config):
    base = {"tag": "fresh", "seed": cps.SEED, "folds": 5, "draws": 10}
    (cache.parent / "results.json").write_text(json.dumps({"config": {**base, **config}}))


def test_published_run_without_results_json(cache):
    run = cps.cached_run(10)
    assert run["tag"] == cps.PUBLISHED_RUN["tag"]
    assert run["source"].startswith("published")


def test_published_run_rejects_more_draws_than_it_holds(cache):
    with pytest.raises(ValueError):
        cps.cached_run(11)


def test_fresh_run_accepted_when_it_matches(cache):
    fresh_run(cache, draws=20)
    run = cps.cached_run(10)
    assert (run["tag"], run["draws"]) == ("fresh", 20)
    assert run["source"].startswith("fresh")


@pytest.mark.parametrize("config", [{"seed": 1}, {"folds": 4}, {"draws": 5}])
def test_fresh_run_rejected_when_it_differs(cache, config):
    fresh_run(cache, **config)
    with pytest.raises(ValueError):
        cps.cached_run(10)


def test_cached_draws_returns_the_requested_prefix(cache):
    fresh_run(cache, draws=20)
    run = cps.cached_run(10)
    draws = np.arange(20 * 7, dtype=float).reshape(20, 7)
    np.savez(cache / "fresh_fold2.npz", qrf=draws)
    np.testing.assert_array_equal(cps.cached_draws(run, 2, 7, 10), draws[:10])
    assert cps.cached_draws(run, 3, 7, 10) is None  # no file: skip the comparison


@pytest.mark.parametrize("shape", [(19, 7), (20, 6)])
def test_cached_draws_rejects_a_file_unlike_its_run(cache, shape):
    fresh_run(cache, draws=20)
    run = cps.cached_run(10)
    np.savez(cache / "fresh_fold0.npz", qrf=np.zeros(shape))
    with pytest.raises(ValueError):
        cps.cached_draws(run, 0, 7, 10)


def _fold(fold, run=None, cached=True):
    r = {"fold": fold}
    if cached:
        r["published_realized"] = {}
    if run is not None:
        r["experiment_run"] = run
    return r


@pytest.mark.parametrize("results", [
    [_fold(0), _fold(1, cached=False)],                    # a fold lacks the comparison
    [_fold(0, cached=False), _fold(1)],                    # ... even fold 0
    [_fold(0), _fold(1, run={**cps.PUBLISHED_RUN, "tag": "fresh",
                             "source": "fresh run (results.json beside the cache)"})],
])
def test_distribution_table_refuses_mixed_cache_comparisons(results):
    """analyze.dist_table labels the cache row by one run, so every fold must share it."""
    import analyze

    with pytest.raises(ValueError):
        analyze.dist_table(results, lambda r: {})


def test_published_run_shares_one_key_recorded_or_not():
    """Folds saved before the harness recorded the run used the published run."""
    import analyze

    fresh = {**cps.PUBLISHED_RUN, "tag": "fresh",
             "source": "fresh run (results.json beside the cache)"}
    assert analyze.cache_run_key(None) == analyze.cache_run_key(cps.PUBLISHED_RUN)
    assert analyze.cache_run_key(fresh) != analyze.cache_run_key(cps.PUBLISHED_RUN)
