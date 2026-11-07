"""Decompose the QRF MOOP shortfall on the 2026 CPS ASEC folds.

Loads the data, units, design matrix and folds exactly as
``experiments/moop_spm/run.py`` does (microimpute ``main``), then runs every
decomposition cell for each fold. The baseline cell is microimpute's QRF fit
with the experiment's arguments and seed (20261105 + fold); its first draw is
compared with the cached draws behind the published results.

Truth on CPS data is the fold's reported total: SPM-unit-weighted
``SPM_MEDXPNS``, as in the experiment.

Usage: python cps.py --folds 0 1 2 3 4 --out results/cps.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "microimpute"))

from cells import decompose, draw_stats  # noqa: E402
from experiments.moop_spm.cps_asec import (  # noqa: E402
    build_units,
    check_spm_identity,
    design_matrix,
    load_persons,
)
from experiments.moop_spm.run import assign_folds  # noqa: E402

# Census's public-use CPS ASEC 2026 files: the experiment's loader downloads
# them here and checks the zip's SHA-256. CACHE holds the experiment's own
# per-fold draws, which `python -m experiments.moop_spm.run --output
# ../data/moop-spm` (run in microimpute/) writes to data/moop-spm/cache;
# without them, main() skips the comparison with the published draws.
DATA_DIR = Path(os.environ.get("QRF_SHORTFALL_DATA", HERE.parent / "data" / "cps-asec-2026"))
CACHE = Path(os.environ.get("QRF_SHORTFALL_CACHE", HERE.parent / "data" / "moop-spm" / "cache"))
PUBLISHED_TAG = "a82da82512c894e6"  # results.json config.tag on main
SEED = 20261105


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    persons = load_persons(DATA_DIR)
    identity = check_spm_identity(persons)
    if any(value != 0 for value in identity.values()):
        raise ValueError(f"SPM identities fail: {identity}")
    units = build_units(persons)
    X = design_matrix(units)
    folds = assign_folds(units["household"], 5, SEED)
    moop = units["moop"].to_numpy(dtype=float)
    weight = units["unit_weight"].to_numpy(dtype=float)
    print(f"{len(units)} units, folds {np.bincount(folds).tolist()}, "
          f"reported total ${(moop * weight).sum() / 1e9:.1f}B", flush=True)

    results = json.loads(args.out.read_text()) if args.out.exists() else []
    done = {r["fold"] for r in results}
    for fold in args.folds:
        if fold in done:
            continue
        t0 = time.time()
        print(f"fold {fold}", flush=True)
        donor = folds != fold
        rec = folds == fold
        out = decompose(
            X.loc[donor],
            moop[donor],
            weight[donor],
            X.loc[rec],
            weight[rec],
            seed=SEED + fold,
            n_draws=args.draws,
        )
        first = out.pop("microimpute_draws_first")
        cached = CACHE / f"{PUBLISHED_TAG}_fold{fold}.npz"
        if cached.exists():
            published = np.load(cached, allow_pickle=False)["qrf"]
            out["published_first_draw_max_abs_diff"] = float(
                np.max(np.abs(published[0] - first))
            )
            out["published_realized"] = draw_stats(published, weight[rec])
        out["truth"] = {
            "reported_total": float((moop[rec] * weight[rec]).sum()),
            "reported_stats": draw_stats(moop[rec][None, :], weight[rec]),
        }
        out["fold"] = fold
        results.append(out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
        print(f"fold {fold} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
