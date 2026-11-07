"""Cache the CPS design matrix, MOOP, weights and folds for other processes.

The PR 219 runner imports microimpute from a worktree at PR 219's head, so
it cannot import the experiment's loaders from ``main`` in the same
process. This writes exactly what ``cps.py`` builds, from ``main``'s
``experiments/moop_spm`` code, to ``results/cps_inputs.npz``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "microimpute"))

from cps import DATA_DIR, SEED  # noqa: E402
from experiments.moop_spm.cps_asec import (  # noqa: E402
    build_units,
    check_spm_identity,
    design_matrix,
    load_persons,
)
from experiments.moop_spm.run import assign_folds  # noqa: E402


def main() -> None:
    persons = load_persons(DATA_DIR)
    identity = check_spm_identity(persons)
    if any(value != 0 for value in identity.values()):
        raise ValueError(f"SPM identities fail: {identity}")
    units = build_units(persons)
    X = design_matrix(units)
    folds = assign_folds(units["household"], 5, SEED)
    out = HERE.parent / "results" / "cps_inputs.npz"
    np.savez_compressed(
        out,
        X=X.to_numpy(dtype=np.float64),
        columns=np.array(list(X.columns)),
        moop=units["moop"].to_numpy(dtype=float),
        weight=units["unit_weight"].to_numpy(dtype=float),
        folds=np.asarray(folds),
    )
    print(f"wrote {out}: {X.shape}, dtypes {sorted(set(map(str, X.dtypes)))}")


if __name__ == "__main__":
    main()
