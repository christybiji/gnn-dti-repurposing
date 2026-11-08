"""
Standalone copy of the notebook's ranking metric (Section 10), extracted so
it can be unit tested independently of the full BindingDB pipeline.

Behavior is intentionally identical to the notebook's `concordance_index`.
"""
from __future__ import annotations

import numpy as np


def concordance_index(y, p, n_pairs: int = 2_000_000, seed: int = 0) -> float:
    """Probability that a randomly chosen comparable pair is ranked correctly.

    Estimated by sampling pairs rather than forming the full n^2 matrix,
    which becomes infeasible at screening scale (millions of predictions).
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    n = len(y)
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    comparable = y[i] > y[j]
    if not comparable.any():
        return float("nan")
    dp = p[i][comparable] - p[j][comparable]
    return float(((dp > 0).sum() + 0.5 * (dp == 0).sum()) / comparable.sum())
