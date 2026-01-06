from __future__ import annotations

import numpy as np


def build_var_design(data: np.ndarray, lags: int, include_intercept: bool = True):
    """Build response and regressor matrices for a VAR(p) with optional intercept."""
    if lags < 1:
        raise ValueError("lags must be >= 1")
    if data.ndim != 2:
        raise ValueError("data must be 2D (T x n)")

    n = data.shape[1]
    Y = data[lags:, :]
    T = Y.shape[0]
    if T <= 0:
        raise ValueError("Not enough observations for requested lags")

    k = n * lags + (1 if include_intercept else 0)
    X = np.zeros((T, k))

    for i in range(lags):
        start = lags - i - 1
        end = -i - 1
        X[:, i * n : (i + 1) * n] = data[start:end, :]

    if include_intercept:
        X[:, -1] = 1.0

    return Y, X
