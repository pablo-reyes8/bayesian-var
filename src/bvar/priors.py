from __future__ import annotations

import numpy as np


def ar1_prior_stats(data: np.ndarray, lags: int):
    """Estimate AR(1) coefficients and residual variances for each series."""
    if data.ndim != 2:
        raise ValueError("data must be 2D (T x n)")
    if lags < 1:
        raise ValueError("lags must be >= 1")

    n = data.shape[1]
    Y = data[lags:, :]
    T = Y.shape[0]

    delta = np.zeros(n)
    su = np.zeros(n)

    for i in range(n):
        y_lag = data[lags - 1 : -1, i].reshape(-1, 1)
        y_current = data[lags:, i].reshape(-1, 1)
        inv_term = np.linalg.inv(y_lag.T @ y_lag)
        delta[i] = (inv_term @ (y_lag.T @ y_current)).item()
        resid = y_current - delta[i] * y_lag
        su[i] = float((resid**2).sum() / T)

    s0 = np.diag(su)
    return delta, su, s0


def minnesota_dummy_observations(
    lambdas: np.ndarray,
    delta: np.ndarray,
    su: np.ndarray,
    s0: np.ndarray,
    y_mean: np.ndarray,
    lags: int,
):
    """Build Minnesota prior dummy observations for a VAR(p)."""
    if len(lambdas) != 5:
        raise ValueError("lambdas must have length 5")

    lambd0, lambd1, lambd3, lambd4, lambd5 = lambdas
    n = delta.shape[0]

    Jp = np.diag(np.arange(1, lags + 1)) ** (lambd1)
    YD1 = np.vstack(
        (np.diag((delta * np.sqrt(su)) / lambd0), np.zeros((n * (lags - 1), n)))
    )
    XD1 = np.hstack((np.kron(Jp, np.sqrt(s0) / lambd0), np.zeros((n * lags, 1))))

    Ycov = np.sqrt(s0)
    Xcov = np.zeros((n, n * lags + 1))
    XDc = np.hstack(
        (np.zeros((1, n * lags)), np.array([[1 / (lambd0 * lambd3)]]))
    )
    YDc = np.zeros((1, n))
    YD = np.vstack((YD1, YDc, Ycov))
    XD = np.vstack((XD1, XDc, Xcov))

    Ys = np.diag(y_mean / lambd4)
    Xs = np.hstack((np.kron(np.ones((1, lags)), Ys), np.zeros((n, 1))))

    Y0 = np.array(y_mean / lambd5).reshape(1, -1)
    X0 = np.hstack((np.kron(np.ones((1, lags)), Y0), np.array([[1 / lambd5]])))

    YP = np.vstack((YD, Ys, Y0))
    XP = np.vstack((XD, Xs, X0))

    return YP, XP
