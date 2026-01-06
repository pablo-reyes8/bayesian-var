from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import multigammaln

from .priors import minnesota_dummy_observations


def matrix_t_neg_logpdf(X, nu, M, Sigma, Omega):
    """Negative log-density of a matrix-variate Student's t distribution."""
    n, p = X.shape

    sign_Omega, logdet_Omega = np.linalg.slogdet(Omega)
    sign_Sigma, logdet_Sigma = np.linalg.slogdet(Sigma)
    if sign_Omega <= 0 or sign_Sigma <= 0:
        return 1e9

    gamma1 = multigammaln((nu + n + p - 1) / 2, p)
    gamma2 = multigammaln((nu + p - 1) / 2, p) + (n * p / 2) * np.log(np.pi)

    log_c0 = gamma1 - gamma2 - (n / 2) * logdet_Omega - (p / 2) * logdet_Sigma

    centered_X = X - M
    inv_Sigma = np.linalg.inv(Sigma)
    inv_Omega = np.linalg.inv(Omega)
    I_n = np.eye(n)

    inside_term = I_n + inv_Sigma @ centered_X @ inv_Omega @ centered_X.T
    sign_inside, logdet_inside = np.linalg.slogdet(inside_term)
    if sign_inside <= 0:
        return 1e9

    exponent = -(nu + n + p - 1) / 2
    density_log = log_c0 + exponent * logdet_inside

    if not np.isfinite(density_log):
        return 1e9
    return -density_log


def neg_log_marginal_likelihood(lambdas, Y, X, delta, su, s0, y_mean, lags):
    """Negative log marginal likelihood under Minnesota conjugate prior."""
    n = Y.shape[1]
    T = Y.shape[0]

    YP, XP = minnesota_dummy_observations(lambdas, delta, su, s0, y_mean, lags)

    B0 = np.linalg.solve(XP.T @ XP, XP.T @ YP)
    resid = YP - XP @ B0
    S0 = resid.T @ resid

    H0 = np.linalg.inv(XP.T @ XP)
    HH = np.eye(T) + X @ H0 @ X.T

    Z = Y.T
    M = (X @ B0).T
    v0 = n + 2
    v = v0 - n + 1
    return matrix_t_neg_logpdf(Z, v, M, S0, HH)


def optimize_minnesota_hyperparams(
    Y,
    X,
    delta,
    su,
    s0,
    y_mean,
    lags,
    bounds=None,
    n_starts=10,
    seed=None,
):
    """Optimize Minnesota prior hyperparameters via marginal likelihood."""
    if bounds is None:
        bounds = [(1e-3, 10.0)] * 5

    rng = np.random.default_rng(seed)
    best = None

    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])

    for _ in range(n_starts):
        initial = rng.uniform(lows, highs)
        result = minimize(
            neg_log_marginal_likelihood,
            initial,
            args=(Y, X, delta, su, s0, y_mean, lags),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if not result.success:
            continue
        if best is None or result.fun < best.fun:
            best = result

    if best is None:
        raise RuntimeError("Hyperparameter optimization failed for all starts")

    return best
