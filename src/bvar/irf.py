from __future__ import annotations

import numpy as np


def cholesky_factors(omega_draws: np.ndarray) -> np.ndarray:
    """Compute Cholesky factors for each covariance draw."""
    return np.stack([np.linalg.cholesky(omega) for omega in omega_draws], axis=0)


def companion_matrices(beta_draws: np.ndarray) -> np.ndarray:
    """Build companion-form matrices from VAR coefficient draws."""
    if beta_draws.ndim != 3:
        raise ValueError("beta_draws must be 3D (draws x n x k)")

    draws, n, k_total = beta_draws.shape
    if (k_total - 1) % n == 0:
        coeffs = beta_draws[:, :, :-1]
    else:
        coeffs = beta_draws

    p = coeffs.shape[2] // n
    if p < 1:
        raise ValueError("Unable to infer lag order from coefficients")

    psi = np.empty((draws, n * p, n * p))
    I = np.eye(n * (p - 1))
    Z = np.zeros((n * (p - 1), n))
    bottom = np.hstack((I, Z))

    for i in range(draws):
        top = coeffs[i]
        psi[i] = np.vstack((top, bottom))

    return psi


def irf_mc(omega_draws: np.ndarray, beta_draws: np.ndarray, horizon: int) -> np.ndarray:
    """Compute Monte Carlo impulse responses for VAR draws."""
    if horizon < 0:
        raise ValueError("horizon must be >= 0")

    draws, n, _ = omega_draws.shape
    psi = companion_matrices(beta_draws)
    p = psi.shape[1] // n

    J = np.hstack((np.eye(n), np.zeros((n, n * (p - 1)))))
    B0 = cholesky_factors(omega_draws)

    irf = np.zeros((n, n * (horizon + 1), draws))

    for i in range(draws):
        irf[:, :n, i] = B0[i]
        A_power = psi[i]
        for h in range(1, horizon + 1):
            col = slice(h * n, (h + 1) * n)
            irf[:, col, i] = J @ A_power @ J.T @ B0[i]
            A_power = A_power @ psi[i]

    return irf


def irf_quantiles(irf: np.ndarray, horizon: int):
    """Compute quantiles and error bands for IRFs."""
    n, total_cols, draws = irf.shape
    if total_cols != n * (horizon + 1):
        raise ValueError("irf shape does not match horizon")

    q_90 = np.zeros((n, n, horizon))
    q_84 = np.zeros((n, n, horizon))
    q_50 = np.zeros((n, n, horizon))
    q_10 = np.zeros((n, n, horizon))
    q_16 = np.zeros((n, n, horizon))

    for i in range(n):
        for j in range(n):
            for h in range(horizon):
                col = j + h * n
                draw_vec = irf[i, col, :]
                q_90[i, j, h] = np.quantile(draw_vec, 0.95)
                q_84[i, j, h] = np.quantile(draw_vec, 0.84)
                q_50[i, j, h] = np.quantile(draw_vec, 0.50)
                q_10[i, j, h] = np.quantile(draw_vec, 0.05)
                q_16[i, j, h] = np.quantile(draw_vec, 0.16)

    bands = {
        "q_90": q_90,
        "q_84": q_84,
        "q_50": q_50,
        "q_10": q_10,
        "q_16": q_16,
        "er_90": q_90 - q_50,
        "er_84": q_84 - q_50,
        "er_10": q_50 - q_10,
        "er_16": q_50 - q_16,
    }

    return bands
