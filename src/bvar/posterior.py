from __future__ import annotations

import numpy as np
from scipy.stats import invwishart, matrix_normal


def posterior_params(Y, X, YP, XP):
    """Compute posterior parameters for conjugate VAR with dummy observations."""
    Yst = np.vstack((Y, YP))
    Xst = np.vstack((X, XP))

    Bps = np.linalg.solve(Xst.T @ Xst, Xst.T @ Yst)
    Hs = np.linalg.inv(Xst.T @ Xst)
    resid = Yst - (Xst @ Bps)
    Ss = resid.T @ resid

    n = Y.shape[1]
    v0 = n + 2
    vd = Y.shape[0] + v0
    return Bps, Hs, Ss, vd


def sample_posterior(Bps, Hs, Ss, vd, draws, seed=None):
    """Draw (Omega, B) samples from the posterior distribution."""
    rng = np.random.default_rng(seed)
    n = Ss.shape[0]
    k = Bps.shape[0]

    omegas = np.zeros((draws, n, n))
    betas = np.zeros((draws, n, k))

    for i in range(draws):
        omega = invwishart.rvs(df=vd, scale=Ss, random_state=rng)
        beta = matrix_normal.rvs(Bps, Hs, omega, random_state=rng)
        omegas[i] = omega
        betas[i] = beta.T

    return omegas, betas
