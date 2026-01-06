import numpy as np

from bvar.fevd import fevd_from_irf
from bvar.irf import irf_mc, irf_quantiles


def test_irf_shapes_and_quantiles():
    rng = np.random.default_rng(1)
    draws, n, lags = 5, 2, 2

    omegas = np.zeros((draws, n, n))
    for i in range(draws):
        A = rng.normal(size=(n, n))
        omegas[i] = A @ A.T + np.eye(n) * 0.1

    betas = rng.normal(size=(draws, n, n * lags + 1))

    horizon = 4
    irf = irf_mc(omegas, betas, horizon=horizon)
    assert irf.shape == (n, n * (horizon + 1), draws)

    bands = irf_quantiles(irf, horizon=horizon)
    assert bands["q_50"].shape == (n, n, horizon)

    irf_mean = irf.mean(axis=2)
    ws = fevd_from_irf(irf_mean, n, horizon)
    assert ws.shape == (n, n * horizon)
