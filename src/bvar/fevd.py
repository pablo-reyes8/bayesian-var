from __future__ import annotations

import numpy as np


def fevd_from_irf(irf: np.ndarray, n: int, horizon: int) -> np.ndarray:
    """Forecast Error Variance Decomposition from impulse responses."""
    Ms = np.zeros((n, n * horizon))
    Fs = np.zeros((n, n * horizon))
    Ws = np.zeros((n, n * horizon))

    Mss = irf[:n, :n] ** 2
    Fss = (irf[:n, :n] @ irf[:n, :n].T) * np.eye(n)

    for j in range(1, horizon):
        block = irf[:n, j * n : (j + 1) * n]
        Mss += block**2
        Ms[:, j * n : (j + 1) * n] = Mss

        Fss += (block @ block.T) * np.eye(n)
        Fs[:, j * n : (j + 1) * n] = Fss

        Ws[:, j * n : (j + 1) * n] = np.linalg.solve(Fss, Mss)

    return Ws
