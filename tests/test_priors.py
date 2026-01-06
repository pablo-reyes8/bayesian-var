import numpy as np

from bvar.priors import ar1_prior_stats, minnesota_dummy_observations


def test_minnesota_dummy_shapes():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(30, 3))
    lags = 2

    delta, su, s0 = ar1_prior_stats(data, lags)
    y_mean = data[: 2 * lags].mean(axis=0)
    lambdas = np.array([0.2, 0.5, 1.0, 1.0, 1.0])

    YP, XP = minnesota_dummy_observations(lambdas, delta, su, s0, y_mean, lags)

    n = data.shape[1]
    expected_rows = n * lags + 2 * n + 2

    assert YP.shape == (expected_rows, n)
    assert XP.shape == (expected_rows, n * lags + 1)
