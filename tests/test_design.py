import numpy as np

from bvar.design import build_var_design


def test_build_var_design_shapes():
    data = np.arange(40).reshape(20, 2)
    Y, X = build_var_design(data, lags=3)

    assert Y.shape == (17, 2)
    assert X.shape == (17, 2 * 3 + 1)
    assert np.allclose(X[:, -1], 1.0)
