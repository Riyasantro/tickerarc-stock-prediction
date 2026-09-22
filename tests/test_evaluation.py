import numpy as np

from scripts.evaluate_rl import max_drawdown


def test_max_drawdown():
    value = max_drawdown([0.10, -0.05, -0.10, 0.04])
    assert np.isclose(value, -0.145)
