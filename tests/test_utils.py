import numpy as np

from logistic.utils import sigmoid


def test_sigmoid():
    assert sigmoid(-1) == 1 / (1 + np.e)
    assert sigmoid(0) == 0.5
    assert sigmoid(1) == 1 / (1 + 1/np.e)
