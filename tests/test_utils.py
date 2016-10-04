import numpy as np

from logistic.utils import hypothesis, sigmoid


def test_sigmoid_handles_small_values():
    assert np.isclose(sigmoid(3), 0.952574127)


def test_sigmoid_handles_large_values():
    assert np.isclose(sigmoid(-100), float('3.72007598e-44'))


def test_sigmoid_accepts_ndarray():
    assert np.isclose(sigmoid(np.array([-1, 1])), 
                      np.array([0.268941421, 0.731058579])).all()


# def test_hypothesis():
#     assert hypothesis(np.array([]), np.array([])) == np.array([])
