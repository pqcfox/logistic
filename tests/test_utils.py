import numpy as np

from logistic.utils import hypothesis, sigmoid


def test_sigmoid_handles_small_values():
    assert np.isclose(sigmoid(3), 0.952574127)


def test_sigmoid_handles_large_values():
    assert np.isclose(sigmoid(-100), float('3.72007598e-44'))


def test_sigmoid_accepts_list():
    assert np.isclose(sigmoid(np.array([-1, 1])), 
                      [0.268941421, 0.731058579]).all()


def test_sigmoid_accepts_array():
    array = np.array([[-1, 1], [1, -1]])
    result = np.array([[0.268941421, 0.731058579],
                       [0.731058579, 0.268941421]])
    assert np.isclose(sigmoid(array), result).all()


def test_empty_sigmoid_is_empty():
    assert np.array_equal(sigmoid(np.array([])), np.array([]))


def test_hypothesis_works_on_lists():
    assert np.isclose(hypothesis([2, 1, 3, 7], [5, 4, -4, 0]), 0.880797078)


def test_hypothesis_works_on_arrays():
    X = np.array([[2, 1, 3, 7], [-3, 5, 1, 0]])
    theta = np.array([5, 4, -4, 0])
    result = np.array([0.880797078, 0.731058579])
    assert np.isclose(hypothesis(X, theta), result).all()
                      

def test_empty_hypothesis_is_empty():
    assert np.array_equal(hypothesis(np.array([]), np.array([])), np.array([]))
