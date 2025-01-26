"""Tests for `rlbook.gridworlds` package."""

import pytest
from jax.numpy import allclose, array

from rlbook.gridworlds.grids import OptimalGrid, RandomGrid

random_3_2_answer = array(
    [
        [3.3089964, 8.789292, 4.4276195, 5.3223677, 1.4921787],
        [1.5215881, 2.992318, 2.25014, 1.9075718, 0.54740274],
        [0.05082268, 0.73817074, 0.67311335, 0.35818636, -0.403141],
        [-0.9735919, -0.4354951, -0.354882, -0.5856047, -1.1830747],
        [-1.8577, -1.3452308, -1.2292669, -1.4229177, -1.9751786],
    ]
)

optimal_3_8_answer = array(
    [
        [21.977476, 24.419418, 21.977476, 19.419418, 17.477476],
        [19.779728, 21.977476, 19.779728, 17.801754, 16.021578],
        [17.801754, 19.779728, 17.801754, 16.021578, 14.419419],
        [16.021578, 17.801754, 16.021578, 14.419419, 12.977477],
        [14.419419, 16.021578, 14.419419, 12.977477, 11.679729],
    ]
)


@pytest.mark.jax
def test_random_state_value_function():
    grid = RandomGrid(
        [[0, 0], [1, 3]], [[4, 2], [1, 3]], array([10, 5]), n_rows=5, n_cols=5
    )
    state_value_estimate = grid.estimate_state_value()
    assert allclose(state_value_estimate, random_3_2_answer)


@pytest.mark.jax
def test_optimal_value_function():
    grid = OptimalGrid(
        [[0, 0], [1, 3]], [[4, 2], [1, 3]], array([10, 5]), n_rows=5, n_cols=5
    )
    state_value_estimate = grid.estimate_state_value()
    assert allclose(state_value_estimate, optimal_3_8_answer)
