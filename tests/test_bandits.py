"""Tests for `rlbook.bandits` package."""

import numpy as np
import pytest

from rlbook.bandits.algorithms import EpsilonGreedy
from rlbook.bandits.testbeds import NormalTestbed

EXPECTED_VALUES = {
    0: {"mean": 2, "std": 1},
    1: {"mean": -1, "std": 1},
    2: {"mean": 1, "std": 1},
    3: {"mean": 0, "std": 1},
    4: {"mean": 1.7, "std": 1},
}


@pytest.fixture
def testbed_fixed():
    return NormalTestbed(EXPECTED_VALUES, p_drift=0)


@pytest.fixture
def egreedy_bandit(testbed_fixed):
    return EpsilonGreedy(np.zeros(testbed_fixed.expected_values["mean"].size))


def test_multirun_bandit_randomness(egreedy_bandit, testbed_fixed):
    """Test that parallel runs are using different random seeds resulting in different actions"""

    egreedy_bandit.run(testbed_fixed, 20, n_runs=20, n_jobs=4)
    av = egreedy_bandit.output_av()[0]

    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(av[0:20, 2], av[20:40, 2])
