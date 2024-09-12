"""Tests for `rlbook.bandits` package."""

import pytest

from pandas.testing import assert_series_equal

from rlbook.bandits import EpsilonGreedy, init_constant
from rlbook.bandits.testbeds import NormalTestbed

EXPECTED_VALUES = {
    1: {"mean": 2, "var": 1},
    2: {"mean": -1, "var": 1},
    3: {"mean": 1, "var": 1},
    4: {"mean": 0, "var": 1},
    5: {"mean": 1.7, "var": 1},
}


@pytest.fixture
def testbed_fixed():
    return NormalTestbed(EXPECTED_VALUES, p_drift=0)

@pytest.fixture
def egreedy_bandit(testbed_fixed):
    return EpsilonGreedy(init_constant(testbed_fixed, q_val=10), epsilon=0.2)


def test_multirun_bandit_randomness(egreedy_bandit, testbed_fixed):
    """Test that parallel runs are using different random seeds resulting in different actions"""

    egreedy_bandit.run(testbed_fixed, 20, n_runs=20, n_jobs=4)
    df = egreedy_bandit.output_df()
    
    # Pivot results:
    # run   0 1 2 3
    # step
    #  0    a a a a
    #  1    a a a a
    #  2    a a a a
    # where a = action taken
    actions_by_run = df[["run", "step", "action"]].pivot(index="step", columns=["run"], values="action")

    assert not all(actions_by_run[0].eq(actions_by_run[1]))
    

