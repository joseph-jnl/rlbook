from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Dict

import numpy as np
import pandas as pd


class Testbed(metaclass=ABCMeta):
    """Base Testbed class

    Attributes:
        expected_values (dict):
            Dict of parameters describing the Testbed distribution
    """

    def __init__(self, expected_values):
        self.initial_ev = expected_values
        self.expected_values = deepcopy(self.initial_ev)

    def estimate_distribution(self, n=1000) -> pd.DataFrame:
        """Provide an estimate of the testbed values across all arms
        n (int): Number of iterations to execute in testbed
        """
        self.p_drift = 0.0
        R_dfs = []
        for a in self.expected_values:
            Ra = pd.DataFrame(self.action_value(a, shape=(n, 1)), columns=["reward"])
            Ra["action"] = a
            Ra["strategy"] = "uniform"
            R_dfs.append(Ra)
        # Also include initial EV if pdrift shifted EVs
        if self.initial_ev != self.expected_values:
            self.expected_values = deepcopy(self.initial_ev)
            for a in self.initial_ev:
                Ra = pd.DataFrame(
                    self.action_value(a, shape=(n, 1)), columns=["reward"]
                )
                Ra["action"] = a
                Ra["strategy"] = "uniform"
                R_dfs.append(Ra)
        R = pd.concat(R_dfs)
        return R

    def reset_ev(self):
        self.expected_values = deepcopy(self.initial_ev)

    def best_action(self):
        """Return true best action that should have been taken based on EV state"""

        A_best = list(self.expected_values.keys())[
            np.argmax([ev["mean"] for ev in self.expected_values.values()])
        ]
        return A_best

    @abstractmethod
    def action_value(self, action, shape=None) -> np.ndarray or float:
        """Return reward value given action"""
        pass


class NormalTestbed(Testbed):
    """Return random value from a Normal Distribution according to expected value config

    Attributes:
        expected_values (dict):
            Dict of means and variances describing Normal Distribution of each arm in the testbed
            Example:
                expected_values = {1: {'mean': 0.5, 'var': 1}, 2: {'mean': 1, 'var': 1}}
        p_drift (float):
            Probability for underlying reward to change ranging from 0.0 to 1.0, defaults to 0
        drift_mag (float):
            Magnitude of reward change when drifting, defaults to 1.0
    """

    def __init__(self, expected_values: Dict, p_drift=0.0, drift_mag=1.0):
        self.p_drift = p_drift
        self.drift_mag = drift_mag
        super().__init__(expected_values)

    def action_value(self, action, shape=None) -> np.ndarray or float:
        """Return reward value given action"""
        if np.random.binomial(1, self.p_drift) == 1:
            A_drift = list(self.expected_values.keys())[
                np.random.randint(len(self.expected_values))
            ]
            self.expected_values[A_drift]["mean"] = self.expected_values[A_drift][
                "mean"
            ] + self.drift_mag * (np.random.random() - 0.5)

        return np.random.normal(
            loc=self.expected_values[action]["mean"],
            scale=self.expected_values[action]["var"],
            size=shape,
        )
