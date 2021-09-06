import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from typing import Callable, Type, Dict
from rlbook.testbeds import Testbed
import warnings
from collections import namedtuple
import logging
from itertools import repeat
from copy import deepcopy
from math import sqrt, log


def init_constant(testbed, q_val=0):
    """Set initial action value estimate as a given constant, defaults to 0"""
    return {a: q_val for a in testbed.expected_values}


class Bandit(metaclass=ABCMeta):
    """Base Bandit class

    Attributes:
        testbed (TestBed class object):
            Testbed object that returns a Reward value for a given Action
        columns (list of strings):
            List of numpy column names to use when outputting results
            as a pandas dataframe.
        action_values (numpy array):
            Stores results of the actions values method.
            Contains Run, Step, Action, and Reward
            Initialized as None, and created with the run method.
        n (int):
            Current step in a run
        Q_init (initialization function):
            Function to use for initializing Q values, defaults to zero init
        Q (dict):
            Action-value estimates in format {action: reward_estimate (float), ...}
        Na (dict):
            Count of how many times an action has been chosen
            {action X: action X count, ...}
        At (int):
            Action that corresponds to the index of the selected testbed arm
    """

    def __init__(self, Q_init: Dict):
        self.columns = [
            "run",
            "step",
            "action",
            "reward",
            "optimal_action",
        ]
        self.action_values = None
        self.n = 1
        self.Q_init = Q_init
        self.Q = deepcopy(Q_init)
        self.Na = {a: 0 for a in self.Q}
        self.At = self.argmax(self.Q)

    def initialization(self, testbed):
        """Reinitialize bandit for a new run when running in serial or parallel"""
        testbed.reset_ev()
        self.n = 1
        self.Q = deepcopy(self.Q_init)
        self.Na = {a: 0 for a in self.Q}
        self.At = self.argmax(self.Q)

    def argmax(self, Q):
        """Return max estimate Q, if tie between actions, choose at random between tied actions"""
        Q_array = np.array(list(self.Q.values()))
        At = np.argwhere(Q_array == np.max(Q_array)).flatten().tolist()

        if len(At) > 1:
            At = np.random.choice(At)
        else:
            At = At[0]

        return list(Q.keys())[At]

    @abstractmethod
    def select_action(self, testbed):
        """Select action logic"""
        pass

    def run(self, testbed, steps, n_runs=1, n_jobs=4, serial=False):
        """Run bandit for specified number of steps and optionally multiple runs"""

        if serial:
            self.action_values = self._serialrun(testbed, steps, n_runs)
        elif n_runs >= 4:
            if n_jobs > cpu_count():
                warnings.warn(
                    f"Warning: running n_jobs: {n_jobs}, with only {cpu_count()} cpu's detected",
                    RuntimeWarning,
                )
            self.action_values = self._multirun(testbed, steps, n_runs, n_jobs=n_jobs)
        else:
            self.action_values = self._serialrun(testbed, steps, n_runs)

    def _serialrun(self, testbed, steps, n_runs):
        action_values = np.empty((steps, len(self.columns), n_runs))
        for k in range(n_runs):
            action_values[:, 0, k] = k
            for n in range(steps):
                action_values[n, 1, k] = n
                action_values[n, 2:, k] = self.select_action(testbed)

            # Reset Q for next run
            self.initialization(testbed)

        return action_values

    def _singlerun(self, testbed, steps, idx_run):
        # Generate different random states for parallel workers
        np.random.seed()

        action_values = np.empty((steps, len(self.columns), 1))
        action_values[:, 0, 0] = idx_run
        for n in range(steps):
            action_values[n, 1, 0] = n
            action_values[n, 2:, 0] = self.select_action(testbed)

        # Reset Q for next run
        self.initialization(testbed)

        return action_values

    def _multirun(self, testbed, steps, n_runs, n_jobs=4):
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            action_values = executor.map(
                self._singlerun,
                repeat(testbed, n_runs),
                [steps for n in range(n_runs)],
                list(range(n_runs)),
            )
        return np.squeeze(np.stack(list(action_values), axis=2))

    def output_df(self):
        """Reshape action_values numpy array and output as pandas dataframe"""
        n_rows = self.action_values.shape[2] * self.action_values.shape[0]
        df = pd.DataFrame(
            data=self.action_values.transpose(2, 0, 1).reshape(-1, len(self.columns)),
            columns=self.columns,
        )

        return df


class EpsilonGreedy(Bandit):
    """Epsilon greedy bandit
    Choose the 'greedy' option that maximizes reward but 'explore' a random action
    for a certain percentage of steps according to the epsilon value

    Attributes:
        epsilon (float):
            epsilon coefficient configuring the probability to explore non-optimal actions,
            ranging from 0.0 to 1.0
        alpha (float or "sample_average"):
            Constant step size ranging from 0.0 to 1.0, resulting in Q being the weighted average
            of past rewards and initial estimate of Q

            Note on varying step sizes such as using 1/n "sample_average":
                self.Q[self.At] = self.Q[self.At] + 1/self.Na[self.At]*(R-self.Q[self.At])
            Theoretically guaranteed to converge, however in practice, slow to converge compared to constant alpha
    """

    def __init__(self, Q_init: Dict, epsilon=0.1, alpha=0.1):
        super().__init__(Q_init)
        self.epsilon = epsilon
        self.alpha = alpha

    def select_action(self, testbed):
        if np.random.binomial(1, self.epsilon) == 1:
            self.At = list(self.Q.keys())[np.random.randint(len(self.Q))]
        else:
            self.At = self.argmax(self.Q)

        A_best = testbed.best_action()
        R = testbed.action_value(self.At)
        self.Na[self.At] += 1
        if self.alpha == "sample_average":
            self.Q[self.At] = self.Q[self.At] + 1 / self.Na[self.At] * (
                R - self.Q[self.At]
            )
        else:
            logging.debug(f"alpha: {self.alpha}, At: {self.At}, R: {R}")
            self.Q[self.At] = self.Q[self.At] + self.alpha * (R - self.Q[self.At])

        self.n += 1

        return (self.At, R, A_best)

    def output_df(self):
        """Reshape action_values numpy array and output as pandas dataframe
        Add epsilon coefficient used for greedy bandit
        """
        df = super().output_df()
        df["epsilon"] = self.epsilon

        return df


class UCL(Bandit):
    """Upper Confidence Limit bandit
    Estimate an upper bound for a given action that includes a measure of uncertainty
    based on how often the action has been chosen in the past
    
    At  = argmax( Qt(a) + c * sqrt(ln(t)/Nt(a)))
    
    Sqrt term is a measure of variance of an action's Upper Bound
    The more often an action is selected, the uncertainty decreases (denominator increases)
    When another action is selected, 
    the uncertainty increases (the numerator since time increase, but in smaller increments due to the ln)

    Attributes:
        c (float):
            c > 0 controls the degree of exploration, specifically the confidence level of a UCL for a given action
        U (dict):
            Action-value uncertainty estimate in format {action: uncertainty (float), ...}
        alpha (float or "sample_average"):
            Constant step size ranging from 0.0 to 1.0, resulting in Q being the weighted average
            of past rewards and initial estimate of Q

            Note on varying step sizes such as using 1/n "sample_average":
                self.Q[self.At] = self.Q[self.At] + 1/self.Na[self.At]*(R-self.Q[self.At])
            Theoretically guaranteed to converge, however in practice, slow to converge compared to constant alpha
    """

    def __init__(self, Q_init: Dict, c=0.1, alpha=0.1):
        """Also use self.Na from base bandit class
        """
        super().__init__(Q_init)
        self.c = c 
        # Initialize self.Na as 1e-100 number instead of 0
        self.Na = {a: 1e-100 for a in self.Na}
        self.alpha = alpha

    def initialization(self, testbed):
        """Reinitialize bandit attributes for a new run"""
        testbed.reset_ev()
        self.n = 1
        self.Q = deepcopy(self.Q_init)
        self.Na = {a: 1e-100 for a in self.Na}

    def select_action(self, testbed):
        logging.debug(f"Na: {self.Na}")
        self.U = {a: Q + self.c * sqrt(log(self.n)/self.Na[a]) for a, Q in self.Q.items()}
        self.At = self.argmax(self.U)

        A_best = testbed.best_action()
        R = testbed.action_value(self.At)
        self.Na[self.At] += 1
        if self.alpha == "sample_average":
            self.Q[self.At] = self.Q[self.At] + 1 / self.Na[self.At] * (
                R - self.Q[self.At]
            )
        else:
            logging.debug(f"alpha: {self.alpha}, At: {self.At}, R: {R}")
            self.Q[self.At] = self.Q[self.At] + self.alpha * (R - self.Q[self.At])

        self.n += 1

        return (self.At, R, A_best)

    def output_df(self):
        """Reshape action_values numpy array and output as pandas dataframe
        Add epsilon coefficient used for greedy bandit
        """
        df = super().output_df()
        df["c"] = self.c

        return df
