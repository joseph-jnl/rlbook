import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from typing import Callable, Type
from rlbook.testbeds import Testbed
import secrets
import warnings


def init_zero(testbed):
    """"""
    return {a: 0 for a in testbed.expected_values}


def init_optimistic(testbed):
    """"""
    op_val = abs(max([testbed.action_value(a) for a in testbed.expected_values])) * 5
    testbed.reset_ev()

    return {a: int(op_val) for a in testbed.expected_values}


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
        Q_init (initialization function):
            Function to use for initializing Q values, defaults to zero init
    """

    def __init__(self, testbed: Type[Testbed], Q_init: Callable = init_zero):
        self.testbed = testbed
        self.columns = [
            "Run",
            "Step",
            "Action",
            "Reward",
            "Optimal_Action",
        ]
        self.action_values = None
        self.Q_init = Q_init
        self.initialization()

    def initialization(self):
        self.testbed.reset_ev()
        self.Q = self.Q_init(self.testbed)
        self.nQ = {a: 0 for a in self.Q}
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
    def select_action(self):
        """Select action logic"""
        pass

    def run(self, steps, n_runs=1, n_jobs=4, serial=False):
        """Run bandit for specified number of steps and optionally multiple runs"""

        if serial:
            self.action_values = self._serialrun(steps, n_runs)
        elif n_runs >= 4:
            if n_jobs > cpu_count():
                warnings.warn(f"Warning: running n_jobs: {n_jobs}, with only {cpu_count()} cpu's detected", RuntimeWarning)
            self.action_values = self._multirun(steps, n_runs, n_jobs=n_jobs)
        else:
            self.action_values = self._serialrun(steps, n_runs)

    def _serialrun(self, steps, n_runs):
        action_values = np.empty((steps, 5, n_runs))
        for k in range(n_runs):
            action_values[:, 0, k] = k
            for n in range(steps):
                action_values[n, 1, k] = n
                action_values[n, 2, k], action_values[n, 3, k], action_values[n, 4, k]  = self.select_action()

            # Reset Q for next run
            self.initialization()

        return action_values

    def _singlerun(self, steps, idx_run):
        # Generate different random states for parallel workers
        np.random.seed()

        action_values = np.empty((steps, 5, 1))
        action_values[:, 0, 0] = idx_run
        for n in range(steps):
            action_values[n, 1, 0] = n
            action_values[n, 2, 0], action_values[n, 3, 0], action_values[n, 4, 0]  = self.select_action()

        # Reset Q for next run
        self.initialization()

        return action_values

    def _multirun(self, steps, n_runs, n_jobs=4):
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            action_values = executor.map(
                self._singlerun, [steps for n in range(n_runs)], list(range(n_runs))
            )
        return np.squeeze(np.stack(list(action_values), axis=2))

    def output_df(self):
        """Reshape action_values numpy array and output as pandas dataframe"""
        n_rows = self.action_values.shape[2] * self.action_values.shape[0]
        df = pd.DataFrame(
            data=self.action_values.transpose(2, 0, 1).reshape(-1, 5),
            columns=self.columns,
        )

        return df


class EpsilonGreedy(Bandit):
    """Epsilon greedy bandit

    Attributes:
        epsilon (float):
            epsilon coefficient configuring the probability to explore non-optimal actions,
            ranging from 0.0 to 1.0
        alpha (float):
            Constant step size ranging from 0.0 to 1.0, resulting in Q being the weighted average
            of past rewards and initial estimate of Q

            Note on varying step sizes such as using 1/n:
                self.Q[self.At] = self.Q[self.At] + 1/self.nQ[self.At]*(R-self.Q[self.At])
            Theoretically guaranteed to converge, however in practice, slow to converge compared to constant alpha
    """

    def __init__(self, testbed, Q_init: Callable = init_zero, epsilon=0.1, alpha=0.1):
        super().__init__(testbed, Q_init)
        self.epsilon = epsilon
        self.alpha = alpha

    def select_action(self):
        if np.random.binomial(1, self.epsilon) == 1:
            self.At = np.random.choice(list(self.Q.keys()))
        else:
            self.At = self.argmax(self.Q)
        A_best = np.argmax([ev['mean'] for ev in self.testbed.expected_values.values()])
        R = self.testbed.action_value(self.At)
        self.nQ[self.At] += 1
        if self.alpha is None:
            self.Q[self.At] = self.Q[self.At] + 1 / self.nQ[self.At] * (
                R - self.Q[self.At]
            )
        else:
            self.Q[self.At] = self.Q[self.At] + self.alpha * (R - self.Q[self.At])

        return (self.At, R, A_best)

    def output_df(self):
        """Reshape action_values numpy array and output as pandas dataframe
        Add epsilon coefficient used for greedy bandit
        """
        df = super().output_df()
        df["epsilon"] = self.epsilon

        return df
