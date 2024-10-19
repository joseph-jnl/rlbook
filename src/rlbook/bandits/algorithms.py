import logging
import warnings
from abc import ABCMeta, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import repeat
from math import ceil, log, sqrt
from multiprocessing import cpu_count
from typing import Dict

import numpy as np
import numpy.typing as npt


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
        Q_init:
            Numpy array of initial Q values with size n matching n actions available in testbed
        Q:
            Numpy array of Q values with size n matching n actions available in testbed
        Qn:
            Length of Q array
        Na:
            Numpy array with count of how many times an action has been chosen
        At (int):
            Action that corresponds to the index of the selected testbed arm
    """

    def __init__(self, Q_init: npt.ArrayLike):
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
        self.Qn = self.Q.shape[0]
        self.Na = np.zeros((Q_init.size), dtype=int)
        self.At = np.argmax(self.Q)

    def initialization(self, testbed):
        """Reinitialize bandit for a new run when running in serial or parallel"""
        testbed.reset_ev()
        self.n = 1
        self.Q = deepcopy(self.Q_init)
        self.Na = np.zeros((self.Q.size), dtype=int)
        self.At = np.argmax(self.Q)

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
                chunksize=ceil(n_runs / n_jobs),
            )
        return np.squeeze(np.stack(list(action_values), axis=2))

    def output_av(self) -> tuple[npt.ArrayLike, list[str]]:
        """Output action_values numpy array reshaped from 3D to 2D and columns names"""

        return (
            self.action_values.transpose(2, 0, 1).reshape(-1, len(self.columns)),
            self.columns,
        )


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
        logging.debug("Q: %s", self.Q)
        if np.random.binomial(1, self.epsilon) == 1:
            self.At = np.random.randint(self.Qn)
        else:
            self.At = np.argmax(self.Q)

        A_best = testbed.best_action()
        R = testbed.action_value(self.At)
        self.Na[self.At] += 1
        if self.alpha == "sample_average":
            self.Q[self.At] = self.Q[self.At] + 1 / self.Na[self.At] * (
                R - self.Q[self.At]
            )
        else:
            logging.debug("alpha: %s, At: %s, R: %s", self.alpha, self.At, R)
            self.Q[self.At] = self.Q[self.At] + self.alpha * (R - self.Q[self.At])

        self.n += 1
        return (self.At, R, A_best)

    def output_av(self):
        """Output action_values numpy array reshaped from 3D to 2D and columns names"""
        arr, cols = super().output_av()
        epsilon = np.ones((arr.shape[0], 1)) * self.epsilon 
        arr_stacked = np.column_stack((arr, epsilon))
        cols.append("epsilon")

        return arr_stacked, cols


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
        """ """
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
        logging.debug("Na: %s", self.Na)
        self.U = {
            a: Q + self.c * sqrt(log(self.n) / self.Na[a]) for a, Q in self.Q.items()
        }
        self.At = self.argmax(self.U)

        A_best = testbed.best_action()
        R = testbed.action_value(self.At)
        self.Na[self.At] += 1
        if self.alpha == "sample_average":
            self.Q[self.At] = self.Q[self.At] + 1 / self.Na[self.At] * (
                R - self.Q[self.At]
            )
        else:
            logging.debug("alpha: %s, At: %s, R: %s", self.alpha, self.At, R)
            self.Q[self.At] = self.Q[self.At] + self.alpha * (R - self.Q[self.At])

        self.n += 1

        return (self.At, R, A_best)

    def output_av(self):
        """Output action_values numpy array reshaped from 3D to 2D and columns names"""
        df = super().output_av()
        df["c"] = self.c

        return df


class Gradient(Bandit):
    """Gradient bandit
    Learn a set of numerical preferences "H" rather than estimate a set of action values "Q"
    H preferences are all relative to each other, no correlation to a potential reward

    Update H using:
    Ht+1(At) = Ht(At) + lr * (Rt - Q[At]) * (1 - softmax(At)) for At
    Ht+1(a) = Ht(a) + lr * (Rt - Q[At]) * softmax(a) for all a != At
    where At is action chosen

    Attributes:
        H (dict):
            Action-value uncertainty estimate in format {action: uncertainty (float), ...}
        lr (float between 0.0-1.0):
            learning rate, step size to update H
        alpha (float or "sample_average"):
            Constant step size ranging from 0.0 to 1.0, resulting in Q being the weighted average
            of past rewards and initial estimate of Q

            Note on varying step sizes such as using 1/n "sample_average":
                self.Q[self.At] = self.Q[self.At] + 1/self.Na[self.At]*(R-self.Q[self.At])
            Theoretically guaranteed to converge, however in practice, slow to converge compared to constant alpha
    """

    def __init__(self, Q_init: Dict, lr=0.1, alpha=0.1):
        """ """
        super().__init__(Q_init)
        self.lr = lr
        self.alpha = alpha
        self.H = deepcopy(self.Q_init)

    def initialization(self, testbed):
        """Reinitialize bandit attributes for a new run"""
        testbed.reset_ev()
        self.n = 1
        self.H = deepcopy(self.Q_init)
        self.Q = deepcopy(self.Q_init)
        self.Na = {a: 0 for a in self.Q}

    def softmax(self, H):
        h = np.array([val for val in H.values()])
        probs = np.exp(h) / sum(np.exp(h))
        return dict(zip(H.keys(), probs))

    def select_action(self, testbed):
        """
        Select At based on H prob

        Then update H via:
        Ht+1(At) = Ht(At) + lr * (Rt - Q[At]) * (1 - softmax(At)) for At
        Ht+1(a) = Ht(a) + lr * (Rt - Q[At]) * softmax(a) for all a != At
        where At is action chosen
        """
        probs = self.softmax(self.H)
        logging.debug("probs: %s", probs)
        self.At = int(np.random.choice(list(self.H.keys()), 1, p=list(probs.values())))

        A_best = testbed.best_action()
        R = testbed.action_value(self.At)
        self.Na[self.At] += 1
        logging.debug("H: %s", self.H)
        logging.debug("Q: %s", self.Q)
        for a in self.H:
            if a == self.At:
                self.H[a] = self.H[a] + self.lr * (R - self.Q[a]) * (1 - probs[a])
            else:
                self.H[a] = self.H[a] - self.lr * (R - self.Q[a]) * (probs[a])

        if self.alpha == "sample_average":
            self.Q[self.At] = self.Q[self.At] + 1 / self.Na[self.At] * (
                R - self.Q[self.At]
            )
        else:
            logging.debug("alpha: %s, At: %s, R: %s", self.alpha, self.At, R)
            self.Q[self.At] = self.Q[self.At] + self.alpha * (R - self.Q[self.At])

        self.n += 1

        return (self.At, R, A_best)

    def output_av(self):
        """Output action_values numpy array reshaped from 3D to 2D and columns names"""
        df = super().output_av()
        df["lr"] = self.lr

        return df
