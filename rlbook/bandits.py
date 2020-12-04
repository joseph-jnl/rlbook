import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod


class Bandit(metaclass=ABCMeta):
    """Base Bandit class

    Attributes:
        testbed (TestBed object):
            Testbed object that returns a Reward value for a given Action
        columns (list of strings): 
            List of numpy column names to use when outputting results
            as a pandas dataframe.
        action_values (numpy array): 
            Stores results of the actions values method. 
            Contains Run, Step, Action, and Reward
            Initialized as None, and created with the run method.
    """
    
    def __init__(self, testbed):
        self.testbed = testbed
        self.columns = ['Run', 'Step', 'Action', 'Reward',]
        self.action_values = None
        self.Q = {a: 0 for a in self.testbed.expected_values}
        self.nQ = {a: 0 for a in self.Q}
        self.At = self.argmax(self.Q)

    def argmax(self, Q):
        """Return max estimate Q, if tie between actions, choose at random between tied actions
        """
        Q_array = np.array(list(self.Q.values()))
        At = np.argwhere(Q_array == np.max(Q_array)).flatten().tolist()
        
        if len(At) > 1:
            At = np.random.choice(At)
        else:
            At = At[0]
        
        return list(Q.keys())[At]

    @abstractmethod
    def select_action(self):
        """Select action logic
        """
        pass

    def run(self, steps, runs=1):
        """Run bandit for specified number of steps and optionally multiple runs
        """
        self.action_values = np.empty((steps, 4, runs))
        for k in range(runs):
            self.action_values[:, 0, k] = k
            for n in range(steps):
                self.action_values[n, 1, k] = n
                self.action_values[n, 2, k], self.action_values[n, 3, k] = self.select_action()

            # Reset Q for next run
            self.Q = {a: 0 for a in self.testbed.expected_values}

    def output_df(self):
        """Reshape action_values numpy array and output as pandas dataframe
        """
        n_rows = self.action_values.shape[2]*self.action_values.shape[0]
        df = pd.DataFrame(
            data=self.action_values.transpose(2, 0, 1).reshape(-1, 4), 
            columns=self.columns
        )

        return df 


class EpsilonGreedy(Bandit):
    """Epsilon greedy bandit

    Attributes:
        epsilon (float): 
            epsilon coefficient configuring the probability to explore non-optimal actions,
            ranging from 0.0 to 1.0
    """

    def __init__(self, testbed, epsilon=0.1):
        super().__init__(testbed)
        self.epsilon = epsilon 
    
    def select_action(self):
        if np.random.binomial(1, self.epsilon) == 1:
            self.At = np.random.choice(list(self.Q.keys()))
        else:
            self.At = self.argmax(self.Q)
        R = self.testbed.action_value(self.At)
        self.nQ[self.At] += 1 
        self.Q[self.At] = self.Q[self.At] + 1/self.nQ[self.At]*(R-self.Q[self.At])
        
        return (self.At, R)

    def output_df(self):
        """Reshape action_values numpy array and output as pandas dataframe
        """
        df = super().output_df()
        df['epsilon'] = self.epsilon

        return df 
