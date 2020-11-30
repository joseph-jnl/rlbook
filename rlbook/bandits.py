import pandas as pd
import numpy as np


class NormalTestbed:
    """Return random value according to expected value config
    
    Attributes:
        expected_values (dict): 
            Dict of means and variances of each arm in the testbed
            Example:
                expected_values = {1: {'mean': 0.5, 'var': 1}, 2: {'mean': 1, 'var': 1}}
        p_drift (float): 
            Probability for underlying reward to change ranging from 0.0 to 1.0, defaults to 0
        drift_mag (float): 
            Magnitude of reward change when drifting, defaults to 1.0
    """
    
    def __init__(self, expected_values, p_drift=0., drift_mag=1.0):
        self.expected_values = expected_values
        self.p_drift=p_drift
        self.drift_mag=drift_mag
        
    def action_value(self, action, shape=None) -> np.ndarray or float:
        """Return reward value given action
        """
        if np.random.binomial(1, self.p_drift) == 1:
            A_drift = np.random.choice(list(self.expected_values.keys()))
            self.expected_values[A_drift]['mean'] = self.expected_values[A_drift]['mean'] + self.drift_mag*(np.random.random()-0.5)   
        
        return np.random.normal(loc=self.expected_values[action]['mean'], 
                                scale=self.expected_values[action]['var'],
                                size=shape)
    
    def estimate_distribution(self, n=1000) -> pd.DataFrame:
        """Provide an estimate of the normal testbed values across all arms
        n (int): Number of iterations to execute in testbed
        """
        R = pd.DataFrame(columns=['Reward', 'Action', 'Strategy'])
        for a in self.expected_values:
            Ra = pd.DataFrame(self.action_value(a, shape=(n, 1)), columns=['Reward'])
            Ra['Action'] = a
            Ra['Strategy'] = 'uniform'
            R = pd.concat([R, Ra])
        return R


class Bandit:
    """Base Bandit class

    Attributes:
        testbed (TestBed object):
            Testbed object that returns a Reward value for a given Action
        columns (list of strings): 
            List of numpy column names to use when outputting results
            as a pandas dataframe.
        action_values (numpy array): 
            Stores results of the actions values method. 
            By default will contain Step, Action, and Reward
            Initialized as None, and created with the run method.
    """
    
    def __init__(self, testbed):
        self.testbed = testbed
        self.columns = ['Step', 'Action', 'Reward']
        self.results = None

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

    def select_action(self):
        raise NotImplementedError()

    def run(self, steps, k=1):
        """Run bandit k times for specified number of steps
        """
        self.actions_rewards = np.empty((steps, 2))
        for n in range(steps):
            self.actions_rewards[n, 0], self.actions_rewards[n, 1] = self.select_action()


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
        self.Q = {a: 0 for a in self.testbed.expected_values}
        self.nQ = {a: 0 for a in self.Q}
        self.At = self.argmax(self.Q)
    
    def select_action(self):
        if np.random.binomial(1, self.epsilon) == 1:
            self.At = np.random.choice(list(self.Q.keys()))
        else:
            self.At = self.argmax(self.Q)
        R = self.testbed.action_value(self.At)
        self.nQ[self.At] += 1 
        self.Q[self.At] = self.Q[self.At] + 1/self.nQ[self.At]*(R-self.Q[self.At])
        
        return (self.At, R)

        
    
