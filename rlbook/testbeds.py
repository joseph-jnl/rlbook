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

