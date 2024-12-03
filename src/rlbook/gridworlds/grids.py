from abc import ABCMeta, abstractmethod

import jax.numpy as jnp
import jax.tree as jtree
from jax.scipy.signal import convolve2d
from jaxtyping import Array, Float


class Grid(metaclass=ABCMeta):
    """ """

    def __init__(self, n_rows: int = 5, n_cols: int = 5, iter: int = 1000):
        self.v = jnp.zeros((n_rows, n_cols))
        self.vp = jnp.zeros((n_rows, n_cols))
        self.n_rows = n_rows
        self.n_cols = n_cols

    @abstractmethod
    def reward(self):
        """Reward for a given action"""

    @abstractmethod
    def state_value(self):
        """"""


class SimpleGrid(Grid):
    """"""

    def __init__(
        self,
        special_states: list[tuple[int, int]],
        special_states_rewards: list[int],
        special_states_prime: list[tuple[int, int]],
        actions: Float[Array, "2 4"] = jnp.array(
            [
                [
                    -1,
                    1,
                    0,
                    0,
                ],
                [0, 0, 1, -1],
            ]
        ),
        actions_probs: Float[Array, "1 4"] = jnp.array([0.25, 0.25, 0.25, 0.25]),
        n_rows: int = 5,
        n_cols: int = 5,
        iter: int = 1000,
    ):
        self.special_states = special_states
        self.special_states_rewards = special_states_rewards
        self.special_states_t = jtree.transpose(
            jtree.structure(["*", "*"]), None, special_states
        )
        self.special_states_prime = special_states_prime
        self.special_states_prime_t = jtree.transpose(
            jtree.structure(["*", "*"]), None, special_states_prime
        )
        self.actions = actions
        self.actions_probs = actions_probs

        super().__init__(n_rows=n_rows, n_cols=n_cols, iter=iter)

        self.R = self.reward()

    def policy(self):
        """"""

        return policy

    def reward(self):
        """Provides reward for all states in grid"""
        R = convolve2d(
            jnp.pad(self.v, pad_width=(1, 1), constant_values=-1),
            self.actions,
            mode="valid",
        )
        R = R.at[self.special_states_t[0], self.special_states_t[1]].set(
            self.special_states_rewards
        )

        return R

    def state_value(
        self,
    ):
        return 0
