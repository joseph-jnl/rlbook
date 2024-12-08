from abc import ABCMeta, abstractmethod

import jax.numpy as jnp
from jax import jit, tree_util
from jax.scipy.signal import convolve2d
from jaxtyping import Array, Float, Int


class Grid(metaclass=ABCMeta):
    """ """

    def __init__(self, n_rows: int = 5, n_cols: int = 5):
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
        special_states: list[list[int, int]],
        special_states_prime: list[list[int, int]],
        special_states_rewards: Int[Array, "{len(special_states)}"],
        actions: Float[Array, "2 4"] = jnp.array([[-1, 1, 0, 0], [0, 0, 1, -1]]),
        actions_probs: Float[Array, "1 4"] = jnp.array([0.25, 0.25, 0.25, 0.25]),
        n_rows: int = 5,
        n_cols: int = 5,
        R: Float[Array, "n_rows n_cols"] = None,
        P: Float[Array, "3 3"] = None,
        v: Float[Array, "n_rows n_cols"] = None,
    ):
        self.special_states = special_states
        self.special_states_rewards = special_states_rewards
        self.special_states_prime = special_states_prime
        self.actions = actions
        self.actions_probs = actions_probs

        self.v = jnp.zeros((n_rows, n_cols))
        self.P = self.policy()
        self.R = self.reward()

    def _tree_flatten(self):
        children = (self.v,)  # arrays / dynamic values
        # static values
        aux_data = {
            "R": self.R,
            "P": self.P,
            "special_states": self.special_states,
            "special_states_prime": self.special_states_prime,
            "special_states_rewards": self.special_states_rewards,
            "actions": self.actions,
            "actions_probs": self.actions_probs,
        }

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        grid = cls(
            aux_data["special_states"],
            aux_data["special_states_prime"],
            aux_data["special_states_rewards"],
            actions=aux_data["actions"],
            actions_probs=aux_data["actions_probs"],
            R=aux_data["R"],
            P=aux_data["P"],
            v=children[0],
        )

        return grid

    def estimate_state_value(self, iter=1000):
        """"""
        for i in range(iter):
            vp = self.state_value(
                self.v,
                self.R,
                self.P,
                self.special_states,
                self.special_states_prime,
                self.special_states_rewards,
                self.actions_probs,
            )
            self.v = vp

    def policy(self):
        """"""
        policy = jnp.zeros((3, 3))
        policy = policy.at[self.actions[0] + 1, self.actions[1] + 1].set(
            self.actions_probs
        )

        return policy

    def reward(self):
        """Provides reward for all states in grid"""
        R = convolve2d(
            jnp.pad(self.v, pad_width=(1, 1), constant_values=-1),
            self.P,
            mode="valid",
        )
        R = R.at[self.special_states[0], self.special_states[1]].set(
            self.special_states_rewards
        )

        return R

    @jit
    def state_value(
        self,
        v,
        R,
        P,
        special_states,
        special_states_prime,
        special_states_rewards,
        actions_probs,
    ):
        # Update interior grid
        vp = (
            R
            + convolve2d(
                jnp.pad(v, pad_width=(1, 1), constant_values=0),
                P,
                mode="valid",
            )
            * 0.9
        )

        # Update edges except for corners
        vp = vp.at[1:-1, 0].add(v[1:-1, 0] * 0.9 * 0.25)
        vp = vp.at[1:-1, -1].add(v[1:-1, -1] * 0.9 * 0.25)
        vp = vp.at[0, 1:-1].add(v[0, 1:-1] * 0.9 * 0.25)
        vp = vp.at[-1, 1:-1].add(v[-1, 1:-1] * 0.9 * 0.25)

        # Update corners
        vp = vp.at[0, 0].add(v[0, 0] * 2 * 0.9 * 0.25)
        vp = vp.at[0, -1].add(v[0, -1] * 2 * 0.9 * 0.25)
        vp = vp.at[-1, -1].add(v[-1, -1] * 2 * 0.9 * 0.25)
        vp = vp.at[-1, 0].add(v[-1, 0] * 2 * 0.9 * 0.25)

        # Update special states
        vp = vp.at[special_states[0], special_states[1]].set(
            jnp.sum(
                actions_probs
                * v[special_states_prime[0], special_states_prime[1]].reshape(2, 1),
                axis=1,
            )
            * 0.9
            + special_states_rewards
        )

        return vp


tree_util.register_pytree_node(
    SimpleGrid, SimpleGrid._tree_flatten, SimpleGrid._tree_unflatten
)
