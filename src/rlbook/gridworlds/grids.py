from abc import ABCMeta

import jax.numpy as jnp
from jax import jit, tree_util
from jax.scipy.signal import convolve2d
from jaxtyping import Array, Float, Int


class Grid(metaclass=ABCMeta):
    """ """

    def __init__(
        self,
        n_rows: int = 5,
        n_cols: int = 5,
        actions: Float[Array, "2 4"] = jnp.array([[-1, 1, 0, 0], [0, 0, 1, -1]]),
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.actions = actions


class SimpleGrid(Grid):
    """"""

    def __init__(
        self,
        special_states: list[list[int, int]],
        special_states_prime: list[list[int, int]],
        special_states_rewards: Int[Array, "{len(special_states)}"],
        n_rows: int = 5,
        n_cols: int = 5,
        policy: str = "random",
        R: Float[Array, "n_rows n_cols"] = None,
        P: Float[Array, "3 3"] = None,
        v: Float[Array, "n_rows n_cols"] = None,
    ):
        super().__init__()
        self.special_states = special_states
        self.special_states_rewards = special_states_rewards
        self.special_states_prime = special_states_prime

        self.v = jnp.zeros((n_rows, n_cols))
        if policy == "random":
            self.P = self._policy_random()
            self.R = self._reward_random()

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
        }

        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        grid = cls(
            aux_data["special_states"],
            aux_data["special_states_prime"],
            aux_data["special_states_rewards"],
            R=aux_data["R"],
            P=aux_data["P"],
            v=children[0],
        )

        return grid

    def _policy_random(self):
        """"""
        policy = jnp.zeros((3, 3))
        policy = policy.at[self.actions[0] + 1, self.actions[1] + 1].set(0.25)

        return policy

    def estimate_state_value(self, iter=1000):
        """"""
        for _ in range(iter):
            vp = self.state_value(
                self.v,
                self.R,
                self.P,
                self.special_states,
                self.special_states_prime,
                self.special_states_rewards,
            )
            self.v = vp

    def _reward_random(self):
        """Provides reward for all states in grid when following a random policy"""
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
    ):
        """"""
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
                v[special_states_prime[0], special_states_prime[1]],
            )
            * 0.9
            + special_states_rewards
        )

        return vp


tree_util.register_pytree_node(
    SimpleGrid, SimpleGrid._tree_flatten, SimpleGrid._tree_unflatten
)
