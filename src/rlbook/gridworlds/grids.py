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

    def _v_init(self):
        return jnp.zeros((self.n_rows, self.n_cols))


class RandomGrid(Grid):
    """"""

    def __init__(
        self,
        special_states: list[list[int, int]],
        special_states_prime: list[list[int, int]],
        special_states_rewards: Int[Array, "{len(special_states)}"],
        n_rows: int = 5,
        n_cols: int = 5,
        R: Float[Array, "n_rows n_cols"] = None,
        P: Float[Array, "3 3"] = None,
    ):
        super().__init__(n_rows=n_rows, n_cols=n_cols)
        self.special_states = special_states
        self.special_states_rewards = special_states_rewards
        self.special_states_prime = special_states_prime

        v = self._v_init()
        self.P = self._policy()
        self.R = self._reward(v)

    def _tree_flatten(self):
        children = ()  # arrays / dynamic values
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
        )

        return grid

    def _policy(self):
        """
        Define random policy conv kernel with equal probabilty of taking each action:

        P = Array([[0,     0.25,  0   ],
                   [0.25,  0,     0.25],
                   [0,     0.25,  0   ],]

        """
        policy = jnp.zeros((3, 3))
        policy = policy.at[self.actions[0] + 1, self.actions[1] + 1].set(0.25)

        return policy

    def estimate_state_value(self, iter=1000):
        """"""
        v = self._v_init()
        for _ in range(iter):
            v = self.state_value(
                v,
                self.R,
                self.P,
                self.special_states,
                self.special_states_prime,
                self.special_states_rewards,
            )
        return v

    def _reward(self, v):
        """Provides reward for all states in grid when following a random policy"""
        R = convolve2d(
            jnp.pad(v, pad_width=(1, 1), constant_values=-1),
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
        discount: float = 0.9,
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
            * discount
        )

        # Update edges except for corners
        vp = vp.at[1:-1, 0].add(v[1:-1, 0] * discount * 0.25)
        vp = vp.at[1:-1, -1].add(v[1:-1, -1] * discount * 0.25)
        vp = vp.at[0, 1:-1].add(v[0, 1:-1] * discount * 0.25)
        vp = vp.at[-1, 1:-1].add(v[-1, 1:-1] * discount * 0.25)

        # Update corners
        vp = vp.at[0, 0].add(v[0, 0] * 2 * discount * 0.25)
        vp = vp.at[0, -1].add(v[0, -1] * 2 * discount * 0.25)
        vp = vp.at[-1, -1].add(v[-1, -1] * 2 * discount * 0.25)
        vp = vp.at[-1, 0].add(v[-1, 0] * 2 * discount * 0.25)

        # Update special states
        vp = vp.at[special_states[0], special_states[1]].set(
            v[special_states_prime[0], special_states_prime[1]] * discount
            + special_states_rewards
        )

        return vp


tree_util.register_pytree_node(
    RandomGrid, RandomGrid._tree_flatten, RandomGrid._tree_unflatten
)
