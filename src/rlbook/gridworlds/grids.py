from abc import ABCMeta, abstractmethod

import jax.numpy as jnp
from jax import jit
from jax.scipy.signal import correlate2d
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float, Int


class Grid(metaclass=ABCMeta):
    """Base grid class with jax jit related helper methods.

    Attributes:
        n_rows: number of rows.
        n_cols: number of columns.
        actions: actions that can be taken in the grid.
        v_init: initial state values.
    """

    def __init__(
        self,
        n_rows: int = 5,
        n_cols: int = 5,
    ):
        """
        Args:
            n_rows: number of rows.
            n_cols: number of columns.
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.actions = jnp.array([[-1, 1, 0, 0], [0, 0, 1, -1]])
        self.v_init = jnp.zeros((self.n_rows, self.n_cols))

    @property
    @abstractmethod
    def policy(self): ...

    @abstractmethod
    def reward(self): ...

    def tree_flatten(self):
        """Jax flatten method for serialization, required to jit class methods."""
        children = (
            self.special_states_rewards,
            self.R,
            self.P,
            self.actions,
            self.v_init,
        )  # arrays and dynamic values
        # static values (non-arrays)
        aux_data = {
            "special_states": self.special_states,
            "special_states_prime": self.special_states_prime,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
        }

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Jax unflatten method for deserialization, required to jit class methods."""
        grid = cls(
            aux_data["special_states"],
            aux_data["special_states_prime"],
            children[0],
            R=children[1],
            P=children[2],
            n_rows=aux_data["n_rows"],
            n_cols=aux_data["n_cols"],
        )
        grid.v_init = children[2]

        return grid


@register_pytree_node_class
class RandomGrid(Grid):
    """RandomGrid class for estimating state values in a gridworld using a random policy."""

    def __init__(
        self,
        special_states: list[list[int, int]],
        special_states_prime: list[list[int, int]],
        special_states_rewards: Int[Array, "1 {len(special_states)}"],
        n_rows: int = 5,
        n_cols: int = 5,
        R: Float[Array, "n_rows n_cols"] = None,
        P: Float[Array, "3 3"] = None,
    ):
        """
        Args:
            special_states: list containing special states row and columns.
              e.g. [[0, 0], [1, 3]] would correspond to special state A located at row 0 and column 1
              and special state B located at row 1 and column 3.
            special_states_prime: list of special states prime rows and columns, see previous special_states example.
            special_states_rewards: jax array of rewards for special states. Note: not a list!
            n_rows: number of rows.
            n_cols: number of columns.
            R: jax array specifying rewards for all states when taking a random policy.
            P: jax array specifying a conv kernel for a random policy.
        """
        super().__init__(n_rows=n_rows, n_cols=n_cols)
        self.special_states = special_states
        self.special_states_prime = special_states_prime
        self.special_states_rewards = special_states_rewards

        self.v_init = jnp.zeros((self.n_rows, self.n_cols))
        self.P = self.policy
        self.R = self.reward

    @property
    def policy(self) -> Float[Array, "3 3"]:
        """
        Define random policy conv kernel with equal probabilty of taking each action:

        P = Array([[0,     0.25,  0   ],
                   [0.25,  0,     0.25],
                   [0,     0.25,  0   ],]
        """
        policy = jnp.zeros((3, 3))
        policy = policy.at[self.actions[0] + 1, self.actions[1] + 1].set(0.25)

        return policy

    @property
    def reward(self) -> Float[Array, "{self.n_rows} {self.n_cols}"]:
        """Provides reward for all states in grid when following a random policy"""
        R = correlate2d(
            jnp.pad(self.v_init, pad_width=(1, 1), constant_values=-1),
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
        v: Float[Array, "n_rows n_cols"],
        R: Float[Array, "n_rows n_cols"],
        P: Float[Array, "3 3"],
        special_states: list[list[int, int]],
        special_states_prime: list[list[int, int]],
        special_states_rewards: Float[Array, "1 {len(special_states)}"],
        discount: float = 0.9,
    ) -> Float[Array, "{self.n_rows} {self.n_cols}"]:
        """State value function for estimating state values in a gridworld using a random policy"""
        # Update states
        vp = (
            R
            + correlate2d(
                jnp.pad(v, pad_width=(1, 1), mode="edge"),
                P,
                mode="valid",
            )
            * discount
        )

        # Update special states
        vp = vp.at[special_states[0], special_states[1]].set(
            v[special_states_prime[0], special_states_prime[1]] * discount
            + special_states_rewards
        )

        return vp

    def estimate_state_value(
        self, iter: int = 1000
    ) -> Float[Array, "{self.n_rows} {self.n_cols}"]:
        """Estimate state values in a gridworld using a random policy"""
        v = self.v_init
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


@register_pytree_node_class
class OptimalGrid(Grid):
    """OptimalGrid class for estimating state values in a gridworld using an optimal policy"""

    def __init__(
        self,
        special_states: list[list[int], list[int]],
        special_states_prime: list[list[int], list[int]],
        special_states_rewards: Int[Array, "1 {len(special_states)}"],
        n_rows: int = 5,
        n_cols: int = 5,
        R: Float[Array, "n_rows n_cols"] = None,
        P: Float[Array, "3 3"] = None,
    ):
        """
        Args:
            special_states: list containing special states row and columns.
              e.g. [[0, 0], [1, 3]] would correspond to special state A located at row 0 and column 1
              and special state B located at row 1 and column 3.
            special_states_prime: list of special states prime rows and columns, see previous special_states example.
            special_states_rewards: jax array of rewards for special states. Note: not a list!
            n_rows: number of rows.
            n_cols: number of columns.
            R: jax array specifying rewards for all states when taking an optimal policy.
            P: jax array specifying a conv kernel for an optimal policy.
        """
        super().__init__(n_rows=n_rows, n_cols=n_cols)
        self.special_states = special_states
        self.special_states_prime = special_states_prime
        self.special_states_rewards = special_states_rewards

        self.P = self.policy
        self.R = self.reward(self.v_init, self.policy)

    @property
    def policy(self) -> Float[Array, "4 3 3"]:
        """
        Define policy conv kernel as a 3d array

        P = Array([[[0., 1., 0.], # action up only
                    [0., 0., 0.],
                    [0., 0., 0.]],

                    [[0., 0., 0.], # action left only
                    [1., 0., 0.],
                    [0., 0., 0.]],

                    [[0., 0., 0.], #action down only
                    [0., 0., 0.],
                    [0., 1., 0.]],

                    [[0., 0., 0.], @action right only
                    [0., 0., 1.],
                    [0., 0., 0.]]]

        """
        policy = jnp.zeros((4, 3, 3))
        policy = policy.at[[0, 1, 2, 3], [0, 1, 2, 1], [1, 0, 1, 2]].set(1)

        return policy

    def reward(
        self, v: Float[Array, "{self.n_rows} {self.n_cols}"], P: Float[Array, "4 3 3"]
    ) -> Float[Array, "{self.n_rows} {self.n_cols}"]:
        """Provides reward for all states in grid when following an optimal policy"""
        R = jnp.zeros((4, self.n_rows, self.n_cols))

        # Policy up only
        R = R.at[0, :, :].set(
            correlate2d(
                jnp.pad(v, pad_width=(1, 1), constant_values=-1),
                P[0, :, :],
                mode="valid",
            )
        )

        # Policy left only
        R = R.at[1, :, :].set(
            correlate2d(
                jnp.pad(v, pad_width=(1, 1), constant_values=-1),
                P[1, :, :],
                mode="valid",
            )
        )

        # Policy down only
        R = R.at[2, :, :].set(
            correlate2d(
                jnp.pad(v, pad_width=(1, 1), constant_values=-1),
                P[2, :, :],
                mode="valid",
            )
        )

        # Policy right only
        R = R.at[3, :, :].set(
            correlate2d(
                jnp.pad(v, pad_width=(1, 1), constant_values=-1),
                P[3, :, :],
                mode="valid",
            )
        )

        # Set special state rewards
        R = R.at[:, self.special_states[0], self.special_states[1]].set(
            self.special_states_rewards
        )

        return R

    @jit
    def state_value(
        self,
        v: Float[Array, "n_rows n_cols"],
        R: Float[Array, "4 n_rows n_cols"],
        P: Float[Array, "4 3 3"],
        special_states: list[list[int], list[int]],
        special_states_prime: list[list[int], list[int]],
        special_states_rewards: Float[Array, "1 {len(special_states)}"],
        discount: float = 0.9,
    ) -> Float[Array, "{self.n_rows} {self.n_cols}"]:
        """State value function for estimating state values in a gridworld using an optimal policy"""

        vp = jnp.zeros((4, self.n_rows, self.n_cols))

        # Policy up only
        vp = vp.at[0, :, :].set(
            R[0, :, :]
            + correlate2d(
                jnp.pad(v, pad_width=(1, 1), mode="edge"),
                P[0, :, :],
                mode="valid",
            )
            * discount
        )

        # Policy left only
        vp = vp.at[1, :, :].set(
            R[1, :, :]
            + correlate2d(
                jnp.pad(v, pad_width=(1, 1), mode="edge"),
                P[1, :, :],
                mode="valid",
            )
            * discount
        )

        # Policy down only
        vp = vp.at[2, :, :].set(
            R[2, :, :]
            + correlate2d(
                jnp.pad(v, pad_width=(1, 1), mode="edge"),
                P[2, :, :],
                mode="valid",
            )
            * discount
        )

        # Policy right only
        vp = vp.at[3, :, :].set(
            R[3, :, :]
            + correlate2d(
                jnp.pad(v, pad_width=(1, 1), mode="edge"),
                P[3, :, :],
                mode="valid",
            )
            * discount
        )

        # Update special states
        vp = vp.at[:, special_states[0], special_states[1]].set(
            v[special_states_prime[0], special_states_prime[1]] * discount
            + special_states_rewards
        )

        return jnp.max(vp, axis=0)

    def estimate_state_value(
        self, iter: int = 1000
    ) -> Float[Array, "{self.n_rows} {self.n_cols}"]:
        """Estimate state values in a gridworld using an optimal policy"""
        v = self.v_init
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
