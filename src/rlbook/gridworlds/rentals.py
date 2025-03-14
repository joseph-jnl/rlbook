import logging

import jax.numpy as jnp
from jax import vmap
from jax._src import config
from jax.scipy.stats import poisson

config.update("jax_platforms", "cuda")  # Disable rocm and tpu init warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Rentals:
    """Base car rental class with jax jit related helper methods.

    Attributes:
        n_rows: number of rows.
        n_cols: number of columns.
        actions: actions that can be taken in the grid.
        v_init: initial state values.
    """

    def __init__(
        self,
        n_rows: int = 3,
        n_cols: int = 3,
        verbose=False,
    ):
        """
        Args:
            n_rows: number of rows.
            n_cols: number of columns.
        """
        assert n_rows == n_cols  # only implement symmetrical car maximums

        if verbose:
            logger.setLevel(logging.DEBUG)

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.len_idx = n_rows * n_cols

        # create a "long" array of row, col indicies of each state
        self.state_indices = jnp.reshape(
            jnp.indices((n_rows, n_cols), dtype=float), (2, self.len_idx)
        )
        logger.debug("\nself.state_indices:\n %s", self.state_indices)

        # Init reward * prob matrix
        # max 5 cars can be moved in one night plus 0 (no action), reuse for both locations
        self.R = jnp.zeros((6, n_rows, n_cols))
        logger.debug("\nself.R init:\n %s", self.R)

        valid_row_rewards, valid_col_rewards = self.reward_broadcast()
        reward_map = vmap(self.reward, in_axes=0)

        for a in range(6):
            self.R = self.R.at[a, :, :].set(
                jnp.sum(
                    reward_map(
                        valid_row_rewards,
                        valid_col_rewards,
                        jnp.ones(valid_row_rewards.shape) * a,
                    ),
                    axis=1,
                ).reshape((self.n_rows, self.n_cols))
            )

        logger.debug(
            "\nreward_map output for action=0:\n %s\nself.R\n %s",
            reward_map(
                valid_row_rewards,
                valid_col_rewards,
                jnp.ones(valid_row_rewards.shape) * 0,
            ),
            self.R,
        )

    def reward_broadcast(self):
        """For each state index, broadcast the set of valid rewards,
            using "-1" padding to mask out invalid rewards and maintain consistent array lengths
        Note: -1 is used since this value had a zero percent chance of occuring in a poisson dist
        """
        valid_row_rewards = jnp.broadcast_to(
            jnp.arange(self.n_rows), (self.len_idx, self.n_rows)
        )
        logger.debug("\nvalid_row_rewards init:\n %s", valid_row_rewards)
        valid_row_rewards = jnp.where(
            valid_row_rewards
            > jnp.broadcast_to(self.state_indices[0], (self.n_rows, self.len_idx)).T,
            -1,
            valid_row_rewards,
        )
        logger.debug("valid_row_rewards:\n %s", valid_row_rewards)

        valid_col_rewards = jnp.broadcast_to(
            jnp.arange(self.n_cols), (self.len_idx, self.n_cols)
        )
        logger.debug("\nvalid_col_rewards init:\n %s", valid_col_rewards)
        valid_col_rewards = jnp.where(
            valid_col_rewards
            > jnp.broadcast_to(self.state_indices[1], (self.n_cols, self.len_idx)).T,
            -1,
            valid_col_rewards,
        )
        logger.debug("valid_col_rewards:\n %s", valid_col_rewards)

        return valid_row_rewards, valid_col_rewards

    def reward(self, row_state_primes, col_state_primes, action):
        p_row = poisson.pmf(row_state_primes, mu=3)
        p_col = poisson.pmf(col_state_primes, mu=4)

        return (
            jnp.arange(p_row.shape[0]) * p_row * 10
            + jnp.arange(p_col.shape[0]) * p_col * 10
            - abs(action * 2)
        )
