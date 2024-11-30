from abc import ABCMeta, abstractmethod

import jax.numpy as jnp


class Grid(metaclass=ABCMeta):
    """ """

    def __init__(self, n_rows: int = 5, n_cols: int = 5, iter: int = 1000):
        self.v = jnp.zeros(n_rows, n_cols)
        self.n_rows = n_rows
        self.n_cols = n_cols

    @abstractmethod
    def actions(self):
        """ """

    @abstractmethod
    def reward(self):
        """Reward for a given action"""


class SimpleGrid(Grid):
    """"""

    def __init__(
        self,
        special_states: dict[tuple[int, int], int],
        special_states_prime: list[tuple[int, int]],
    ):
        self.special_states = special_states
        self.special_states_prime = special_states_prime

    def reward(self, s: tuple[int, int], sp: tuple[int, int]):
        """Reward for a given action
        Args:
            s: Tuple containing row and column indices of current state
            sp: Tuple containing row and column indices of next state moved to by the taken action

        """
        condlist = [
            # Out of bounds
            (sp[0] == -1) | (sp[0] > (self.n_rows - 1)),
            (sp[1] == -1) | (sp[1] > (self.n_cols - 1)),
            # Exit special state
            s in self.special_states,
        ]

        choicelist = [
            # Out of bounds
            -1,
            -1,
            # Exit special state
            self.special_states[s] if s in self.special_states else 0,
        ]

        return jnp.select(condlist, choicelist, default=0)  # Else return 0 as default
