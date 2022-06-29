from typing import List

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pax
import treeo as to
import chex


class ChessEngine(to.Tree):
    current_board: chex.Array = to.field(node=True)
    who_to_move: chex.Array = to.field(node=True)
    actions_sequence: List[int] = to.field(node=True)

    def __init__(self):
        self.who_to_move = jnp.array(1, dtype=jnp.int32)
        self.actions_sequence = []

    def get_board_state(self):
        return self.current_board


class ChessValues(to.Tree):
    penc: dict = to.field(node=True)
    penc_r: dict = to.field(node=True)
    initial_board: chex.Array = to.field(node=True)

    def __init__(self):
        self.penc = {" ": 0, 'r': 1, 'n': 2, 'b': 3, 'q': 4, 'k': 5, 'p': 6,  'R': 7, 'N': 8, 'B': 9, 'Q': 10, 'K': 11, 'P': 12}
        self.penc_r = {0: " ", 1: "r", 2: "n", 3: "b", 4: "q", 5: "k", 6: "p", 7: "R", 8: "N", 9: "B", 10: "Q", 11: "K", 12: "P"}
        self.initial_board = jnp.zeros(shape=(8, 8), dtype=jnp.int32)
        self.initial_board = self.initial_board.at[0, :] = jnp.array([1, 2, 3, 5, 4, 3, 2, 1], dtype=jnp.int32)
        self.initial_board = self.initial_board.at[1, :] = 6 * np.ones(shape=(8,), dtype=jnp.int32)
        self.initial_board = self.initial_board.at[6, :] = 12 * np.ones(shape=(8,), dtype=jnp.int32)
        self.initial_board = self.initial_board.at[7, :] = jnp.array([7, 8, 9, 11, 10, 3, 9, 7], dtype=jnp.int32)

    def get_encoding(self):
        return self.penc

    def get_encoding_reversed(self):
        return self.penc_r