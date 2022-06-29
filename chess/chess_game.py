from typing import Tuple

import pax
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.nn as jnn
import jax.random as jr
import gym
from gym import Wrapper
from gym_chess import alphazero
import chess
import chex
import random

from env import Environment


class ChessGame(Environment):
    board: Wrapper
    who_play: chex.Array
    terminated: chex.Array
    winner: chex.Array
    count: chex.Array
    winner: chex.Array

    def __int__(self):
        super().__init__()
        self.reset()
        self.board = gym.make('ChessAlphaZero-v0')
        self.board.seed(random.randint(0, 88888))
        self.who_play = jnp.array(1, dtype=jnp.int32)
        self.count = jnp.array(0, dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.winner = jnp.array(0, dtype=jnp.int32)

    def reset(self):
        self.board = gym.make('ChessAlphaZero-v0')
        self.board.seed(random.randint(0, 88888))
        self.who_play = jnp.array(1, dtype=jnp.int32)
        self.count = jnp.array(0, dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.winner = jnp.array(0, dtype=jnp.int32)

    def num_actions(self) -> int:
        return len(self.board.legal_actions)

    def invalid_actions(self) -> chex.Array:
        legal_actions = set(self.board.legal_actions)
        return jnp.array(list(filter(lambda t: t not in legal_actions, range(4672))), dtype=jnp.int32)







    def step(self, action: chex.Array) -> Tuple["ChessGame", chex.Array]:
        possible_actions = self.board.legal_actions

