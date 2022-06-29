from typing import Tuple, List, Optional, Dict

import einops
import numpy as np
import pax
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.nn as jnn
import jax.random as jr
import gym
from gym import Wrapper, spaces
import chess
import chex
import functools as ft

import treeo as to

from gym_pytree.alphazero.board_encoding import BoardEncoding
from gym_pytree.envs import Chess
from gym_pytree.alphazero.move_encoding.move_encoding import MoveEncoding
from env import Environment


class ChessMain(to.Tree):
    b: MoveEncoding = to.field(node=False)
    who_play: chex.Array = to.field(node=True)
    count: chex.Array = to.field(node=True)
    terminated: chex.Array = to.field(node=True)

    def __init__(self, b=MoveEncoding(BoardEncoding(Chess(), history_length=80))):
        super().__init__()
        self.b = b
        self.who_play = jnp.array(1, dtype=jnp.int32)
        self.count = jnp.array(0, dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.reset()

    def reset(self):
        chessenv = Chess()
        chessazenv = BoardEncoding(chessenv, history_length=80)
        self.b = MoveEncoding(chessazenv)
        self.who_play = jnp.array(1, dtype=jnp.int32)
        self.count = jnp.array(0, dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.b.reset()

    def num_actions(self) -> int:
        return 4672

    def _check_illegal(self, legal_actions):
        b = jnp.arange(4672)

        d = jnp.ones_like(b)
        d = d.at[legal_actions].set(0)

        return b * d

    def invalid_actions(self) -> chex.Array:
        return self._check_illegal(self.b.legal_actions) * self.terminated[..., None]

    def observation(self) -> chex.Array:
        cb = jnp.argmax(self.b.observation(self.b.observation_p), axis=2)
        cb = jnp.where(cb < 14, cb, 14)
        return cb

    def canonical_observation(self):
        return self.observation() * self.who_play[..., None, None]

    def is_terminated(self) -> chex.Array:
        return self.terminated

    def max_num_steps(self) -> int:
        return 144

    def render(self):
        print(self.b.render())

    def _compute_to_vectors(self, action, n=4672):
        legal_actions = self.b.legal_actions
        print("Legal ", legal_actions)
        a = jnp.ones(shape=action.shape, dtype=jnp.int32)
        a = a.at[:, legal_actions].set(0)
        b = jnp.zeros(shape=action.shape, dtype=jnp.int32)
        b = b.at[action].set(1)
        invalid_move = a * b

        print("Actions ", action)

        print("Invalid move: ", invalid_move)

        result = jnp.max(invalid_move, axis=1)

        print("Result!!! ", result)

        return result

    @pax.pure
    def step(self, action: chex.Array) -> Tuple["ChessMain", chex.Array]:
        invalid_move = self._compute_to_vectors(action)
        (obsb, reward, done, _) = self.b.step(np.array(action))
        reward = 0.5 * self.who_play if \
            (obsb.can_claim_threefold_repetition() or obsb.can_claim_fifty_moves() or obsb.can_claim_draw() or obsb.is_fivefold_repetition() or obsb.is_seventyfive_moves()) \
            else 1.0 * self.who_play if (abs(reward) == 1) else 0.0

        self.who_play = -self.who_play

        self.terminated = 1 * jnp.ones_like(action, dtype=bool) if \
            jnp.logical_or(self.terminated, jnp.abs(reward) == 0.5, jnp.abs(reward) == 1.0) else 0 * jnp.ones_like(action, dtype=bool)

        reward = jnp.ones_like(action) * reward

        return self, reward


