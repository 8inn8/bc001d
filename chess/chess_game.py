from typing import Tuple, List, Optional, Dict

import numpy as np
import pax
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.nn as jnn
import jax.random as jr
import gym
from gym import Wrapper, spaces
from gym_chess import alphazero, Chess, BoardEncoding, MoveEncoding
import chess
import chex

from env import Environment


class ChessBase(gym.Env):
    """Base environment for the game of chess.
    This env does not have a built-in opponent; moves are made for both the
    black and the white player in turn. At any given timestep, the env expects a
    legal move for the current player, otherwise an Error is raised.
    The agent is awarded a reward of +1 if the white player makes a winning move
    and -1 if the black player makes a winning move. All other rewards are zero.
    Since the winning player is always the last one to move, this is the only
    way to assign meaningful rewards based on the outcome of the game.
    Observations and actions are represented as `Board` and `Move` objects,
    respectively. The actual encoding as numpy arrays is left to wrapper classes
    for flexibility and separation of concerns (see the `wrappers` module for
    examples). As a consequence, the `observation_space` and `action_space`
    members are set to `None`.
    Observation:
        Type: chess.Board
        Note: Modifying the returned `Board` instance does not modify the
        internal state of this env.
    Actions:
        Type: chess.Move
    Reward:
        +1/-1 if white or black makes a winning move, respectively.
    Starting State:
        The usual initial board position for chess, as defined by FIDE
    Episode Termination:
        Either player wins.
        The game ends in a draw (e.g. stalemate, insufficient matieral,
        fifty-move rule, threefold repetition)
        Note: Surrendering is not an option.
    """

    # We deliberately use the render mode 'unicode' instead of the canonical
    # 'ansi' mode, since the output string contains non-ascii characters.
    meta = {
        'render.modes': ['unicode']
    }

    action_space = None
    observation_space = None

    reward_range = (-1, 1)

    """Maps game outcomes returned by `chess.Board.result()` to rewards."""
    _rewards: Dict[str, float] = {
        '*': 0.0,  # Game not over yet
        '1/2-1/2': 0.0,  # Draw
        '1-0': +1.0,  # White wins
        '0-1': -1.0,  # Black wins
    }

    def __init__(self, board: chess.Board = None) -> None:
        #: The underlying chess.Board instance that represents the game.
        self._board: Optional[chess.Board] = board

        #: Indicates whether the env has been reset since it has been created
        #: or the previous game has ended.
        self._ready: bool = False

    def reset(self) -> chess.Board:

        self._board = chess.Board()
        self._ready = True

        return self.observation_p

    def step(self, action: chess.Move) -> Tuple[chess.Board, float, bool, None]:

        assert self._ready, "Cannot call env.step() before calling reset()"

        if action not in self._board.legal_moves:
            raise ValueError(
                f"Illegal move {action} for board position {self._board.fen()}"
            )

        self._board.push(action)

        observation = self.observation_p
        reward = self._reward()
        done = self._board.is_game_over()

        if done:
            self._ready = False

        return observation, reward, done, None

    def render(self, mode: str = 'unicode') -> Optional[str]:
        """
        Renders the current board position.
        The following render modes are supported:
        - unicode: Returns a string (str) representation of the current
          position, using non-ascii characters to represent individual pieces.
        Args:
            mode: (see above)
        """

        board = self._board if self._board else chess.Board()

        if mode == 'unicode':
            return board.unicode()

        else:
            return super(ChessBase, self).render(mode=mode)

    @property
    def legal_moves(self) -> List[chess.Move]:
        """Legal moves for the current player."""
        assert self._ready, "Cannot compute legal moves before calling reset()"

        return list(self._board.legal_moves)

    @property
    def observation_p(self) -> chess.Board:
        """Returns the current board position."""
        return self._board.copy()

    def _reward(self) -> float:
        """Returns the reward for the most recent move."""
        result = self._board.result()
        reward = Chess._rewards[result]

        return reward

    def _repr_svg_(self) -> str:
        """Returns an SVG representation of the current board position"""
        board = self._board if self._board else chess.Board()
        return str(board._repr_svg_())


class ChessMaster(Environment):
    b: MoveEncoding
    who_play: chex.Array
    count: chex.Array
    terminated: chex.Array

    def __init__(self, b=MoveEncoding(BoardEncoding(ChessBase(), history_length=80))):
        super().__init__()
        self.b = b
        self.who_play = jnp.array(1, dtype=jnp.int32)
        self.count = jnp.array(0, dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.reset()

    def reset(self):
        chessenv = ChessBase()
        chessazenv = BoardEncoding(chessenv, history_length=80)
        self.b = MoveEncoding(chessazenv)
        self.who_play = jnp.array(1, dtype=jnp.int32)
        self.count = jnp.array(0, dtype=jnp.int32)
        self.terminated = jnp.array(0, dtype=jnp.bool_)
        self.winner = jnp.array(0, dtype=jnp.int32)

    def num_actions(self) -> int:
        return 4672

    def invalid_actions(self) -> chex.Array:
        legal_actions = set(self.b.legal_actions)
        f = lambda t: 1 if t not in legal_actions else 0
        return jnp.array(list(map(f, range(4672))), dtype=bool)

    def observation(self) -> chex.Array:
        b = self.b.observation_p
        return jnp.array(b)

    def canonical_observation(self):
        return self.observation() * self.who_play[..., None, None]

    def is_terminated(self) -> chex.Array:
        return self.terminated

    def max_num_steps(self) -> int:
        return 100000

    def render(self):
        print(self.b.render())

    @pax.pure
    def step(self, action: chex.Array) -> Tuple["ChessMaster", chex.Array]:
        legal_actions = set(self.b.legal_actions)
        f = lambda t: 1 if t not in legal_actions else 0
        invalid_move = jnp.array(list(map(f, action)), dtype=bool)
        if jnp.sum(invalid_move) > 0:
            rewards = -100.0 * jnp.ones_like(action)
            return self, rewards

        (obsb, reward, done, _) = self.b.step(np.array(action))
        reward = 0.5 * self.who_play if \
            (obsb.can_claim_threefold_repetition() or obsb.can_claim_fifty_moves() or obsb.can_claim_draw() or obsb.is_fivefold_repetition() or obsb.is_seventyfive_moves()) \
            else 1.0 * self.who_play if (abs(reward) == 1) else 0.0

        self.who_play = -self.who_play

        self.terminated = True * jnp.ones_like(action, dtype=bool) if \
            jnp.logical_or(self.terminated, jnp.abs(reward) == 0.5, jnp.abs(reward) == 1.0) else False * jnp.ones_like(action, dtype=bool)

        reward = jnp.ones_like(action) * reward

        return self, reward
