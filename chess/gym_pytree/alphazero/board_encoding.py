# -*- coding: utf-8 -*-

import numpy as np
import chess

import gym
import pax
from gym import spaces
from gym_chess import Chess

import jax.numpy as jnp

from jax import tree_util


class BoardEncoding(gym.ObservationWrapper):
    """Implements the board encoding from AlphaZero.
    
    This wrapper converts observations (i.e. `chess.Board` instances) from the 
    wrapped `Chess` environment to numpy arrays, using the encoding proposed in 
    [Silver et al., 2017].
    
    An observation is represented as an array of shape (8, 8, k * 14 + 7). This
    can be thought of as a stack of "planes" of size 8 x 8, where each of the 
    k * 14 + 7 planes represents a different aspect of the game.   

    The first k * 14 planes encode the k most recent board positions, here 
    refered to as the board's "history", grouped into sets of 14 planes per
    position. The first six planes in a set of 14 denote the pieces of the 
    active player (i.e. the player for which the agent will move next). Each of 
    the six planes is associated with a particular piece type  (pawn, knight, 
    bishop, queen and king, respectively). The subsequent six planes encode the 
    pieces of the opponent player, following the same scheme. The final two 
    planes are binary (i.e. either all ones or zeros), and indicate two-fold 
    and three-fold repetition of the encoded position over the course of the 
    current game. Note that in each step, all board representations in the 
    history are reoriented to the perspective of the active player.

    The remaining 7 planes encode meta-information about the game state: the 
    color of the active player (0 = Black, 1 = White), the total move count, 
    castling rights for the active player and her opponent (kingside and 
    queenside, respectively), and lastly, the halfmove-clock.

    Args:
        env: The wrapped env.
        history_length: The number of recent board positions encoded in an
            observation (corresponds to the 'k' parameter above).

    Observations:
        Box(8, 8, k * 14 + 7)

        Note: Not all possible instances of this observation space encode legal 
        board positions. It is thus not advised to use 
        env.observation_space.sample() to generate board positions.
    """
    # TODO how to type env correctly?
    def __init__(self, env, history_length: int = 80) -> None:
        super(BoardEncoding, self).__init__(env)

        self._history = BoardHistory(history_length)
        
        self.observation_space = spaces.Box(
            low=0,
            high=np.iinfo(np.int).max,
            shape=(8, 8, history_length * 14 + 7),
            dtype=np.int
        )


    def reset(self, **kwargs) -> np.array:
        """Resets the environment and clears board history."""

        self._history.reset()
        return super(BoardEncoding, self).reset(**kwargs)


    def observation(self, board: chess.Board) -> np.array:
        """Converts chess.Board observations instance to numpy arrays.
        
        Note: 
            This method is called by gym.ObservationWrapper to transform the 
            observations returned by the wrapped env's step() method. In 
            particular, calling this method will add the given board position
            to the history. Do NOT call this method manually to recover a 
            previous observation returned by step().
        """

        self._history.push(board)

        history = self._history.view(orientation=board.turn)
        
        meta = jnp.zeros(
            shape=(8, 8, 7),
            dtype=jnp.int32
        )
    
        # Active player color
        meta = meta.at[:, :, 0].set(jnp.int32(board.turn))
    
        # Total move count
        meta = meta.at[:, :, 1].set(board.fullmove_number)

        # Active player castling rights
        meta = meta.at[:, :, 2].set(board.has_kingside_castling_rights(board.turn))
        meta = meta.at[:, :, 3].set(board.has_queenside_castling_rights(board.turn))
    
        # Opponent player castling rights
        meta = meta.at[:, :, 4].set(board.has_kingside_castling_rights(not board.turn))
        meta = meta.at[:, :, 5].set(board.has_queenside_castling_rights(not board.turn))
    
        # No-progress counter
        meta = meta.at[:, :, 6].set(board.halfmove_clock)

        observation = jnp.concatenate([history, meta], axis=-1)
        return observation

    def tree_flatten(self):
        children = (self.env,)
        aux_data = {'_history': self._history}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        x = cls(children[0])
        x._history = aux_data['_history']
        return x

class BoardHistory(pax.Module):
    """Maintains a history of recent board positions, encoded as numpy arrays.

    New positions are added to the history via the push() method, which will
    store the position as a numpy array (using the encoding described in
    [Silver et al., 2017]). The history only retains the k most recent board
    positions; older positions are discarded when new ones are added. An array
    view of the history can be obtained via the view() function.
    
    Args:
        length: The number of most recent board positions to retain (corresponds
        to the 'k' parameter above).
    """

    def __init__(self, length: int) -> None:
        super(BoardHistory, self).__init__()
        #: Ring buffer of recent board encodings; stored boards are always
        #: oriented towards the White player. 
        self._buffer = jnp.zeros((length, 8, 8, 14), dtype=np.int)

    @pax.pure
    def push(self, board: chess.Board) -> None:
        """Adds a new board to the history."""

        board_array = self.encode(board)

        # Overwrite oldest element in the buffer.
        self._buffer = self._buffer.at[-1].set(board_array)
        
        # Roll inserted element to the top (= most recent position); all older
        # elements are pushed towards the end of the buffer
        self._buffer = jnp.roll(self._buffer, 1, axis=0)

    @pax.pure
    def encode(self, board: chess.Board) -> np.array:
        """Converts a board to numpy array representation."""

        array = jnp.zeros((8, 8, 14), dtype=np.int)

        for square, piece in board.piece_map().items():
            rank, file = chess.square_rank(square), chess.square_file(square)
            piece_type, color = piece.piece_type, piece.color
        
            # The first six planes encode the pieces of the active player, 
            # the following six those of the active player's opponent. Since
            # this class always stores boards oriented towards the white player,
            # White is considered to be the active player here.
            offset = 0 if color == chess.WHITE else 6
            
            # Chess enumerates piece types beginning with one, which we have
            # to account for
            idx = piece_type - 1
        
            array = array.at[rank, file, idx + offset].set(1)

        # Repetition counters
        array = array.at[:, :, 12].set(board.is_repetition(2))
        array = array.at[:, :, 13].set(board.is_repetition(3))

        return array

    @pax.pure
    def view(self, orientation: bool = chess.WHITE) -> np.array:
        """Returns an array view of the board history.

        This method returns a (8, 8, k * 14) array view of the k most recently
        added positions. If less than k positions have been added since the 
        last reset (or since the class was instantiated), missing positions are
        zeroed out. 
        
        By default, positions are oriented towards the white player; setting the
        optional orientation parameter to 'chess.BLACK' will reorient the view 
        towards the black player.
        
        Args:
            orientation: The player from which perspective the positions should
            be encoded.
        """

        # Copy buffer to not let reorientation affect the internal buffer
        array = self._buffer.copy()
        
        if orientation == chess.BLACK:
            for board_array in array:

                # Rotate all planes encoding the position by 180 degrees
                rotated = jnp.rot90(board_array[:, :, :12], k=2)

                # In the buffer, the first six planes encode white's pieces; 
                # swap with the second six planes
                rotated = jnp.roll(rotated, axis=-1, shift=6)

                board_array = board_array.at[:, :, :12].set(rotated)

        # Concatenate k stacks of 14 planes to one stack of k * 14 planes
        array = jnp.concatenate(array, axis=-1)
        return array

    @pax.pure
    def reset(self) -> None:
        """Clears the history."""
        self._buffer = self._buffer.at[:].set(0)


tree_util.register_pytree_node(BoardEncoding, BoardEncoding.tree_flatten, BoardEncoding.tree_unflatten)