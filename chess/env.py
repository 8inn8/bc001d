from typing import Any, Tuple, TypeVar
import chex
import pax

E = TypeVar("E")


class Environment(pax.Module):
    def __init__(self):
        super().__init__()

    def step(self: E, action: chex.Array) -> Tuple[E, chex.Array]:
        raise NotImplementedError()

    def reset(self):
        pass

    def is_terminated(self) -> chex.Array:
        raise NotImplementedError()

    def observation(self) -> Any:
        pass

    def canonical_observation(self) -> Any:
        pass

    def num_actions(self) -> int:
        raise NotImplementedError()

    def invalid_actions(self) -> chex.Array:
        raise NotImplementedError()

    def max_num_steps(self) -> int:
        raise NotImplementedError()

    def render(self) -> Any:
        pass

