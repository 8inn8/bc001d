import functools as ft
import importlib
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import pax

from env import Environment as E


@pax.pure
def batched_policy(agent, states):
    return agent, agent(states, batched=True)


@ft.partial(jax.jit, static_argnums=1)
def replicate(value: chex.ArrayTree, repeat: int) -> chex.ArrayTree:
    return jax.tree_map(lambda x: jnp.stack([x] * repeat), value)


@pax.pure
def reset_env(env: E) -> E:
    env.reset()
    return env


def env_step(env: E, action: chex.Array) -> Tuple[E, chex.Array]:
    env, reward = env.step(action)
    return env, reward


def import_class(path: str) -> E:
    names = path.split(".")
    mod_path, class_name = names[:-1], names[-1]
    mod = importlib.import_module(".".join(mod_path))
    return getattr(mod, class_name)


