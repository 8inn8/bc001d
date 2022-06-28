import pickle
import random
import warnings
from functools import partial

import chex
import jax
import jax.numpy as jnp
from fire import Fire

from env import Environment
from tree_search import improve_policy_with_mcts, recurrent_fn
from utils import env_step, import_class, replicate, reset_env


@jax.jit
def _apply_temperature(logits, temperature):
    logits = logits - jnp.max(logits, keepdims=True, axis=-1)
    tiny = jnp.finfo(logits.dtype).tiny
    return logits / jnp.maximum(tiny, temperature)


