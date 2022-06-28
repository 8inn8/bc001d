import os
import pickle
import random
from functools import partial
from typing import Deque

import chex
import click
import fire

import jax
import jax.random as jr
import jax.numpy as jnp
import jax.nn as jnn

import numpy as np

import opax
import optax
import pax

from env import Environment
from play import agent_vs_agent_multiple_games
from tree_search import improve_policy_with_mcts, recurrent_fn
from utils import batched_policy, env_step, import_class, replicate, reset_env

EPSILON = 1e-9


@chex.dataclass(frozen=True)
class TrainingExample:
    state: chex.Array
    action_weights: chex.Array
    value: chex.Array


@chex.dataclass(frozen=True)
class MoveOutput:
    state: chex.Array
    reward: chex.Array
    terminated: chex.Array
    action_weights: chex.Array


@partial(jax.pmap, in_axes=(None, None, 0), static_broadcasted_argnums=(3, 4, 5))
def collect_batched_self_play_data(agent, env: Environment, rng_key: chex.Array, batch_size: int, num_simulations_per_move: int, temperature_decay: float):

    def single_move(prev, inputs):
        env, rng_key, step = prev
        del inputs
        rng_key, rng_key_next = jax.random.split(rng_key)
        state = env.canonical_observation()
        terminated = env.is_terminated()
        temperature = jnp.power(temperature_decay, step)
        policy_output = improve_policy_with_mcts(agent, env, rng_key, recurrent_fn, num_simulations_per_move, temperature=temperature)
        env, reward = jax.vmap(env_step)(env, policy_output.action)
        return (env, rng_key_next, step + 1), MoveOutput(state=state, reward=reward, terminated=terminated, action_weights=policy_output.action_weights)

    env = reset_env(env)
    env = replicate(env, batch_size)
    step = jnp.array(1)

    return pax.scan(single_move, (env, rng_key, step), None, length=env.max_num_steps(), time_major=False)[1] # self_play_data


def collect_self_play_data(agent, env: Environment, rng_key: chex.Array, batch_size: int, data_size: int, num_simulations_per_move: int, temperature_decay: float):
    N = data_size // batch_size
    devices = jax.local_devices()
    num_devices = len(devices)
    rng_keys = jr.split(rng_key, N * num_devices)
    rng_keys = jnp.stack(rng_keys).reshape((N, num_devices, -1))
    data = []

    with click.progressbar(range(N), label=" self play ") as bar:
        for i in bar:
            batch = collect_batched_self_play_data(agent, env, rng_keys[i], batch_size // num_devices, num_simulations_per_move, temperature_decay)
            batch = jax.device_get(batch)
            batch = jax.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), batch)
            data.append(batch)
    data = jax.tree_map(lambda *xs: np.concatenate(xs), *data)
    return data


def prepare_training_data(data: MoveOutput):
    buffer = []
    N = len(data.terminated)
    for i in range(N):
        state = data.state[i]
        is_terminated = data.terminated[i]
        action_weights = data.action_weights[i]
        reward = data.reward[i]
        L = len(is_terminated)
        value = None
        for idx in reversed(range(L)):
            if is_terminated[idx]:
                continue
            value = reward[idx] if value is None else -value
            buffer.append(TrainingExample(state=state[idx], action_weights=action_weights[idx], value=np.array(value, dtype=np.float32)))

    return buffer


@jax.jit
@partial(jax.value_and_grad, has_aux=True)
def loss_fn(net, data: TrainingExample):
    net, (action_logits, value) = batched_policy(net, data.space)

    mse_loss = optax.l2_loss(value, data.value)
    mse_loss = jnp.mean(mse_loss)

    target_pr = data.action_weights

    target_pr = jnp.where(target_pr == 0, EPSILON, target_pr)
    action_logits = jnn.log_softmax(action_logits, axis=-1)
    kl_loss = jnp.sum(target_pr * (jnp.log(target_pr) - action_logits), axis=-1)
    kl_loss = jnp.mean(kl_loss)

    return mse_loss + kl_loss, (net, (mse_loss, kl_loss))


@jax.jit
def train_step(net, optim, data: TrainingExample):
    (_, (net, losses)), grads = loss_fn(net, data)
    net, optim = opax.apply_gradients(net, optim, grads)
    return net, optim, losses



