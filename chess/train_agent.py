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
@partial(jax.jit, static_argnums=(3, 4, 5))
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
    net, (action_logits, value) = batched_policy(net, data.state)

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


def train(game_class="chess_game.ChessGame", agent_class="ResNetPolicy.ResnetPolicyValueNet", batch_size: int = 128,
          num_iterations: int = 256, num_simulations_per_move: int = 2048, num_self_plays_per_interaction: int = 4096, num_sim_games: int = 512,
          learning_rate: float = 0.001, ckpt_filename: str = "./data/agent.ckpt", random_seed: int = 88, weight_decay: float = 1e-4,
          temperature_decay=0.9, buffer_size: int = 66_666, rng_key=None):
    env = import_class(game_class)()
    agent = import_class(agent_class)(input_dims=env.observation_p().shape, num_actions=env.num_actions())
    optim = opax.adamw(learning_rate, weight_decay=weight_decay).init(agent.parameters())
    if os.path.isfile(ckpt_filename):
        print("Loading weights at ", ckpt_filename)
        with open(ckpt_filename, "rb") as f:
            d = pickle.load(f)
            agent = agent.load_state_dict(d["agent"])
            optim = optim.load_state_dict(d["optim"])
            start_iter = d["iter"] + 1
            # Todo migrate optimizer to Radam or lookahead + radam
    else:
        start_iter = 0
    rng_key = jax.random.PRNGKey(random_seed) if rng_key is None else rng_key
    shuffler = random.Random(random_seed)
    # noinspection PyTypeHints
    buffer = Deque(maxlen=buffer_size)

    for iteration in range(start_iter, num_iterations):
        print(f"Iteration {iteration}")
        rng_key_1, rng_key_2, rng_key_3, rng_key = jax.random.split(rng_key, 4)
        agent = agent.eval()
        data = collect_self_play_data(agent, env, rng_key_1, batch_size, num_self_plays_per_interaction, num_simulations_per_move, temperature_decay)
        data = prepare_training_data(data)
        buffer.extend(data)
        data = list(buffer)
        shuffler.shuffle(data)
        N = len(data)
        losses = []
        old_agent = jax.tree_map(lambda x: jnp.copy(x), agent)
        agent = agent.train()
        with click.progressbar(range(0, N - batch_size, batch_size), label="train agent") as bar:
            for i in bar:
                batch = data[i : (i + batch_size)]
                batch = jax.tree_map(lambda *xs: jnp.stack(xs), *batch)
                agent, optim, loss = train_step(agent, optim, batch)
                losses.append(loss)

        value_loss, policy_loss = zip(*losses)
        value_loss = sum(value_loss).item() / len(value_loss)
        policy_loss = sum(policy_loss).item() / len(policy_loss)
        print(f" train losses:  value {value_loss:.3f}  policy {policy_loss:.3f}")
        win_count1, draw_count1, loss_count1 = agent_vs_agent_multiple_games(agent.eval(), old_agent, env, rng_key_2, num_simulations_per_move=num_simulations_per_move, num_games=num_sim_games)
        loss_count2, draw_count2, win_count2 = agent_vs_agent_multiple_games(old_agent, agent.eval(), env, rng_key_3, num_simulations_per_move=num_simulations_per_move, num_games=num_sim_games)

        print("  play against previous version: {} win - {} draw - {} loss".format(win_count1 + win_count2, draw_count1 + draw_count2, loss_count1 + loss_count2))
        with open(ckpt_filename, "wb") as f:
            dic = {
                "agent": agent.state_dict(),
                "optim": optim.state_dict(),
                "iter": iteration,
            }
            pickle.dump(dic, f)
    print("Done!")


if __name__ == '__main__':
    print("Cores :::: ", jax.local_devices())

    train = partial(train, game_class="chess_game.ChessGame", batch_size=32, num_iterations=512, num_simulations_per_move=64, num_self_plays_per_interaction=888, num_sim_games=32)

    fire.Fire(train)
