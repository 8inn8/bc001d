import pickle
import random
import warnings
from functools import partial

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from fire import Fire

from env import Environment
from tree_search import improve_policy_with_mcts, recurrent_fn
from utils import env_step, import_class, replicate, reset_env


@jax.jit
def _apply_temperature(logits, temperature):
    logits = logits - jnp.max(logits, keepdims=True, axis=-1)
    tiny = jnp.finfo(logits.dtype).tiny
    return logits / jnp.maximum(tiny, temperature)


@partial(jax.jit, static_argnames=("temperature", "num_simulations", "enable_mcts"))
def play_one_move(agent, env: Environment, rng_key: chex.Array, enable_mcts: bool = True, num_simulations: int = 2048, temperature=0.2):
    if enable_mcts:
        batched_env = replicate(env, 1)
        policy_output = improve_policy_with_mcts(agent, batched_env, rng_key, rec_fn=recurrent_fn, num_simulations=num_simulations, temperature=temperature)
        action = policy_output.action
        action_weights = jnp.log(policy_output.action_weights)
        root_idx = policy_output.search_tree.ROOT_INDEX
        value = policy_output.search_tree.node_values[0, root_idx]
    else:
        action_logits, value = agent(env.canonical_observation())
        action_logits_ = _apply_temperature(action_logits, temperature)
        action_weights = jax.nn.softmax(action_logits_, axis=-1)
        action = jr.categorical(rng_key, action_logits)

    return action, action_weights, value


@partial(jax.jit, static_argnames=("temperature", "num_simulations", "enable_mcts"))
def agent_vs_agent(agent1, agent2, env:Environment, rng_key: chex.Array, enable_mcts: bool = True, num_simulations_per_move: int = 2048, temperature: float = 0.2):
    def cond_fn(state):
        env, *_ = state
        return env.is_terminated() == False

    def loop_fn(state):
        env, a1, a2, _, rng_key, turn = state
        rng_key_1, rng_key_2 = jr.split(rng_key)
        action, _, _ = play_one_move(a1, env, rng_key_1, enable_mcts=enable_mcts, num_simulations=num_simulations_per_move, temperature=temperature)
        env, reward = env_step(env, action)
        state = (env, a2, a1, turn * reward, rng_key, -turn)
        return state

    state = (reset_env(env), agent1, agent2, jnp.array(0), rng_key, jnp.array(1))
    state = lax.while_loop(cond_fn, loop_fn, state)
    return state[3]


@partial(jax.jit, static_argnums=(4, 5, 6, 7))
def agent_vs_agent_multiple_games(agent1, agent2, env, rng_key, enable_mcts: bool = True, num_simulations_per_move: int = 2048, temperature: float = 0.2, num_games: int = 512):
    rng_keys = jr.split(rng_key, num_games)
    rng_keys = jnp.stack(rng_keys, axis=0)
    avsa = partial(agent_vs_agent, enable_mcts=enable_mcts, temperature=temperature, num_simulations_per_move=num_simulations_per_move)
    batched_avsa = jax.vmap(avsa, in_axes=(None, None, 0, 0))
    envs = replicate(env, num_games)
    results = batched_avsa(agent1, agent2, envs, rng_keys)
    win_count = jnp.sum(results == 1)
    draw_count = jnp.sum(results == 0)
    loss_count = jnp.sum(results == -1)
    return win_count, draw_count, loss_count


def human_vs_agent(agent, env: Environment, human_first: bool = True, enable_mcts: bool = False, num_simulations_per_move: int = 2048, temperature: float = 0.2, rng_key=None):
    env = reset_env(env)
    agent_turn = 1 if human_first else 0
    rng_key = jr.PRNGKey(random.randint(0, 999999)) if rng_key is None else rng_key
    for i in range(4096):
        print()
        print(f"Move {i}")
        print("=====")
        print()
        env.render()
        if i % 2 == agent_turn:
            print()
            s = env.canonical_observation()
            print("# s =", s)
            rng_key_1, rng_key = jr.split(rng_key)

            action, action_weights, value = play_one_move(agent, env, rng_key_1, enable_mcts=enable_mcts, num_simulations=num_simulations_per_move, temperature=temperature)
            print("#  A(s) =", action_weights)
            print("#  V(s) =", value)
            env, reward = env_step(env, action.item())
            print(f"#  Agent selected action {action}, got reward {reward}")
        else:
            action = int(input("> "))
            env, reward = env_step(env, action)
            print(f"#  Human selected action {action}, got reward {reward}")
        if env.is_terminated().item():
            break
    else:
        print("Timeout!")
    print()
    print("Final board")
    print("===========")
    print()
    env.render()
    print()
