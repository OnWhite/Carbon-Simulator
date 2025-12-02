from pathlib import Path

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms import ppo
from dataclasses import replace

from rllib.DP.DynamicProgram import DPImpl, load_config, Action, State
from rllib.RL.CarbonEnv import CarbonEnv
import numpy as np

from rllib.RL.callback import ResultInfoMetricsCallback
from rllib.env_wrapper import pretty_print
import shutil
shutil.rmtree('/tmp/ray', ignore_errors=True)


def env_creator(env_config):
    return CarbonEnv(env_config)


def print_optimal_trajectory(algo, env):
    # Initial state (adapt to your model's start state)
    state = State(
        coin=0.0,
        carbon=0.0,
        research_yearly=0,
        research_count=0,
        labor=0,
        research_history=(0, 0),
        total_green=0,
        on_certificate=0,
        timestep=0
    )
    timesteps = env.get_max_timesteps()
    for t in range(timesteps):
        obs = env._state_to_obs(state)
        action = algo.compute_single_action(obs, explore=False)
        print(f"Current state (timestep {t}): {state}")
        print(f"RL selected action index: {action}")
        print(f"Environment-mapped action: {env.get_action(action)}")
        state, reward = env.single_transition(action, state)
        print(f"Reward received after transition: {reward:.4f}")


def compare_rl_to_dp(rl_algo, dp_instance, env):
    """Compare RL and DP policies on identical rollouts"""
    count = 0
    rewards1 = 0
    rewards2 = 0
    actions = [
        Action(b, g, r, m)
        for b in [0, 1]
        for g in [0, 1]
        for r in [0, 1]
        for m in [0, 1]
    ]
    startstate = State(0, 0, 0, 0, 0, (0, 0), 0, 0, 0)
    states = []
    for a in actions:
        states.append(dp_instance.state_transition(a, startstate))
    tot_states = states.copy()
    for state in states:
        for a in actions:
            tot_states.append(dp_instance.state_transition(a, state))
    for state in tot_states:
        observation = env._state_to_obs(state)
        action = rl_algo.compute_single_action(observation, explore=False)
        state_idx = dp_instance.state_to_index(state)
        action_idx = dp_instance.optimal_policy[state_idx]
        rewards1 += (dp_instance.reward(dp_instance.state_transition(dp_instance.actions[action], state)))
        rewards2 += (dp_instance.reward(dp_instance.state_transition(dp_instance.actions[action_idx], state)))
        if action != action_idx:
            if state.on_certificate == 0 and dp_instance.actions[action].green != dp_instance.actions[action_idx].green:
                new_action1 = replace(dp_instance.actions[action], green=0)
                new_action2 = replace(dp_instance.actions[action_idx], green=0)
                if new_action1 == new_action2:
                    continue
            else:
                new_action1 = dp_instance.actions[action]
                new_action2 = dp_instance.actions[action_idx]
            print(state)
            print(action)
            print(new_action1)
            print(action_idx)
            print(new_action2)
            count += 1
    print(f"Total mismatches: {count} out of {len(dp_instance.statespace)} states")
    print(f"Reward difference: {(rewards1 - rewards2) / (rewards1 + rewards2)}%")


def compare_rl_vs_dp(rl_algo, dp_instance, env, n_eval_episodes=20):
    """Compare RL and DP policies on identical rollouts"""
    rl_returns = []
    dp_returns = []

    for ep in range(n_eval_episodes):
        # Reset environment once
        obs, info = env.reset()

        # RL rollout
        rl_ep_return = 0.0
        done = False
        truncated = False
        rl_obs = obs.copy()

        while not (done or truncated):
            action = rl_algo.compute_single_action(rl_obs, explore=False)
            rl_obs, reward, done, truncated, info = env.step(action)
            rl_ep_return += reward
            print(env._obs_to_state(rl_obs))
            print(action)
            print(dp_instance.actions[action])
        rl_returns.append(rl_ep_return)

        # DP rollout (same initial state)
        obs, _ = env.reset(seed=ep)  # Use same seed
        dp_ep_return = 0.0
        done = False
        truncated = False

        while not (done or truncated):
            # Convert continuous obs to DP state tuple
            state_tuple = env._obs_to_state(obs)  # You need this helper
            state_idx = dp_instance.state_to_index(state_tuple)
            action_idx = dp_instance.optimal_policy[state_idx]
            print(state_tuple)
            print(action_idx)
            print(dp_instance.actions[action_idx])
            obs, reward, done, truncated, info = env.step(action_idx)
            dp_ep_return += reward

        dp_returns.append(dp_ep_return)

    print(f"RL  - Mean: {np.mean(rl_returns):.2f}, Std: {np.std(rl_returns):.2f}")
    print(f"DP  - Mean: {np.mean(dp_returns):.2f}, Std: {np.std(dp_returns):.2f}")
    print(f"Difference: {np.mean(rl_returns) - np.mean(dp_returns):.2f}")

    return rl_returns, dp_returns


if __name__ == "__main__":
    ray.init(
        _temp_dir="/nas/ucb/sophialudewig/ray_temp",
        object_store_memory=10 * 1024 * 1024 * 1024,  # Increase to 10GB
        num_cpus=32,  # Adjust based on your server
        num_gpus=1
    )

    register_env("carbon_env", env_creator)

    config = (
        ppo.PPOConfig()
        .environment(
            env="carbon_env",
            env_config={
                "config_path": "/nas/ucb/sophialudewig/Minimalist/rllib/DP/config.yaml",
            },
        )
        .framework("torch")
        .resources(num_gpus=1,
        num_cpus_per_worker=1)
        .callbacks(ResultInfoMetricsCallback)  # Pass the class
        .rollouts(num_rollout_workers=32)
        .training(
            gamma=0.998,
            lr=3e-4,
        ).reporting(min_time_s_per_iteration=0)
    )

    algo = config.build()

    for i in range(200):
        result = algo.train()
        # print(f"Iter {i}: reward_mean={result['episode_reward_mean']:.2f}")

        # NOW your custom metrics should appear here!
        """for key, value in result['hist_stats'].items():
            arr = np.asarray(value)
            print(f"{key:15s}: {arr}")"""
    # Load DP solution (already computed, no training)
    policy = algo.get_policy()
    model = policy.model
    env = CarbonEnv({"config_path": "/nas/ucb/sophialudewig/Minimalist/rllib/DP/config.yaml"})
    print_optimal_trajectory(algo, env)
""" config_path = "/Users/work/PycharmProjects/Carbon-Simulator/rllib/DP/config.yaml"
    dp = DPImpl(load_config(Path(config_path)))
    dp.solve_mdp()  # One-time computation

    # Now compare
    env = CarbonEnv({"config_path": "/Users/work/PycharmProjects/Carbon-Simulator/rllib/DP/config.yaml"})

    compare_rl_vs_dp(algo, dp, env)

    n_eval_episodes = 20
    returns = []

    for ep in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_return = 0.0

        while not (done or truncated):
            # use the TRAINED net, but deterministic
            action = algo.compute_single_action(obs, explore=False)
            obs, reward, done, truncated, info = env.step(action)
            ep_return += reward

            # here you can log (state, action, reward, carbon_idx, coin, etc.)

        returns.append(ep_return)

    print("Mean eval return:", np.mean(returns))
    print("Std eval return:", np.std(returns))"""
