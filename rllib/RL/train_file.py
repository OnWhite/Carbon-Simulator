from pathlib import Path

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms import ppo
from rllib.RL.CarbonEnv import CarbonEnv
import numpy as np

from rllib.RL.callback import ResultInfoMetricsCallback
from rllib.env_wrapper import pretty_print


def env_creator(env_config):
    return CarbonEnv(env_config)


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
            action = rl_algo.compute_single_action(rl_obs, explore=False, policy_id="a")
            rl_obs, reward, done, truncated, info = env.step(action)
            rl_ep_return += reward

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

            obs, reward, done, truncated, info = env.step(action_idx)
            dp_ep_return += reward

        dp_returns.append(dp_ep_return)

    print(f"RL  - Mean: {np.mean(rl_returns):.2f}, Std: {np.std(rl_returns):.2f}")
    print(f"DP  - Mean: {np.mean(dp_returns):.2f}, Std: {np.std(dp_returns):.2f}")
    print(f"Difference: {np.mean(rl_returns) - np.mean(dp_returns):.2f}")

    return rl_returns, dp_returns


if __name__ == "__main__":
    ray.init()

    register_env("carbon_env", env_creator)

    config = (
        ppo.PPOConfig()
        .environment(
            env="carbon_env",
            env_config={
                "config_path": "/Users/work/PycharmProjects/Carbon-Simulator/rllib/DP/config.yaml",
            },
        )
        .framework("torch")
        .callbacks(ResultInfoMetricsCallback)  # Pass the class
        .rollouts(num_rollout_workers=2)
        .training(
            gamma=0.998,
            lr=3e-4,
            train_batch_size=4000,
            sgd_minibatch_size=256,
            num_sgd_iter=10,
        )
    )

    algo = config.build()

    for i in range(10):
        result = algo.train()
        print(f"Iter {i}: reward_mean={result['episode_reward_mean']:.2f}")

        # NOW your custom metrics should appear here!
        for key, value in result['hist_stats'].items():
            arr = np.asarray(value)
            print(f"{key:15s}: {arr}")
    """# Load DP solution (already computed, no training)
    config_path = "/Users/work/PycharmProjects/Carbon-Simulator/rllib/DP/config.yaml"
    dp = DPImpl(load_config(Path(config_path)))
    dp.solve_mdp()  # One-time computation

    # Now compare
    env = CarbonEnv({"config_path": "/Users/work/PycharmProjects/Carbon-Simulator/rllib/DP/config.yaml"})

    compare_rl_vs_dp(algo, dp, env, n_eval_episodes=20)

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
    print("Std eval return:", np.std(returns))
    """
