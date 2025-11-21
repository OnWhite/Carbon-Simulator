import numpy as np


import numpy as np
import os


def eval_marl(trainer, env, n=20):
    policy_a = trainer.get_policy("a")
    policy_p = trainer.get_policy("p")
    returns = []

    for ep in range(n):
        obs, info = env.reset()
        states = {aid: policy_a.get_initial_state() for aid in obs.keys() if aid != 'p'}
        state_p = policy_p.get_initial_state()
        done = False
        ep_ret = 0

        while not done:
            actions = {}

            # Get actions for all agents
            for agent_id in obs.keys():
                if agent_id == 'p':
                    action, state_p, _ = policy_p.compute_single_action(
                        obs[agent_id], state_p, explore=False
                    )
                else:
                    action, states[agent_id], _ = policy_a.compute_single_action(
                        obs[agent_id], states[agent_id], explore=False
                    )
                actions[agent_id] = action

            obs, rew, done_dict, truncated_dict, info = env.step(actions)
            done = done_dict.get('__all__', False) or truncated_dict.get('__all__', False)

            # Only accumulate reward for agent "0"
            ep_ret += rew.get("0", 0)

        returns.append(ep_ret)

    returns = np.array(returns, dtype=float)
    return returns, float(returns.mean()), float(returns.std())


def eval_dp(dp, env, n=20):

    returns = []

    for _ in range(n):
        obs, info = env.reset()
        dp_ep_return = 0.0
        done = False
        truncated = False

        while not (done or truncated):

            state_tuple = env._obs_to_state(obs)
            state_idx = dp.state_to_index(state_tuple)
            action_idx = dp.optimal_policy[state_idx]

            obs, reward, done, truncated, info = env.step(action_idx)
            dp_ep_return += reward

        returns.append(dp_ep_return)

    return np.mean(returns), np.std(returns)
