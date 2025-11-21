import numpy as np


def eval_marl(trainer, env, n=20):
    """
    Evaluate MARL policy only for agent 0 on RLlibEnvWrapper.
    This fixes:
    FIX-1: Use RLlibEnvWrapper instead of CarbonEnv
    FIX-2: Use correct observation dict for ConvRnn
    FIX-3: Use LSTM state and deterministic actions
    FIX-4: Agent-id consistent ("0")
    """
    policy = trainer.get_policy("a")
    returns = []

    for _ in range(n):
        obs, info = env.reset()
        state = policy.get_initial_state()
        done = False
        ep_ret = 0.0

        while not done:
            # RLlibEnvWrapper returns dict: {agent_id: obs_dict}
            obs0 = obs["0"]
            action0, state, _ = policy.compute_single_action(
                obs0,
                state=state,
                explore=False
            )

            # Multiagent step: only agent 0 acts
            obs, rew, term, trunc, info = env.step({"0": action0})
            done = term["__all__"] or trunc["__all__"]
            ep_ret += rew["0"]

        returns.append(ep_ret)

    return np.mean(returns), np.std(returns)


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
