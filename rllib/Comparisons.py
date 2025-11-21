import numpy as np


import numpy as np
import os

def eval_marl(trainer, env, n=20, run_dir=None):
    """
    Evaluate MARL policy only for agent 0 on RLlibEnvWrapper.

    Logs all debugging info to:  <run_dir>/dp_eval_debug.log
    """

    # === Create debug log file ===
    log_path = None
    if run_dir is not None:
        log_path = os.path.join(run_dir, "dp_eval_debug.log")
        f = open(log_path, "w")
        f.write("=== MARL EVALUATION DEBUG LOG ===\n\n")
    else:
        f = None  # no logging if run_dir missing

    def log(msg):
        if f:
            f.write(msg + "\n")

    # === Get policy ===
    policy = trainer.get_policy("a")
    planner_policy = None
    try:
        planner_policy = trainer.get_policy("p")
    except Exception:
        pass  # planner may not exist

    returns = []

    # === Run episodes ===
    for ep_idx in range(n):
        obs, info = env.reset()
        state = policy.get_initial_state()
        planner_state = None

        if planner_policy:
            planner_state = planner_policy.get_initial_state()

        # Agent IDs
        agent_keys = list(obs.keys())
        log(f"\n--- EPISODE {ep_idx} ---")
        log(f"Initial obs keys: {agent_keys}")

        # Pick agent_0
        agent0 = None
        planner_id = None

        for k in agent_keys:
            if k.isdigit() or k == "0" or k == "agent_0":
                agent0 = k
                break

        # Identify planner id
        for k in agent_keys:
            if k != agent0:
                planner_id = k
                break

        log(f"Using agent0 = {agent0}")
        log(f"Planner id = {planner_id}")

        done = False
        trunc = False
        ep_ret = 0.0
        t = 0

        # === Episode loop ===
        while not (done or trunc):
            t += 1
            obs0 = obs[agent0]

            # Compute agent action
            action0, state, _ = policy.compute_single_action(
                obs0, state=state, explore=False
            )

            # Prepare action dict
            action_dict = {agent0: action0}

            # Planner action if exists
            if planner_policy and planner_id in obs:
                obs_p = obs[planner_id]
                action_p, planner_state, _ = planner_policy.compute_single_action(
                    obs_p, state=planner_state, explore=False
                )
                action_dict[planner_id] = action_p
            else:
                action_p = None  # for logging

            # Step environment
            obs, rew, term, trunc, info = env.step(action_dict)

            # Log everything
            log(f"\nTimestep {t}:")
            log(f"  Obs keys: {list(obs.keys())}")
            log(f"  Agent0 action: {action0}")
            log(f"  Planner action: {action_p}")
            log(f"  Reward dict: {rew}")
            log(f"  Term flags: {term}")
            log(f"  Trunc flags: {trunc}")

            done = term.get("__all__", False)
            trunc = trunc.get("__all__", False)

            ep_ret += rew.get(agent0, 0.0)

        log(f"Episode return: {ep_ret}")
        returns.append(ep_ret)

    # Close log file
    if f:
        f.write("\n=== END OF LOG ===\n")
        f.close()

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
