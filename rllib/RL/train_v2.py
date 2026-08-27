"""V2 of the verification chain: single-agent RL on the reduced environment.

This is the tier that answers "can PPO recover the optimum". It trains on
exactly the MDP the exact dynamic program solves, so the comparison is not a
behavioural analogy -- it is a value gap measured in closed form.

What it reports, replacing `compare_rl_to_dp`:

    rel_regret                (V*(s0) - V^pi(s0)) / |V*(s0)|, by exact policy
                              evaluation over the reachable set. No rollouts,
                              so no sampling error.
    agree_occupancy_weighted  action agreement weighted by how often the
                              optimal policy actually visits each state.
    near_tie_share            of the disagreements, the fraction within
                              epsilon of optimal, i.e. ties rather than errors.

The old statistic counted argmax mismatches over 272 states reached by two
steps of enumeration and printed len(dp.statespace) -- 4032 -- as the
denominator.

Usage:
    PYTHONPATH=. python3 rllib/RL/train_v2.py --run_dir rllib/exp/verify_v2_s1 \
        --horizon 10 --seed 1 --max-hours 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import ray
import wandb
from ray.rllib.algorithms import ppo
from ray.tune.registry import register_env

from rllib.DP.DynamicProgram import DPImpl, State, load_config
from rllib.DP.exact_dp import ExactDP, state_key
from rllib.DP.verify import verify, optimal_action_of, git_sha
from rllib.RL.CarbonEnv import CarbonEnv

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "rllib" / "DP" / "config.yaml"


# ----------------------------------------------------------------------
# Credentials: identical policy to training_script.py -- never prompt.
# ----------------------------------------------------------------------

def wandb_auth() -> bool:
    mode = os.environ.get("WANDB_MODE", "").lower()
    if mode in ("offline", "disabled", "dryrun"):
        print(f"[wandb] WANDB_MODE={mode}: not syncing.")
        return True
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        try:
            import netrc as _netrc
            auth = _netrc.netrc().authenticators("api.wandb.ai")
            if auth and auth[2]:
                key = auth[2]
        except Exception:
            pass
    if not key:
        sys.exit(
            "W&B credentials not found.\n"
            "  export WANDB_API_KEY=<key from https://wandb.ai/authorize>\n"
            "or set WANDB_MODE=offline.\n"
            "Refusing to prompt: an interactive login hangs an unattended run."
        )
    # Do not call wandb.login(): it validates a 40-character key while the
    # server now issues the longer wandb_v1_* format. init() reads the env var.
    os.environ["WANDB_API_KEY"] = key
    print(f"[wandb] key configured from environment ({len(key)} chars)")
    return True


# ----------------------------------------------------------------------
# Turning a trained network into the `action_of` callable verify.py wants
# ----------------------------------------------------------------------

def policy_action_table(algo, env: CarbonEnv, ex: ExactDP, batch: int = 4096):
    """Greedy action for every reachable decision state, batched.

    One forward pass per state would be ~700k separate calls at T=10. Batching
    by layer keeps it to a few dozen. The result is a plain dict, so the
    policy queried by verify() is a genuine function of the state -- querying
    the network repeatedly would not be, and regret would be ill-defined.
    """
    policy = algo.get_policy()
    table = {}
    for layer in ex.layers[:-1]:                       # terminal layer decides nothing
        keys = list(layer.states.keys())
        if not keys:
            continue
        obs = np.stack([env._state_to_obs(layer.states[k]) for k in keys])
        for i in range(0, len(keys), batch):
            chunk = obs[i:i + batch]
            actions, _, _ = policy.compute_actions(chunk, explore=False)
            actions = np.asarray(actions).reshape(-1)
            for k, a in zip(keys[i:i + batch], actions):
                table[k] = int(a)
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--horizon", type=int, default=10,
                    help="T. Must match the horizon V1 is solved at.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--max-hours", type=float, default=2.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--project", type=str,
                    default=os.environ.get("WANDB_PROJECT", "carbon-verification"))
    args = ap.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    wandb_auth()

    sha = git_sha(REPO)
    tier = "v2-rl"
    env_config = {"config_path": str(args.config), "horizon": args.horizon}

    # ---- V1 first: the benchmark this run is measured against ----------
    print(f"[v1] solving the exact DP at T={args.horizon} ...")
    t0 = time.perf_counter()
    cfg = load_config(args.config)
    dp = DPImpl(cfg)
    gamma = float(cfg.get("agent_policy", {}).get("gamma", 0.998))
    ex = ExactDP(dp, horizon=args.horizon, gamma=gamma)
    ex.enumerate_reachable()
    ex.solve()
    v_star = ex.values[0][state_key(ex.start_state())]
    print(f"[v1] {ex.total_states():,} reachable states, V*(s0) = {v_star:.6f}, "
          f"{time.perf_counter() - t0:.1f}s")

    manifest = {
        "tier": tier, "commit": sha, "seed": args.seed, "horizon": args.horizon,
        "gamma": gamma, "v_star": v_star,
        "n_reachable_states": ex.total_states(),
        "p_research_success": ex.p_research, "p_permit": ex.p_permit,
        "config_path": str(args.config),
    }
    (args.run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str))

    wandb.init(
        project=args.project,
        group=tier,
        job_type="train",
        name=f"{tier}-T{args.horizon}-s{args.seed}",
        tags=[f"tier={tier}", f"T={args.horizon}", f"seed={args.seed}",
              f"commit={sha[:8]}"],
        config={**manifest, "iterations": args.iterations},
        dir=str(args.run_dir),
    )
    wandb.define_metric("env_steps")
    wandb.define_metric("*", step_metric="env_steps")

    # ---- V2: train ------------------------------------------------------
    ray.init(ignore_reinit_error=True, include_dashboard=False,
             num_cpus=args.num_workers + 1)
    register_env("carbon_env", lambda c: CarbonEnv(c))

    algo = (
        ppo.PPOConfig()
        .environment(env="carbon_env", env_config=env_config)
        .framework("torch")
        .rollouts(num_rollout_workers=args.num_workers)
        .debugging(seed=args.seed)
        .reporting(min_time_s_per_iteration=0,
                   metrics_num_episodes_for_smoothing=50)
        .training(gamma=gamma, lr=3e-4, train_batch_size=4000,
                  vf_clip_param=500.0)
        .build()
    )

    deadline = time.time() + args.max_hours * 3600 if args.max_hours > 0 else None
    for i in range(args.iterations):
        if deadline and time.time() >= deadline:
            print(f"[v2] wall-clock budget of {args.max_hours} h reached.")
            break
        r = algo.train()
        wandb.log({
            "env_steps": r["timesteps_total"],
            "episodes": r["episodes_total"],
            "iteration": r["training_iteration"],
            "policy/reward_mean": r.get("episode_reward_mean", float("nan")),
            "policy/reward_max": r.get("episode_reward_max", float("nan")),
        })
        if i % 25 == 0:
            print(f"[v2] iter {i} | env_steps {r['timesteps_total']} | "
                  f"reward_mean {r.get('episode_reward_mean', float('nan')):.4f}")

    # ---- V1 vs V2: the measurement --------------------------------------
    print("[verify] extracting the greedy policy over the reachable set ...")
    env = CarbonEnv(env_config)
    assert env.horizon == ex.horizon, (env.horizon, ex.horizon)
    table = policy_action_table(algo, env, ex)
    missing = sum(1 for layer in ex.layers[:-1] for k in layer.states if k not in table)
    assert missing == 0, f"{missing} reachable states have no action"

    report = verify(ex, lambda s: table[state_key(s)],
                    label=f"{tier}-T{args.horizon}-s{args.seed}", seed=args.seed)
    print()
    print(report.render())
    report.to_json(args.run_dir / "verification.json")
    wandb.log({**report.as_row(), "env_steps": algo._timesteps_total or 0})
    wandb.summary.update(report.as_row())

    algo.save(str(args.run_dir / "checkpoint"))
    wandb.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
