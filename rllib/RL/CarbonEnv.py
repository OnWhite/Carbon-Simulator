from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Any

import numpy as np
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces

from rllib.DP.DynamicProgram import DPImpl, Action, State
from rllib.DP.exact_dp import start_state


class CarbonEnv(gym.Env):
    """
    Gymnasium wrapper around DPImpl using Action and State dataclasses.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__()
        config = config or {}

        cfg_path = Path(
            config.get(
                "config_path",
                "/nas/ucb/sophialudewig/Minimalist/rllib/DP/config.yaml",
            )
        )

        cfg = self.load_config(cfg_path)
        self.dp = DPImpl(cfg)

        # Let a run override the horizon without editing the shared config, so
        # the RL tier can train at whatever T the exact DP can still verify.
        if config.get("horizon") is not None:
            self.dp.max_timesteps = int(config["horizon"])

        # Action set (16 actions)
        self.actions = [
            Action(b, g, r, m)
            for b in [0, 1]
            for g in [0, 1]
            for r in [0, 1]
            for m in [0, 1]
        ]
        self.n_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.n_actions)

        # Observation space
        self.max_hist_len = self.dp.hist_len

        high = np.array(
            [
                2000,  # coin
                2000,  # carbon
                self.dp.yearsteps,  # research_yearly
                self.dp.max_timesteps,  # research_count  <-- NEW
                10000,  # labor
                *([1.0] * self.max_hist_len),  # research history bits
                self.dp.total_idx,  # total_green
                1.0,  # on_certificate
                self.dp.max_timesteps
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32,
        )

        self.state: State | None = None

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def load_config(self, path: Path) -> Dict[str, Any]:
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _state_to_obs(self, s: State) -> np.ndarray:
        rh = list(s.research_history)
        if len(rh) < self.max_hist_len:
            rh += [0] * (self.max_hist_len - len(rh))
        else:
            rh = rh[: self.max_hist_len]

        return np.array(
            [
                s.coin,
                s.carbon,
                s.research_yearly,
                s.research_count,
                s.labor,
                *rh,
                s.total_green,
                s.on_certificate,
                s.timestep,
            ],
            dtype=np.float32,
        )

    def _obs_to_state(self, obs: np.ndarray) -> State:
        rh = tuple(int(obs[5 + i]) for i in range(self.max_hist_len))

        return State(
            coin=float(obs[0]),
            carbon=float(obs[1]),
            research_yearly=int(obs[2]),
            research_count=int(obs[3]),
            labor=float(obs[4]),
            research_history=rh,
            total_green=float(obs[5 + self.max_hist_len]),
            on_certificate=int(obs[6 + self.max_hist_len]),
            timestep=int(obs[7 + self.max_hist_len]),
        )

    # ----------------------------------------------------------------------
    # Gym API
    # ----------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # One shared definition of the start state (rllib.DP.exact_dp), instead
        # of three copies that disagreed about the initial labor value.
        self.state = start_state(self.dp)

        return self._state_to_obs(self.state), {"action": None, "state": None}

    def step(self, action_idx: int):
        """Step with Action dataclass."""
        action = self.actions[action_idx]

        # Both shocks are drawn every step, whether or not the action can
        # trigger them, so that the RNG stream stays aligned across policies
        # and a seeded episode is comparable between runs.
        research_success = int(self.np_random.random() < self.dp.p_research_success)
        permit_granted = int(self.np_random.random() < self.dp.p_permit)

        next_state = self.dp.state_transition(
            action,
            self.state,
            research_success=research_success,
            permit_granted=permit_granted,
        )
        reward = self.dp.reward(next_state, self.state)

        # T decisions per episode, at t = 0 .. T-1. The old bound was
        # `max_timesteps - 1`, which terminated after a single decision when
        # episode_length was 2, so the agent solved a one-step bandit while the
        # dynamic program planned over the full horizon.
        terminated = next_state.timestep >= self.dp.max_timesteps
        truncated = False

        self.state = next_state
        return (
            self._state_to_obs(next_state),
            reward,
            terminated,
            truncated,
            {"action": action, "state": next_state},
        )
    def single_transition(self, action_idx: int, state: State,
                          research_success: int = 1, permit_granted: int = 0):
        """Advance one step from an arbitrary state with given shock outcomes."""
        action = self.actions[action_idx]

        next_state = self.dp.state_transition(
            action, state,
            research_success=research_success,
            permit_granted=permit_granted,
        )
        reward = self.dp.reward(next_state, state)

        self.state = next_state
        return next_state, reward

    def deterministic_step(self, action_idx: int, state: State,
                          research_success: int, permit_granted: int):
        """Pure transition used by the parity test. Does not touch self.state."""
        next_state = self.dp.state_transition(
            self.actions[action_idx], state,
            research_success=research_success,
            permit_granted=permit_granted,
        )
        return next_state, self.dp.reward(next_state, state)

    @property
    def horizon(self) -> int:
        return int(self.dp.max_timesteps)


    def render(self):
        pass
    def get_action(self, action_idx: int) -> Action:
        """Get Action dataclass from action index."""
        return self.actions[action_idx]
    def get_state(self,state_idx) -> State:
        """Get State dataclass from state index."""
        return self.dp.index_to_state(state_idx)
    def get_max_timesteps(self) ->int:
        return self.dp.max_timesteps
