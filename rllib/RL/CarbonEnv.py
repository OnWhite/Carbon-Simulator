from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Any

import numpy as np
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces

from rllib.DP.DynamicProgram import DPImpl, Action, State


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
        self.max_hist_len = max(self.dp.delay, self.dp.forget)

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

        self.state = State(
            coin=0.0,
            carbon=0.0,
            research_yearly=0,
            research_count=0,
            labor=1.0,
            research_history=(0,) * self.max_hist_len,
            total_green=0.0,
            on_certificate=0,
            timestep=0,
        )

        return self._state_to_obs(self.state), {"action": None, "state": None}

    def step(self, action_idx: int):
        """Step with Action dataclass."""
        action = self.actions[action_idx]

        next_state = self.dp.state_transition(action, self.state)
        reward = self.dp.reward(next_state)

        terminated = next_state.timestep >= self.dp.max_timesteps - 1
        truncated = False

        self.state = next_state
        return (
            self._state_to_obs(next_state),
            reward,
            terminated,
            truncated,
            {"action": action, "state": next_state},
        )
    def single_transition(self, action_idx: int, state:State):
        """Step with Action dataclass."""
        action = self.actions[action_idx]

        next_state = self.dp.state_transition(action, state)
        reward = self.dp.reward(next_state)

        self.state = next_state
        return (
            next_state,
            reward
        )


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
