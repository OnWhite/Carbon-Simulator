"""Exact finite-horizon dynamic program over the reachable state space.

Replaces the bin-grid + `mdptoolbox.PolicyIteration` path in DynamicProgram.py.

Three differences that matter:

1.  No discretisation.  States are enumerated by forward search from the start
    state, so every value a variable actually takes is its own state.  The old
    grid pinned `research_count` and `research_yearly` to a single bin, which
    made research literally unrepresentable, and clipped coin into [-3, 3].

2.  Backward induction with V_T = 0, which is the Bellman equation the thesis
    states.  Policy iteration on a state space where `timestep` was a clipped
    feature let terminal states loop into themselves and accrue a discounted
    tail worth ~1/(1-gamma) periods of reward.

3.  Both exogenous shocks (research success, permit on move) enter as an
    explicit expectation over branches.  The old code drew `random.random()`
    inside the transition while assembling the transition matrix, which baked
    one unseeded coin flip per (state, action) pair into P with probability 1.

The value convention follows the environment: the reward of a transition is
`reward(next_state)`, matching `CarbonEnv.step`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from rllib.DP.DynamicProgram import Action, DPImpl, State, load_config

# Floats in the transition come from exp() and from sums of rationals, so two
# paths to "the same" state can differ in the last bits.  Keys are rounded to
# this many decimals; 1e-9 is far below any economically meaningful difference
# and far above float noise accumulated over a few dozen steps.
KEY_DECIMALS = 9

StateKey = Tuple

ALL_ACTIONS: List[Action] = [
    Action(b, g, r, m)
    for b in (0, 1)
    for g in (0, 1)
    for r in (0, 1)
    for m in (0, 1)
]


def state_key(s: State) -> StateKey:
    """Hashable identity of a state, stable against float noise."""
    return (
        round(float(s.coin), KEY_DECIMALS),
        round(float(s.carbon), KEY_DECIMALS),
        int(s.research_yearly),
        int(s.research_count),
        round(float(s.labor), KEY_DECIMALS),
        tuple(int(h) for h in s.research_history),
        round(float(s.total_green), KEY_DECIMALS),
        int(s.on_certificate),
        int(s.timestep),
    )


@dataclass
class Layer:
    """One timestep of the reachable set."""

    timestep: int
    states: Dict[StateKey, State]

    def __len__(self) -> int:
        return len(self.states)


class ExactDP:
    def __init__(
            self,
            dp: DPImpl,
            horizon: Optional[int] = None,
            gamma: float = 0.998,
            actions: Sequence[Action] = tuple(ALL_ACTIONS),
    ):
        self.dp = dp
        self.horizon = int(dp.max_timesteps if horizon is None else horizon)
        self.gamma = float(gamma)
        self.actions = list(actions)

        self.p_research = float(dp.p_research_success)
        self.p_permit = float(dp.p_permit)
        assert 0.0 <= self.p_research <= 1.0, self.p_research
        assert 0.0 <= self.p_permit <= 1.0, self.p_permit

        self.layers: List[Layer] = []
        self.values: List[Dict[StateKey, float]] = []
        self.policy: List[Dict[StateKey, int]] = []

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def start_state(self) -> State:
        return start_state(self.dp)

    def branches(self, action: Action, state: State) -> List[Tuple[float, State]]:
        """Expectation over the exogenous shocks for one (state, action).

        Shocks that the action cannot trigger collapse to a single branch, so a
        no-research no-move action yields exactly one successor.
        """
        if action.research:
            research = ((self.p_research, 1), (1.0 - self.p_research, 0))
        else:
            research = ((1.0, 0),)

        if action.move:
            permit = ((self.p_permit, 1), (1.0 - self.p_permit, 0))
        else:
            permit = ((1.0, 0),)

        out: List[Tuple[float, State]] = []
        for p_r, r in research:
            for p_m, m in permit:
                p = p_r * p_m
                if p <= 0.0:
                    continue
                out.append(
                    (p, self.dp.state_transition(
                        action, state, research_success=r, permit_granted=m))
                )
        total = sum(p for p, _ in out)
        assert abs(total - 1.0) < 1e-12, f"branch probabilities sum to {total}"
        return out

    # ------------------------------------------------------------------
    # Reachable set
    # ------------------------------------------------------------------

    def enumerate_reachable(self, start: Optional[State] = None,
                            cap: Optional[int] = None) -> List[Layer]:
        """Forward search, one layer per timestep.

        `cap` aborts with a RuntimeError once a layer exceeds that many states,
        so a horizon probe fails fast instead of exhausting memory.
        """
        s0 = self.start_state() if start is None else start
        layers = [Layer(0, {state_key(s0): s0})]

        for t in range(self.horizon):
            nxt: Dict[StateKey, State] = {}
            for s in layers[t].states.values():
                for a in self.actions:
                    for _, s2 in self.branches(a, s):
                        nxt.setdefault(state_key(s2), s2)
            if cap is not None and len(nxt) > cap:
                raise RuntimeError(
                    f"reachable layer t={t + 1} exceeded cap {cap} "
                    f"({len(nxt)} states)"
                )
            layers.append(Layer(t + 1, nxt))

        self.layers = layers
        return layers

    def total_states(self) -> int:
        return sum(len(layer) for layer in self.layers)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[Dict[StateKey, float]], List[Dict[StateKey, int]]]:
        """Backward induction.  V_T = 0 by construction."""
        if not self.layers:
            self.enumerate_reachable()

        # Terminal layer: no decision is taken, so V_T = 0 exactly.
        values: List[Dict[StateKey, float]] = [dict() for _ in range(self.horizon + 1)]
        policy: List[Dict[StateKey, int]] = [dict() for _ in range(self.horizon + 1)]
        values[self.horizon] = {k: 0.0 for k in self.layers[self.horizon].states}

        for t in range(self.horizon - 1, -1, -1):
            nxt_v = values[t + 1]
            v_t: Dict[StateKey, float] = {}
            pi_t: Dict[StateKey, int] = {}
            for k, s in self.layers[t].states.items():
                best_q = -math.inf
                best_a = 0
                for a_idx, a in enumerate(self.actions):
                    q = self.q_value(s, a, nxt_v)
                    if q > best_q:
                        best_q, best_a = q, a_idx
                v_t[k] = best_q
                pi_t[k] = best_a
            values[t] = v_t
            policy[t] = pi_t

        self.values = values
        self.policy = policy
        self._assert_terminal_zero()
        return values, policy

    def q_value(self, state: State, action: Action,
                next_values: Dict[StateKey, float]) -> float:
        q = 0.0
        for p, s2 in self.branches(action, state):
            q += p * (self.dp.reward(s2) + self.gamma * next_values.get(state_key(s2), 0.0))
        return q

    def _assert_terminal_zero(self) -> None:
        assert all(v == 0.0 for v in self.values[self.horizon].values()), \
            "terminal values are not exactly zero"

    # ------------------------------------------------------------------
    # Policy evaluation, occupancy, regret
    # ------------------------------------------------------------------

    def evaluate_policy(
            self, action_of: Callable[[State], int]
    ) -> List[Dict[StateKey, float]]:
        """Exact value of an arbitrary policy on the same MDP.

        No rollouts, so no sampling error: `action_of` is queried once per
        reachable state and the expectation is taken in closed form.
        """
        assert self.layers, "call enumerate_reachable() first"
        values: List[Dict[StateKey, float]] = [dict() for _ in range(self.horizon + 1)]
        values[self.horizon] = {k: 0.0 for k in self.layers[self.horizon].states}

        for t in range(self.horizon - 1, -1, -1):
            nxt_v = values[t + 1]
            values[t] = {
                k: self.q_value(s, self.actions[action_of(s)], nxt_v)
                for k, s in self.layers[t].states.items()
            }
        return values

    def occupancy(self, policy: Optional[List[Dict[StateKey, int]]] = None,
                  action_of: Optional[Callable[[State], int]] = None
                  ) -> List[Dict[StateKey, float]]:
        """Visitation probability of each reachable state under a policy.

        Used to weight action-agreement rates.  An unweighted count over the
        reachable set treats a state visited with probability 1e-6 exactly like
        the start state.
        """
        assert self.layers, "call enumerate_reachable() first"
        if action_of is None:
            assert policy is not None, "pass either policy or action_of"

            def action_of(s: State) -> int:  # noqa: F811
                return policy[s.timestep][state_key(s)]

        occ: List[Dict[StateKey, float]] = [dict() for _ in range(self.horizon + 1)]
        s0 = self.start_state()
        occ[0] = {state_key(s0): 1.0}

        for t in range(self.horizon):
            for k, mass in occ[t].items():
                if mass <= 0.0:
                    continue
                s = self.layers[t].states[k]
                a = self.actions[action_of(s)]
                for p, s2 in self.branches(a, s):
                    k2 = state_key(s2)
                    occ[t + 1][k2] = occ[t + 1].get(k2, 0.0) + mass * p
        return occ

    def regret(self, action_of: Callable[[State], int]) -> Dict[str, float]:
        """Normalised value gap at the start state, plus the raw values."""
        if not self.values:
            self.solve()
        k0 = state_key(self.start_state())
        v_star = self.values[0][k0]
        v_pi = self.evaluate_policy(action_of)[0][k0]
        denom = abs(v_star) if abs(v_star) > 1e-12 else 1.0
        return {
            "v_star": v_star,
            "v_pi": v_pi,
            "abs_regret": v_star - v_pi,
            "rel_regret": (v_star - v_pi) / denom,
        }

    def agreement(
            self,
            action_of: Callable[[State], int],
            epsilon: float = 1e-6,
    ) -> Dict[str, float]:
        """Action agreement with the optimal policy, three ways.

        Replaces the "200 of 4032 states" statistic, which counted mismatches
        over 272 states and printed the size of a 4032-state grid.

        Returns the unweighted rate over reachable decision states, the rate
        weighted by the optimal policy's occupancy measure, and the share of
        disagreements where the chosen action is within `epsilon` of optimal
        (i.e. a tie, not an error).
        """
        if not self.values:
            self.solve()
        occ = self.occupancy(policy=self.policy)

        n = 0
        n_agree = 0
        w_total = 0.0
        w_agree = 0.0
        n_disagree = 0
        n_near_tie = 0

        for t in range(self.horizon):
            nxt_v = self.values[t + 1]
            for k, s in self.layers[t].states.items():
                a_pi = action_of(s)
                a_star = self.policy[t][k]
                w = occ[t].get(k, 0.0)
                n += 1
                w_total += w
                if a_pi == a_star:
                    n_agree += 1
                    w_agree += w
                else:
                    n_disagree += 1
                    q_pi = self.q_value(s, self.actions[a_pi], nxt_v)
                    if self.values[t][k] - q_pi <= epsilon:
                        n_near_tie += 1

        return {
            "n_decision_states": float(n),
            "agree_unweighted": n_agree / n if n else float("nan"),
            "agree_occupancy_weighted": w_agree / w_total if w_total > 0 else float("nan"),
            "n_disagreements": float(n_disagree),
            "near_tie_share": (n_near_tie / n_disagree) if n_disagree else float("nan"),
            "epsilon": epsilon,
        }


# ----------------------------------------------------------------------
# The start state, defined once
# ----------------------------------------------------------------------

def start_state(dp: DPImpl) -> State:
    """Single definition of the episode start state.

    Previously this existed in three places with two different values:
    CarbonEnv.reset used labor=1.0, DP.print_optimal_trajectory used labor=0,
    and DP.verify_state_indexing used labor=1.  Every consumer imports this.
    """
    hist_len = dp.hist_len
    return State(
        coin=0.0,
        carbon=0.0,
        research_yearly=0,
        research_count=0,
        labor=0.0,
        research_history=(0,) * hist_len,
        total_green=0.0,
        on_certificate=0,
        timestep=0,
    )


def build(config_path: str | Path, horizon: Optional[int] = None) -> ExactDP:
    cfg = load_config(Path(config_path))
    dp = DPImpl(cfg)
    gamma = float(cfg.get("agent_policy", {}).get("gamma", 0.998))
    return ExactDP(dp, horizon=horizon, gamma=gamma)
