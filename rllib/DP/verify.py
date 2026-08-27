"""The verification report: what replaces `compare_rl_to_dp`.

The old statistic counted argmax mismatches over 272 states reached by
enumerating two steps of successors, and printed `len(dp.statespace)` (4032) as
the denominator. It also summed one-step rewards rather than comparing values,
and labelled a raw ratio as a percentage.

This module reports three numbers instead, each with a denominator that is the
set actually evaluated:

  rel_regret                (V*(s0) - V^pi(s0)) / |V*(s0)|, computed by exact
                            policy evaluation on the same MDP -- no rollouts,
                            so no sampling error.
  agree_occupancy_weighted  action agreement weighted by how often pi* actually
                            visits each state. An unweighted count treats a
                            state visited with probability 1e-6 like the start
                            state.
  near_tie_share            of the disagreements, the fraction where the chosen
                            action is within epsilon of optimal -- i.e. a tie
                            rather than an error. The thesis asserts this
                            qualitatively; here it is measured, with epsilon
                            stated.

Nothing here imports torch or ray, so it can be exercised against a tabular
policy in the test suite. `rllib/RL/rl_policy_adapter.py` turns a trained
RLlib algorithm into the `action_of` callable this module consumes.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from rllib.DP.DynamicProgram import State
from rllib.DP.exact_dp import ExactDP, state_key

ActionOf = Callable[[State], int]


def git_sha(repo: Optional[Path] = None) -> str:
    """Commit the numbers were produced at, or 'unknown' outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo or Path(__file__).resolve().parent),
            capture_output=True, text=True, check=True, timeout=10)
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo or Path(__file__).resolve().parent),
            capture_output=True, text=True, timeout=10).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


@dataclass
class VerificationReport:
    """Everything needed to reproduce and to judge one comparison."""

    label: str
    horizon: int
    gamma: float
    seed: Optional[int]

    # provenance
    commit: str = field(default_factory=git_sha)
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform_: str = field(default_factory=platform.platform)

    # model size
    n_reachable_states: int = 0
    n_decision_states: int = 0

    # shock probabilities, restated so the report is self-contained
    p_research_success: float = 0.0
    p_permit: float = 0.0

    # results
    v_star: float = 0.0
    v_pi: float = 0.0
    abs_regret: float = 0.0
    rel_regret: float = 0.0
    agree_unweighted: float = 0.0
    agree_occupancy_weighted: float = 0.0
    n_disagreements: int = 0
    near_tie_share: float = 0.0
    epsilon: float = 0.0

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    def as_row(self) -> Dict[str, object]:
        """Flat dict suitable for wandb.log or a results table."""
        return {
            "verify/rel_regret": self.rel_regret,
            "verify/abs_regret": self.abs_regret,
            "verify/v_star": self.v_star,
            "verify/v_pi": self.v_pi,
            "verify/agree_occupancy_weighted": self.agree_occupancy_weighted,
            "verify/agree_unweighted": self.agree_unweighted,
            "verify/near_tie_share": self.near_tie_share,
            "verify/n_disagreements": self.n_disagreements,
            "verify/n_decision_states": self.n_decision_states,
        }

    def render(self) -> str:
        pct = lambda x: "n/a" if x != x else f"{100.0 * x:.2f}%"
        return "\n".join([
            f"verification report: {self.label}",
            f"  commit                     {self.commit}",
            f"  horizon T                  {self.horizon}   (gamma {self.gamma})",
            f"  seed                       {self.seed}",
            f"  reachable states           {self.n_reachable_states:,}",
            f"  decision states evaluated  {self.n_decision_states:,}",
            f"  p(research success)        {self.p_research_success:.4f}",
            f"  p(permit | move)           {self.p_permit:.6g}",
            "",
            f"  V*(s0)                     {self.v_star:.6f}",
            f"  V^pi(s0)                   {self.v_pi:.6f}",
            f"  regret                     {self.abs_regret:.6f}"
            f"  ({pct(self.rel_regret)} of |V*|)",
            "",
            f"  action agreement",
            f"    occupancy weighted       {pct(self.agree_occupancy_weighted)}",
            f"    unweighted               {pct(self.agree_unweighted)}"
            f"   ({self.n_decision_states - self.n_disagreements:,}"
            f"/{self.n_decision_states:,})",
            f"    disagreements            {self.n_disagreements:,}",
            f"    of those, ties (<{self.epsilon:g})  {pct(self.near_tie_share)}",
        ])


def verify(
        ex: ExactDP,
        action_of: ActionOf,
        label: str,
        seed: Optional[int] = None,
        epsilon: float = 1e-6,
) -> VerificationReport:
    """Measure one policy against the exact optimum."""
    if not ex.layers:
        ex.enumerate_reachable()
    if not ex.values:
        ex.solve()

    reg = ex.regret(action_of)
    agr = ex.agreement(action_of, epsilon=epsilon)

    return VerificationReport(
        label=label,
        horizon=ex.horizon,
        gamma=ex.gamma,
        seed=seed,
        n_reachable_states=ex.total_states(),
        n_decision_states=int(agr["n_decision_states"]),
        p_research_success=ex.p_research,
        p_permit=ex.p_permit,
        v_star=reg["v_star"],
        v_pi=reg["v_pi"],
        abs_regret=reg["abs_regret"],
        rel_regret=reg["rel_regret"],
        agree_unweighted=agr["agree_unweighted"],
        agree_occupancy_weighted=agr["agree_occupancy_weighted"],
        n_disagreements=int(agr["n_disagreements"]),
        near_tie_share=agr["near_tie_share"],
        epsilon=epsilon,
    )


def optimal_action_of(ex: ExactDP) -> ActionOf:
    """The exact optimal policy as an `action_of` callable."""
    def f(s: State) -> int:
        return ex.policy[s.timestep][state_key(s)]
    return f


def render_table(reports: List[VerificationReport]) -> str:
    """One row per seed, for the verification table in the results section."""
    head = (f"{'label':<24} {'T':>3} {'V*':>10} {'V^pi':>10} "
            f"{'regret':>9} {'agree(occ)':>11} {'ties':>7}")
    lines = [head, "-" * len(head)]
    for r in reports:
        lines.append(
            f"{r.label:<24} {r.horizon:>3} {r.v_star:>10.4f} {r.v_pi:>10.4f} "
            f"{100 * r.rel_regret:>8.2f}% {100 * r.agree_occupancy_weighted:>10.2f}% "
            f"{100 * r.near_tie_share:>6.1f}%")
    return "\n".join(lines)
