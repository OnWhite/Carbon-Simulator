from copy import deepcopy

import numpy as np
import math
from typing import Dict, Any, List, Tuple
from pathlib import Path
import yaml
import mdptoolbox
import random
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Action:
    build: int
    green: int
    research: int
    move: int


@dataclass
class State:
    coin: float
    carbon: float
    research_yearly: int
    research_count: int
    labor: float
    research_history: Tuple[int, ...]
    total_green: float
    on_certificate: int
    timestep: int


class DPImpl:
    """
    State: [
        state.coin_total, state.carbon_idx_total, action.research_total, state.research_yearly,
        state.labor_total, state.research_history_arr, state.total_action.green, pos_xy
    ]
    Action: [action.build, action.green, action.research, action.move]
    Planner: list of (year_percent, punishment)
    """

    def __init__(self, cfg: Dict[str, Any]):
        env = cfg["env"]
        comps = flatten_components(cfg)

        # Direct from config
        self.payment: float = comps["Carbon_component"].get("payment", 10)
        self.require_carbon_idx: float = comps["Carbon_component"].get("require_Carbon_idx", 1.0)
        self.lowest_rate: float = comps["Carbon_component"].get("lowest_rate", 0.1)
        self.a: float = comps["Carbon_component"].get("research_setting", {})[1]
        self.failrate: float = comps["Carbon_component"].get("random_fails")
        self.delay: int = comps["Carbon_component"]["delay"]
        self.forget: int = comps["Carbon_component"]["forget"]
        self.yearsteps: int = env.get("period", 50)
        self.total_idx: int = comps["CarbonRedistribution"].get("total_idx", 200)
        self.isoelastic_eta: float = env.get("isoelastic_eta", 0.23)
        self.energy_cost: float = env.get("energy_cost", 0.1)
        self.energy_warmup_constant: float = env.get("energy_warmup_constant", 1)

        # Mapped / renamed
        self.collect_cost: float = comps.get("Gather", {}).get("collect_cost_coin", 10)

        # Required extras (must be provided somewhere in cfg; raise if missing)
        self.research_ability = comps.get("research_ability", 1)
        self.manufacture_volume = comps.get("manufacture_volume", 1)
        self.collectidx = comps.get("Gather", {}).get("collect_idx", 1.0)
        self.l_move = comps.get("Gather", {}).get("move_labor", 1)
        self.l_build = comps["Carbon_component"].get("labor", 1)
        self.l_research = comps["Carbon_component"].get("labor", 1)
        self.l_green = comps.get("Gather", {}).get("collect_labor", 1)
        self.planner = [(100, 5)]
        self.left_envidx = 0
        self.greenbudget: float = 0.0
        self.max_greenbudget = 0.0
        self.max_timesteps = env.get("episode_length", 2)
        print(self.total_idx)
        self.max_greenbudget = (1 / 3) * self.total_idx
        ws = env.get("world_size", (100, 100))
        self.worldsize = ws[0] * ws[1]
        self.statespace = []

    from copy import deepcopy

    def state_transition(self, action: Action, state: State) -> State:
        new_state = deepcopy(state)
        new_action = deepcopy(action)

        # Research history must become a list for mutation
        hist = list(new_state.research_history)
        if not hist:
            hist.append(0)

        # Year bucket logic
        start_idx = 0
        if new_state.timestep % self.yearsteps == 0:
            year = new_state.timestep // self.yearsteps
            if year < len(self.planner):
                year_pct = self.planner[year][0]
                start_idx = (year_pct / 100.0) * 0.5 * self.total_idx

        # Pipeline logic
        if new_state.research_yearly > 0 and sum(hist[: self.forget]) == 0:
            new_state.research_yearly -= 1
            new_state.research_count -= 1
            hist[0] = 2
        if len(hist) > self.delay and hist[self.delay] == 1:
            new_state.research_yearly += 1
            new_state.research_count += 1

        total_power_eff = math.exp(-new_state.research_count * self.a * self.research_ability)
        green_rate = max(1.0 - new_state.total_green / (total_power_eff + new_state.total_green), 0)
        emit_rate = max(total_power_eff * green_rate, self.lowest_rate)

        # Shift history
        hist[1:] = hist[:-1]
        hist[0] = 1 if new_action.research else 0

        # Green budget
        if new_state.total_green >= self.max_greenbudget and new_action.green:
            new_action.green = 0
        elif new_action.green and new_state.on_certificate == 1:
            new_state.total_green += self.collectidx
            new_state.on_certificate = 0
        elif new_action.green:
            new_action.green = 0

        # Move → certificate
        if new_action.move == 1:
            if random.random() <= self.max_greenbudget / self.worldsize:
                new_state.on_certificate = 1

        # Coin update
        new_state.coin += (
                self.payment * self.manufacture_volume * new_action.build
                - self.collect_cost * new_action.green
                - self.payment / (2 * self.research_ability) * new_action.research
        )

        # Carbon update
        new_state.carbon += (
                start_idx
                - self.require_carbon_idx * self.manufacture_volume * new_action.build * emit_rate
                + self.collectidx * new_action.green
        )

        # Labor update
        new_state.labor += (
                self.l_build * new_action.build
                + self.l_research * new_action.research * self.research_ability
                + self.l_green * new_action.green
                + self.l_move * new_action.move
        )

        # Yearly punishment
        if (new_state.timestep + 1) % self.yearsteps == 0:
            year = min(new_state.timestep // self.yearsteps, len(self.planner) - 1)
            punish = self.planner[year][1]
            if new_state.carbon < 0:
                new_state.coin -= (-new_state.carbon) * punish
                new_state.carbon = 0

        # Finalise
        new_state.research_history = tuple(hist)
        new_state.timestep += 1
        return new_state

    def discretize_state_space1(self):
        """Create discrete bins for each state variable"""
        # Reduce granularity to make problem tractable
        self.coin_bins = np.linspace(
            -self.payment / (2 * self.research_ability) * self.max_timesteps,
            self.payment * self.manufacture_volume * self.total_idx,
            3
        )
        self.carbon_bins = np.linspace(
            -self.require_carbon_idx * self.max_timesteps,
            self.total_idx,
            3
        )
        self.research_yearly_bins = np.linspace(0, self.yearsteps, 3)
        self.labor_bins = np.linspace(
            0,
            self.l_build * self.max_timesteps + self.l_research * self.max_timesteps,
            3
        )
        self.total_green_bins = np.linspace(0, self.total_idx, 3)
        self.on_certificate_bins = [0, 1]
        # Research history as bit patterns (0-2 states per position)
        max_history_len = max(self.delay, self.forget)
        self.history_states = 2 ** max_history_len
        self.timestep = range(0, self.max_timesteps)

        # Calculate total number of states
        self.n_states = (
                len(self.coin_bins) *
                len(self.carbon_bins) *
                len(self.research_yearly_bins) *
                len(self.labor_bins) *
                self.history_states *
                len(self.total_green_bins) *
                len(self.on_certificate_bins) *
                len(self.timestep)

        )

        print(f"Total discrete states: {self.n_states}")

    def discretize_state_space(self):
        # Exact unique values observed in your printed next-states
        self.coin_bins = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        self.carbon_bins = np.array([0, 1.0, 2.0])
        self.research_yearly_bins = np.array([0])  # or np.array([0, 1]) if you want to allow growth
        self.labor_bins = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        self.total_green_bins = np.array([0, 1])
        self.on_certificate_bins = np.array([0, 1])
        self.timestep = ([0, 1, 2])  # your printed next-states always have t=2
        self.research_count_bins = np.array([0])
        max_history_len = max(self.delay, self.forget)  # used for history bit-encoding
        self.history_states = 2 ** max_history_len

        self.n_states = (
                len(self.coin_bins) *
                len(self.carbon_bins) *
                len(self.research_yearly_bins) *
                len(self.labor_bins) *
                self.history_states *
                len(self.total_green_bins) *
                len(self.on_certificate_bins) *
                len(self.timestep) *
                len(self.research_count_bins)
        )
        print(f"Total discrete states: {self.n_states}")

    def state_to_index(self, s: State) -> int:
        """Convert a State object into a discrete index."""

        # Digitize each scalar field
        coin_idx = np.clip(np.digitize(s.coin, self.coin_bins) - 1, 0, len(self.coin_bins) - 1)
        carbon_idx = np.clip(np.digitize(s.carbon, self.carbon_bins) - 1, 0, len(self.carbon_bins) - 1)
        ryearly_idx = np.clip(np.digitize(s.research_yearly, self.research_yearly_bins) - 1,
                              0, len(self.research_yearly_bins) - 1)
        rcount_idx = np.clip(np.digitize(s.research_count, self.research_count_bins) - 1,
                             0, len(self.research_count_bins) - 1)
        labor_idx = np.clip(np.digitize(s.labor, self.labor_bins) - 1, 0, len(self.labor_bins) - 1)
        total_green_idx = np.clip(np.digitize(s.total_green, self.total_green_bins) - 1,
                                  0, len(self.total_green_bins) - 1)
        on_cert_idx = np.clip(np.digitize(s.on_certificate, self.on_certificate_bins) - 1,
                              0, len(self.on_certificate_bins) - 1)
        timestep_idx = np.clip(np.digitize(s.timestep, self.timestep) - 1,
                               0, len(self.timestep) - 1)

        # Research history → integer
        max_hist_len = max(self.delay, self.forget)
        hist_idx = sum(v * (2 ** i) for i, v in enumerate(s.research_history[:max_hist_len]))
        hist_idx = min(hist_idx, self.history_states - 1)

        # Lexicographic product:
        # coin → carbon → r_yearly → r_count → labor → history → total_green → on_cert → timestep
        index = coin_idx
        index = index * len(self.carbon_bins) + carbon_idx
        index = index * len(self.research_yearly_bins) + ryearly_idx
        index = index * len(self.research_count_bins) + rcount_idx
        index = index * len(self.labor_bins) + labor_idx
        index = index * self.history_states + hist_idx
        index = index * len(self.total_green_bins) + total_green_idx
        index = index * len(self.on_certificate_bins) + on_cert_idx
        index = index * len(self.timestep) + timestep_idx

        return int(index)

    from copy import deepcopy

    def build_transition_and_reward_matrices(self):
        """Build P[a][s][s'] and R[a][s] using dataclasses and research_count"""

        # Action space as dataclasses
        self.actions = [
            Action(b, g, r, m)
            for b in (0, 1)
            for g in (0, 1)
            for r in (0, 1)
            for m in (0, 1)
        ]
        self.n_actions = len(self.actions)

        P = np.zeros((self.n_actions, self.n_states, self.n_states))
        R = np.zeros((self.n_actions, self.n_states))

        print("Building transition matrices with full enumeration...")
        print(f"State space: {self.n_states} states, {self.n_actions} actions")

        states_visited = set()
        transitions_recorded = 0
        sample_transitions = []

        max_hist_len = max(self.delay, self.forget)
        total_combinations = 0

        # === Enumerate State Space ===
        for coin in self.coin_bins:
            for carbon in self.carbon_bins:
                for r_yearly in self.research_yearly_bins:
                    for r_count in self.research_count_bins:
                        for labor in self.labor_bins:
                            for total_green in self.total_green_bins:
                                for on_certificate in self.on_certificate_bins:
                                    for timestep in self.timestep:
                                        for hist_idx in range(self.history_states):

                                            # Decode history bits
                                            r_hist = tuple(
                                                (hist_idx // (2 ** i)) % 2
                                                for i in range(max_hist_len)
                                            )

                                            # Build State dataclass
                                            state = State(
                                                coin=coin,
                                                carbon=carbon,
                                                research_yearly=int(r_yearly),
                                                research_count=int(r_count),
                                                labor=labor,
                                                research_history=r_hist,
                                                total_green=total_green,
                                                on_certificate=on_certificate,
                                                timestep=timestep
                                            )
                                            self.statespace.append(state)
                                            state_idx = self.state_to_index(state)
                                            states_visited.add(state_idx)

                                            # === For each action ===
                                            for a_idx, act in enumerate(self.actions):

                                                # If research fails
                                                if act.research == 1:
                                                    # Success branch
                                                    next_succ = self.state_transition(act, state)
                                                    next_succ_idx = self.state_to_index(next_succ)
                                                    P[a_idx, state_idx, next_succ_idx] = 1.0 - self.failrate

                                                    # Failure branch (same action but research=0)
                                                    fail_act = Action(
                                                        build=act.build,
                                                        green=act.green,
                                                        research=0,
                                                        move=act.move
                                                    )
                                                    next_fail = self.state_transition(fail_act, state)
                                                    next_fail_idx = self.state_to_index(next_fail)
                                                    P[a_idx, state_idx, next_fail_idx] = self.failrate

                                                    rew_succ = self.reward(next_succ)
                                                    rew_fail = self.reward(next_fail)
                                                    R[a_idx, state_idx] = (
                                                            (1 - self.failrate) * rew_succ +
                                                            self.failrate * rew_fail
                                                    )

                                                    # Logging
                                                    if len(sample_transitions) < 5:
                                                        sample_transitions.append({
                                                            "state": state,
                                                            "action": act,
                                                            "next_success": next_succ,
                                                            "next_failure": next_fail,
                                                            "reward_success": rew_succ,
                                                            "reward_failure": rew_fail,
                                                            "expected_reward": R[a_idx, state_idx],
                                                        })

                                                else:
                                                    # Deterministic transition
                                                    next_state = self.state_transition(act, state)
                                                    next_idx = self.state_to_index(next_state)
                                                    P[a_idx, state_idx, next_idx] = 1.0
                                                    R[a_idx, state_idx] = self.reward(next_state)

                                                    if len(sample_transitions) < 5:
                                                        sample_transitions.append({
                                                            "state": state,
                                                            "action": act,
                                                            "next_state": next_state,
                                                            "reward": R[a_idx, state_idx]
                                                        })

                                                transitions_recorded += 1

                                            total_combinations += 1
                                            if total_combinations % 1000 == 0:
                                                print(f"Processed {total_combinations}/{self.n_states} states...")

        print(f"Completed: {total_combinations} states enumerated")

        print("\n=== DIAGNOSTIC INFO ===")
        print(f"Unique state indices visited: {len(states_visited)}")
        print(f"Expected states: {self.n_states}")
        print(f"Transitions recorded: {transitions_recorded}")

        print("\n=== SAMPLE TRANSITIONS ===")
        for i, sample in enumerate(sample_transitions[:3]):
            print(f"\nSample {i + 1}:")
            print(f"  State: {sample['state']}")
            print(f"  Action: {sample['action']}")
            if "next_state" in sample:
                print(f"  Next: {sample['next_state']}")
                print(f"  Reward: {sample['reward']:.2f}")
            else:
                print(f"  Next (success): {sample['next_success']}")
                print(f"  Next (failure): {sample['next_failure']}")
                print(f"  Expected reward: {sample['expected_reward']:.2f}")

        print("\n=== REWARD STATISTICS ===")
        print(f"Reward range: [{R.min():.2f}, {R.max():.2f}]")
        print(f"Non-zero rewards: {(R != 0).sum()} / {R.size}")
        print(f"Positive rewards: {(R > 0).sum()}")
        print(f"Negative rewards: {(R < 0).sum()}")

        return P, R.T

    def verify_state_indexing(self):
        print("\n=== VERIFYING STATE INDEXING ===")

        # Two simple test states
        test_states = [
            State(
                coin=0.0,
                carbon=0.0,
                research_yearly=0,
                research_count=0,
                labor=1,
                research_history=(0, 0),
                total_green=0,
                on_certificate=0,
                timestep=0
            ),
            State(
                coin=2.0,
                carbon=1.0,
                research_yearly=0,
                research_count=0,
                labor=2,
                research_history=(1, 0),
                total_green=0,
                on_certificate=1,
                timestep=1
            ),
        ]

        for s in test_states:
            idx = self.state_to_index(s)
            print(f"State {s} -> index {idx}")

        # Check if distinct states map to distinct indices
        unique_indices = set()
        print("\nChecking first 4 combinations:")
        for coin in self.coin_bins[:2]:
            for carbon in self.carbon_bins[:2]:
                s = State(
                    coin=coin,
                    carbon=carbon,
                    research_yearly=0,
                    research_count=0,
                    labor=1,
                    research_history=(0, 0),
                    total_green=0,
                    on_certificate=0,
                    timestep=0
                )
                idx = self.state_to_index(s)
                unique_indices.add(idx)
                print(f"  (coin={coin}, carbon={carbon}) -> index {idx}")

        print(f"Unique indices from 4 states: {len(unique_indices)} (expected: 4)")

    def run_ddp_example(self):
        coin_total = np.linspace(-self.payment / (2 * self.research_ability) * self.max_timesteps,
                                 self.payment * self.manufacture_volume * self.total_idx, 10)
        carbon_idx_total = np.linspace(- self.require_carbon_idx * self.max_timesteps, self.total_idx, 10)
        research_total, research_yearly = np.linspace(0, self.max_timesteps, 10), np.linspace(0, self.yearsteps, 10)
        labor_total = np.linspace(0, self.l_build * self.max_timesteps + self.l_research * self.max_timesteps, 10)
        # total_green=np.linspace(0,self.total_idx,10)
        # timestep=np.linspace(0,self.max_timesteps,10)
        lenresrach_hist = max(self.delay, self.forget)
        research_history = range(1 << lenresrach_hist)

        ####### Actions #######
        build_actions = [0, 1]  # build or not
        green_actions = [0, 1]  # green or not
        research_actions = [0, 1]  # research or not

        Grid = [(b, g, r) for b in green_actions for g in green_actions for r in research_actions]

    def solve_mdp(self):
        """Solve the MDP after discretising and validating indexing."""
        self.discretize_state_space()
        self.verify_state_indexing()

        P, R = self.build_transition_and_reward_matrices()

        mdp = mdptoolbox.mdp.PolicyIteration(P, R, discount=0.998)
        mdp.run()

        # mdp.policy is array of shape (horizon, n_states)
        # Each entry is the optimal action index for that state at that time
        self.optimal_policy = mdp.policy
        return self.optimal_policy

    def reward(self, s: State) -> float:
        """Compute reward for a State dataclass."""
        labor_coeff = self.energy_cost * (
                1.0 - math.exp(-s.timestep / self.energy_warmup_constant)
        )

        return isoelastic_coin_minus_labor(
            coin_endowment=s.coin,
            total_labor=s.labor,
            isoelastic_eta=self.isoelastic_eta,
            labor_coefficient=labor_coeff,
        )


def isoelastic_coin_minus_labor(
        coin_endowment, total_labor, isoelastic_eta, labor_coefficient
):
    assert 0 <= isoelastic_eta <= 1.0
    if isoelastic_eta == 1.0:
        util_c = np.log(np.maximum(1, coin_endowment))
    else:
        if np.all(np.asarray(coin_endowment) >= 0):
            util_c = (coin_endowment ** (1 - isoelastic_eta) - 1) / (1 - isoelastic_eta)
        else:
            util_c = coin_endowment - 1
    util_l = total_labor * labor_coefficient
    return float(util_c)


def load_config(path: Path) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def flatten_components(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested component structure from config"""
    env_cfg = cfg.get("env", {})
    components = env_cfg.get("components", [])

    # Convert list of component dicts to single dict
    flattened = {}
    for component in components:
        flattened.update(component)

    return flattened


def main():
    """Main function using YAML config"""
    CONFIG_PATH = Path("/Users/work/PycharmProjects/Carbon-Simulator/rllib/DP/config.yaml")
    cfg = load_config(CONFIG_PATH)

    try:
        dp = DPImpl(cfg)
        print("✓ DP instance created")
        print(f"  - Max timesteps: {dp.max_timesteps}")
        print(f"  - Period length: {dp.yearsteps}")
        print(f"  - Delay: {dp.delay}, Forget: {dp.forget}")
        print(f"  - Failure rate: {dp.failrate}")
    except Exception as e:
        print(f"❌ Failed to initialize DP: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    print("\n" + "=" * 60)
    print("Building and Solving MDP")
    print("=" * 60)

    try:
        policy = dp.solve_mdp()

        if policy is None:
            print("❌ No policy returned (MDP solution failed).")
            return dp, policy

        print("\n" + "=" * 60)
        print("Solution Complete!")
        print("=" * 60)

        # mdptoolbox PolicyIteration returns a vector of size n_states
        print("\nOptimal policy (first 10 states):")
        print(policy[:10])
        print(f"Number of states: {len(policy)}")

        print("\n" + "=" * 60)
        print("Example Decoded Actions:")
        print("=" * 60)

        for s in range(10):
            action_idx = policy[s]
            action = dp.actions[action_idx]
            print(
                f"state={s}: action_idx={action_idx} "
                f"-> (build={action.build}, green={action.green}, "
                f"research={action.research}, move={action.move})"
            )

        print_state_matrix(dp)
        print_optimal_trajectory(dp)
        return dp, policy

    except Exception as e:
        print(f"\n❌ Error during MDP solution: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def print_state_matrix(dp, num=10):
    """Print the first `num` states in exact index order (0,1,2,...)."""

    print("\n=== State Matrix Sample ===")

    max_hist_len = max(dp.delay, dp.forget)
    printed = 0

    for state in dp.statespace:

        idx = dp.state_to_index(state)

        # Now states will appear in correct order
        print(f"Index {idx}: {state}")
        printed += 1
        if printed >= num:
            return


def print_optimal_trajectory(dp):
    """Print the optimal action trajectory starting from zero state."""
    print("\n=== Optimal Policy Trajectory from Zero State ===")

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

    for t in range(dp.max_timesteps):
        idx = dp.state_to_index(state)

        # depending on mdptoolbox version
        if hasattr(dp, "optimal_policy"):
            action_idx = dp.optimal_policy[idx]
        else:
            action_idx = dp.optimal_policy[t, idx]

        action = dp.actions[action_idx]

        print(f"Timestep {t}:")
        print(f"  State = {state}")
        print(f"  Action idx = {action_idx},  Action = {action}")

        # Transition
        next_state = dp.state_transition(action, state)

        print(f"  → Next State = {next_state}")

        state = next_state


if __name__ == "__main__":
    dp_instance, optimal_policy = main()

    if optimal_policy is not None:
        print("\n" + "=" * 60)
        print("✓ Successfully solved MDP!")
        print("=" * 60)
        print("\nYou can now:")
        print("1. Inspect 'optimal_policy' array")
        print("2. Use dp_instance.actions to decode action indices")
        print("3. Increase bin resolution in discretize_state_space()")
        print("4. Adjust horizon in solve_finite_horizon_mdp()")

    """
    CONFIG_PATH = Path("/Users/work/PycharmProjects/Carbon-Simulator/rllib/DP/config.yaml")
    cfg = load_config(CONFIG_PATH)
    dp = DPImpl(cfg)
    #   Action: [build, green, research, move]
    # print(dp.state_transition([1,0,0,0],[0,0,0,0,(),0,0,0])) #(1.0, -1.0,  0, 1, (0,), 0, 0, 1)
    # [0,1,0,0],[0,0,0,0,(),0,0,0] #######(0.0, 0.0,  0, 0, (0,), 0, 0, 1)
    # [0,1,0,0],[0,0,0,0,(),0,1,0] #######(-1.0, 1.0,  0, 1, (0,), 1.0, 0, 1)
    # [0,0,0,1],[0,0,0,0,(),0,0,0] ####### (0.0, 0.0,  0, 1, (0,), 0, 0, 1)* 0.9875
    # [1,1,1,1],[20,-2,0,2,(1,1),0,0,0] ####### (10.5, -0.9048374180359595, 1, 5, (1, 1), 0, 0, 1)
    # [1,1,1,1],[20,-2,0,2,(1,1),0,1,0] ####### (9.5, 0.09516258196404048, 1, 6, (1, 1), 1.0, 0, 1)
    actions = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [1, 1, 1, 0],
    ]
    states=[(0, 0, 0, 0, (0, 0), 0, 0, 0)]
    states_1 = [
        (2.0, 0.0, 0, 1, (0, 0), 0, 0, 1),
        (0.0, 1.0, 0, 0, (0, 0), 0, 0, 1),
        (-1.0, 1.0, 0, 1, (1, 0), 0, 0, 1),
        (0.0, 1.0, 0, 1, (0, 0), 0, 1, 1),
        (1.0, 0.0, 0, 3, (1, 0), 0, 1, 1),
        (2.0, 0.0, 0, 1, (0, 0), 0, 0, 1),
        (1.0, 0.0, 0, 2, (1, 0), 0, 0, 1),
        (2.0, 0.0, 0, 2, (0, 0), 0, 1, 1),
        (1.0, 0.0, 0, 2, (1, 0), 0, 0, 1),
    ]
    states_2 = [
        (4.0, 0.0, 0, 2, (0, 0), 0, 0, 2),
        (2.0, 1.0, 0, 1, (0, 0), 0, 0, 2),
        (1.0, 1.0, 0, 2, (0, 1), 0, 0, 2),
        (2.0, 1.0, 0, 2, (0, 0), 0, 1, 2),
        (3.0, 0.0, 0, 4, (0, 1), 0, 1, 2),
        (4.0, 0.0, 0, 2, (0, 0), 0, 0, 2),
        (3.0, 0.0, 0, 3, (0, 1), 0, 0, 2),
        (4.0, 0.0, 0, 3, (0, 0), 0, 1, 2),
        (3.0, 0.0, 0, 3, (0, 1), 0, 0, 2),
        (2.0, 1.0, 0, 1, (0, 0), 0, 0, 2),
        (0.0, 2.0, 0, 0, (0, 0), 0, 0, 2),
        (-1.0, 2.0, 0, 1, (0, 1), 0, 0, 2),
        (-1.0, 3.0, 0, 2, (0, 0), 1.0, 0, 2),
        (0.0, 2.0, 0, 4, (0, 1), 1.0, 0, 2),
        (2.0, 1.0, 0, 1, (0, 0), 0, 0, 2),
        (1.0, 1.0, 0, 2, (0, 1), 0, 0, 2),
        (1.0, 2.0, 0, 3, (0, 0), 1.0, 0, 2),
        (1.0, 1.0, 0, 2, (0, 1), 0, 0, 2),
        (1.0, 1.0, 0, 2, (1, 0), 0, 0, 2),
        (-1.0, 2.0, 0, 1, (1, 0), 0, 0, 2),
        (-2.0, 2.0, 0, 2, (1, 1), 0, 0, 2),
        (-1.0, 2.0, 0, 2, (1, 0), 0, 1, 2),
        (0.0, 1.0, 0, 4, (1, 1), 0, 1, 2),
        (1.0, 1.0, 0, 2, (1, 0), 0, 0, 2),
        (0.0, 1.0, 0, 3, (1, 1), 0, 0, 2),
        (1.0, 1.0, 0, 3, (1, 0), 0, 1, 2),
        (0.0, 1.0, 0, 3, (1, 1), 0, 0, 2),
        (2.0, 1.0, 0, 2, (0, 0), 0, 1, 2),
        (0.0, 2.0, 0, 1, (0, 0), 0, 1, 2),
        (-1.0, 2.0, 0, 2, (0, 1), 0, 1, 2),
        (0.0, 2.0, 0, 2, (0, 0), 0, 1, 2),
        (1.0, 1.0, 0, 4, (0, 1), 0, 1, 2),
        (2.0, 1.0, 0, 2, (0, 0), 0, 1, 2),
        (1.0, 1.0, 0, 3, (0, 1), 0, 1, 2),
        (2.0, 1.0, 0, 3, (0, 0), 0, 1, 2),
        (1.0, 1.0, 0, 3, (0, 1), 0, 1, 2),
        (3.0, 0.0, 0, 4, (1, 0), 0, 1, 2),
        (1.0, 1.0, 0, 3, (1, 0), 0, 1, 2),
        (0.0, 1.0, 0, 4, (1, 1), 0, 1, 2),
        (0.0, 2.0, 0, 5, (1, 0), 1.0, 1, 2),
        (1.0, 1.0, 0, 7, (1, 1), 1.0, 1, 2),
        (3.0, 0.0, 0, 4, (1, 0), 0, 1, 2),
        (2.0, 0.0, 0, 5, (1, 1), 0, 1, 2),
        (2.0, 1.0, 0, 6, (1, 0), 1.0, 1, 2),
        (2.0, 0.0, 0, 5, (1, 1), 0, 1, 2),
        (4.0, 0.0, 0, 2, (0, 0), 0, 0, 2),
        (2.0, 1.0, 0, 1, (0, 0), 0, 0, 2),
        (1.0, 1.0, 0, 2, (0, 1), 0, 0, 2),
        (1.0, 2.0, 0, 3, (0, 0), 1.0, 0, 2),
        (2.0, 1.0, 0, 5, (0, 1), 1.0, 0, 2),
        (4.0, 0.0, 0, 2, (0, 0), 0, 0, 2),
        (3.0, 0.0, 0, 3, (0, 1), 0, 0, 2),
        (3.0, 1.0, 0, 4, (0, 0), 1.0, 0, 2),
        (3.0, 0.0, 0, 3, (0, 1), 0, 0, 2),
        (3.0, 0.0, 0, 3, (1, 0), 0, 0, 2),
        (1.0, 1.0, 0, 2, (1, 0), 0, 0, 2),
        (0.0, 1.0, 0, 3, (1, 1), 0, 0, 2),
        (1.0, 1.0, 0, 3, (1, 0), 0, 1, 2),
        (2.0, 0.0, 0, 5, (1, 1), 0, 1, 2),
        (3.0, 0.0, 0, 3, (1, 0), 0, 0, 2),
        (2.0, 0.0, 0, 4, (1, 1), 0, 0, 2),
        (3.0, 0.0, 0, 4, (1, 0), 0, 1, 2),
        (2.0, 0.0, 0, 4, (1, 1), 0, 0, 2),
        (4.0, 0.0, 0, 3, (0, 0), 0, 1, 2),
        (2.0, 1.0, 0, 2, (0, 0), 0, 1, 2),
        (1.0, 1.0, 0, 3, (0, 1), 0, 1, 2),
        (2.0, 1.0, 0, 3, (0, 0), 0, 1, 2),
        (3.0, 0.0, 0, 5, (0, 1), 0, 1, 2),
        (4.0, 0.0, 0, 3, (0, 0), 0, 1, 2),
        (3.0, 0.0, 0, 4, (0, 1), 0, 1, 2),
        (4.0, 0.0, 0, 4, (0, 0), 0, 1, 2),
        (3.0, 0.0, 0, 4, (0, 1), 0, 1, 2),
        (3.0, 0.0, 0, 3, (1, 0), 0, 0, 2),
        (1.0, 1.0, 0, 2, (1, 0), 0, 0, 2),
        (0.0, 1.0, 0, 3, (1, 1), 0, 0, 2),
        (0.0, 2.0, 0, 4, (1, 0), 1.0, 0, 2),
        (1.0, 1.0, 0, 6, (1, 1), 1.0, 0, 2),
        (3.0, 0.0, 0, 3, (1, 0), 0, 0, 2),
        (2.0, 0.0, 0, 4, (1, 1), 0, 0, 2),
        (2.0, 1.0, 0, 5, (1, 0), 1.0, 0, 2),
        (2.0, 0.0, 0, 4, (1, 1), 0, 0, 2),
    ]
    new_state=[]
    new_state_2=[]
    for action in actions:
            new_state.append( dp.state_transition(action, (0, 0, 0, 0, (0, 0), 0, 0, 0)))

    discretization_arrs= [[], [], [], [], [], [], [], []]
    for action in actions:
        for state in new_state:
            new_state_3=dp.state_transition(action, state)
            i=0
            for s in new_state_3:
                if s not in discretization_arrs[i]:
                    discretization_arrs[i].append(s)
                i+=1
    for i in range(len(discretization_arrs)):
        discretization_arrs[i].sort()
    print(discretization_arrs)
    
    results_states = [
        (4.0, 0.0, 0, 2, (0, 0), 0, 1, 2),
        (1.0, 2.5, 0, 2, (0, 0), 1.0, 1, 2),
        (1.0, 1.0, 0, 2, (0, 1), 0, 1, 2),
        (2.0, 1.0, 0, 2, (0, 0), 0, 1, 2),
        (2.0, 1.5, 0, 5, (0, 1), 1.0, 1, 2),
        (3.0, 1.5, 0, 3, (0, 0), 1.0, 1, 2),
        (3.0, 0.0, 0, 3, (0, 1), 0, 1, 2),
        (4.0, 0.0, 0, 3, (0, 0), 0, 1, 2),
        (2.0, 1.5, 0, 4, (0, 1), 1.0, 1, 2),
        (1.0, 2.0, 0, 2, (0, 0), 1.0, 0, 2),
        (-1.0, 3.0, 0, 1, (0, 0), 1.0, 1, 2),
        (-2.0, 3.0, 0, 2, (0, 1), 1.0, 0, 2),
        (-1.0, 3.0, 0, 2, (0, 0), 1.0, 0, 2),
        (0.0, 2.0, 0, 4, (0, 1), 1.0, 1, 2),
        (1.0, 2.0, 0, 2, (0, 0), 1.0, 1, 2),
        (0.0, 2.0, 0, 3, (0, 1), 1.0, 0, 2),
        (1.0, 2.0, 0, 3, (0, 0), 1.0, 0, 2),
        (0.0, 2.0, 0, 3, (0, 1), 1.0, 1, 2),
        (1.0, 1.0, 0, 2, (1, 0), 0, 1, 2),
        (-2.0, 3.0, 0, 2, (1, 0), 1.0, 1, 2),
        (-2.0, 2.0, 0, 2, (1, 1), 0, 1, 2),
        (-1.0, 2.0, 0, 2, (1, 0), 0, 1, 2),
        (-1.0, 2.0, 0, 5, (1, 1), 1.0, 1, 2),
        (0.0, 2.0, 0, 3, (1, 0), 1.0, 1, 2),
        (0.0, 1.0, 0, 3, (1, 1), 0, 1, 2),
        (1.0, 1.0, 0, 3, (1, 0), 0, 1, 2),
        (-1.0, 2.0, 0, 4, (1, 1), 1.0, 1, 2),
        (2.0, 1.0, 0, 2, (0, 0), 0, 1, 2),
        (-1.0, 3.0, 0, 2, (0, 0), 1.0, 1, 2),
        (-1.0, 2.0, 0, 2, (0, 1), 0, 1, 2),
        (0.0, 2.0, 0, 2, (0, 0), 0, 1, 2),
        (0.0, 2.0, 0, 5, (0, 1), 1.0, 1, 2),
        (1.0, 2.0, 0, 3, (0, 0), 1.0, 1, 2),
        (1.0, 1.0, 0, 3, (0, 1), 0, 1, 2),
        (2.0, 1.0, 0, 3, (0, 0), 0, 1, 2),
        (0.0, 2.0, 0, 4, (0, 1), 1.0, 1, 2),
        (2.0, 1.0, 0, 5, (1, 0), 1.0, 1, 2),
        (0.0, 2.5, 0, 4, (1, 0), 1.0, 1, 2),
        (-1.0, 2.0, 0, 5, (1, 1), 1.0, 1, 2),
        (0.0, 2.0, 0, 5, (1, 0), 1.0, 1, 2),
        (1.0, 1.5, 0, 7, (1, 1), 1.0, 1, 2),
        (2.0, 1.5, 0, 5, (1, 0), 1.0, 1, 2),
        (1.0, 1.0, 0, 6, (1, 1), 1.0, 1, 2),
        (2.0, 1.0, 0, 6, (1, 0), 1.0, 1, 2),
        (1.0, 1.5, 0, 6, (1, 1), 1.0, 1, 2),
        (3.0, 1.0, 0, 3, (0, 0), 1.0, 0, 2),
        (1.0, 2.5, 0, 2, (0, 0), 1.0, 1, 2),
        (0.0, 2.0, 0, 3, (0, 1), 1.0, 0, 2),
        (1.0, 2.0, 0, 3, (0, 0), 1.0, 0, 2),
        (2.0, 1.5, 0, 5, (0, 1), 1.0, 1, 2),
        (3.0, 1.5, 0, 3, (0, 0), 1.0, 1, 2),
        (2.0, 1.0, 0, 4, (0, 1), 1.0, 0, 2),
        (3.0, 1.0, 0, 4, (0, 0), 1.0, 0, 2),
        (2.0, 1.5, 0, 4, (0, 1), 1.0, 1, 2),
        (3.0, 0.0, 0, 3, (1, 0), 0, 1, 2),
        (0.0, 2.5, 0, 3, (1, 0), 1.0, 1, 2),
        (0.0, 1.0, 0, 3, (1, 1), 0, 1, 2),
        (1.0, 1.0, 0, 3, (1, 0), 0, 1, 2),
        (1.0, 1.5, 0, 6, (1, 1), 1.0, 1, 2),
        (2.0, 1.5, 0, 4, (1, 0), 1.0, 1, 2),
        (2.0, 0.0, 0, 4, (1, 1), 0, 1, 2),
        (3.0, 0.0, 0, 4, (1, 0), 0, 1, 2),
        (1.0, 1.5, 0, 5, (1, 1), 1.0, 1, 2),
        (4.0, 0.0, 0, 3, (0, 0), 0, 1, 2),
        (1.0, 2.5, 0, 3, (0, 0), 1.0, 1, 2),
        (1.0, 1.0, 0, 3, (0, 1), 0, 1, 2),
        (2.0, 1.0, 0, 3, (0, 0), 0, 1, 2),
        (2.0, 1.5, 0, 6, (0, 1), 1.0, 1, 2),
        (3.0, 1.5, 0, 4, (0, 0), 1.0, 1, 2),
        (3.0, 0.0, 0, 4, (0, 1), 0, 1, 2),
        (4.0, 0.0, 0, 4, (0, 0), 0, 1, 2),
        (2.0, 1.5, 0, 5, (0, 1), 1.0, 1, 2),
        (2.0, 1.0, 0, 4, (1, 0), 1.0, 0, 2),
        (0.0, 2.5, 0, 3, (1, 0), 1.0, 1, 2),
        (-1.0, 2.0, 0, 4, (1, 1), 1.0, 0, 2),
        (0.0, 2.0, 0, 4, (1, 0), 1.0, 0, 2),
        (1.0, 1.5, 0, 6, (1, 1), 1.0, 1, 2),
        (2.0, 1.5, 0, 4, (1, 0), 1.0, 1, 2),
        (1.0, 1.0, 0, 5, (1, 1), 1.0, 0, 2),
        (2.0, 1.0, 0, 5, (1, 0), 1.0, 0, 2),
        (1.0, 1.5, 0, 5, (1, 1), 1.0, 1, 2)
    ]
    for res in states:
        print("Reward:", dp.reward(res))"""
