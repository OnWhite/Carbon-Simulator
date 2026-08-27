"""Equality tests for the first two links of the verification chain.

The old pipeline compared *learned policies* at every link, so a disagreement
anywhere could be attributed to anything: a learning failure, a horizon
mismatch, or a transition-function bug all looked the same. These tests turn
the first two links into assertions about code.

Run:
    PYTHONPATH=. python -m unittest rllib.DP.test_parity -v
    PYTHONPATH=. python -m unittest rllib.DP.test_parity.TestModelStructure -v

The slow test (full reachable-set parity) is skipped unless PARITY_FULL=1, so
the suite stays usable as a pre-commit check.
"""

from __future__ import annotations

import math
import os
import unittest
from pathlib import Path

from rllib.DP.DynamicProgram import Action, DPImpl, State, load_config
from rllib.DP.exact_dp import ALL_ACTIONS, ExactDP, start_state, state_key
from rllib.RL.CarbonEnv import CarbonEnv

CONFIG = Path(__file__).resolve().parent / "config.yaml"

# Horizon for the fast tests. The exact reachable set is ~540*T^2 states, so
# T=6 keeps the whole suite under a couple of seconds while still spanning
# several years at period=1 (yearly allocation, expiry and penalty all fire).
FAST_T = 6
FULL_T = int(os.environ.get("PARITY_T", "10"))

TOL = 1e-9


def make(horizon: int):
    cfg = load_config(CONFIG)
    dp = DPImpl(cfg)
    gamma = float(cfg.get("agent_policy", {}).get("gamma", 0.998))
    ex = ExactDP(dp, horizon=horizon, gamma=gamma)
    env = CarbonEnv({"config_path": str(CONFIG), "horizon": horizon})
    return ex, env


class TestModelStructure(unittest.TestCase):
    """Structural invariants of the exact MDP. All fast."""

    @classmethod
    def setUpClass(cls):
        cls.ex, cls.env = make(FAST_T)
        cls.ex.enumerate_reachable()
        cls.ex.solve()

    def test_branch_probabilities_sum_to_one(self):
        """Every (state, action) defines a distribution, not a partial one.

        The old builder wrote P[a, s, succ] = 1 - failrate and then
        P[a, s, fail] = failrate. When both landed in the same discretised bin
        the row summed to failrate and the solver accepted it silently.
        """
        checked = 0
        for layer in self.ex.layers[:-1]:
            for s in layer.states.values():
                for a in ALL_ACTIONS:
                    total = sum(p for p, _ in self.ex.branches(a, s))
                    self.assertAlmostEqual(total, 1.0, delta=1e-12)
                    checked += 1
        self.assertGreater(checked, 0)

    def test_terminal_value_is_exactly_zero(self):
        """V_T = 0, as the Bellman equation in the thesis specifies.

        Infinite-horizon policy iteration on a state space where `timestep` was
        a clipped feature gave terminal states a self-loop worth roughly
        1/(1 - 0.998) periods of reward instead.
        """
        terminal = self.ex.values[self.ex.horizon]
        self.assertTrue(terminal)
        for v in terminal.values():
            self.assertEqual(v, 0.0)

    def test_state_keys_are_injective_on_the_reachable_set(self):
        """Two distinct states never collapse onto one key.

        This is the property the bin grid did not have: research_count and
        research_yearly each had a single bin, so every post-research state
        aliased onto research_count = 0.
        """
        for layer in self.ex.layers:
            for k, s in layer.states.items():
                self.assertEqual(k, state_key(s))
            self.assertEqual(len(layer.states), len(set(layer.states)))

    def test_research_actually_moves_the_state(self):
        """Regression test for the collapsed research bins.

        If research is representable at all, some reachable state must have
        research_count > 0. Under the old grid this was impossible by
        construction, which is what produced the finding that research does not
        pay -- an artefact of binning, not of the model.
        """
        counts = {
            s.research_count
            for layer in self.ex.layers
            for s in layer.states.values()
        }
        self.assertTrue(any(c > 0 for c in counts),
                        f"research_count never leaves 0; observed {sorted(counts)}")

    def test_horizon_is_the_number_of_decisions(self):
        """One decision layer per timestep, and the env agrees.

        `terminated = timestep >= max_timesteps - 1` used to end the episode
        after a single decision at episode_length = 2.
        """
        self.assertEqual(self.env.horizon, self.ex.horizon)
        self.assertEqual(len(self.ex.layers), self.ex.horizon + 1)

        steps = 0
        self.env.reset(seed=0)
        done = False
        while not done:
            _, _, done, trunc, _ = self.env.step(0)
            steps += 1
            self.assertLessEqual(steps, self.ex.horizon + 1)
        self.assertEqual(steps, self.ex.horizon)


class TestTransitionParity(unittest.TestCase):
    """V1 == V2: the dynamic program and the gym env are the same model.

    No test of this kind existed anywhere in the repo. It is worth more than
    the whole policy-comparison apparatus, because it makes "the models agree"
    a statement about two functions rather than about two learned policies.
    """

    @classmethod
    def setUpClass(cls):
        cls.ex, cls.env = make(FAST_T)
        cls.ex.enumerate_reachable()

    def _compare(self, layers):
        n = 0
        for layer in layers[:-1]:
            for s in layer.states.values():
                for a_idx, a in enumerate(ALL_ACTIONS):
                    for xi in (0, 1):        # research success
                        for pi in (0, 1):    # permit granted
                            dp_next = self.ex.dp.state_transition(
                                a, s, research_success=xi, permit_granted=pi)
                            env_next, env_rew = self.env.deterministic_step(
                                a_idx, s, research_success=xi, permit_granted=pi)

                            self.assertEqual(
                                state_key(dp_next), state_key(env_next),
                                msg=(f"transition mismatch\n  state  {s}\n"
                                     f"  action {a} xi={xi} permit={pi}\n"
                                     f"  dp   -> {dp_next}\n  env  -> {env_next}"))

                            dp_rew = self.ex.dp.reward(dp_next)
                            self.assertLess(
                                abs(dp_rew - env_rew), TOL,
                                msg=f"reward mismatch at {s} / {a}: "
                                    f"{dp_rew} vs {env_rew}")
                            n += 1
        return n

    def test_parity_on_reachable_set(self):
        n = self._compare(self.ex.layers)
        self.assertGreater(n, 0)
        print(f"\n  transition parity verified on {n:,} (state, action, shock) triples")

    @unittest.skipUnless(os.environ.get("PARITY_FULL") == "1",
                         "set PARITY_FULL=1 to run the full-horizon sweep")
    def test_parity_on_full_horizon(self):
        ex, env = make(FULL_T)
        ex.enumerate_reachable()
        self.ex, self.env = ex, env
        n = self._compare(ex.layers)
        print(f"\n  transition parity verified on {n:,} triples at T={FULL_T}")


class TestResearchCostsArePaidOnFailure(unittest.TestCase):
    """The failure branch must cost what the real environment charges.

    In Produce_and_Invest.component_step the labor and coin lines for a
    research action sit OUTSIDE the success check -- only the pipeline bit
    Research_history[0] = 1 is conditional. The old DP modelled failure as
    Action(research=0), which refunded both costs and so overvalued research.
    """

    def setUp(self):
        self.dp = DPImpl(load_config(CONFIG))
        self.s0 = start_state(self.dp)
        self.act = Action(build=0, green=0, research=1, move=0)

    def test_failure_costs_the_same_as_success(self):
        succ = self.dp.state_transition(self.act, self.s0,
                                        research_success=1, permit_granted=0)
        fail = self.dp.state_transition(self.act, self.s0,
                                        research_success=0, permit_granted=0)

        self.assertAlmostEqual(succ.coin, fail.coin, delta=TOL,
                               msg="a failed research attempt was refunded its coin cost")
        self.assertAlmostEqual(succ.labor, fail.labor, delta=TOL,
                               msg="a failed research attempt was refunded its labor cost")

    def test_failure_does_not_enter_the_pipeline(self):
        succ = self.dp.state_transition(self.act, self.s0,
                                        research_success=1, permit_granted=0)
        fail = self.dp.state_transition(self.act, self.s0,
                                        research_success=0, permit_granted=0)
        self.assertEqual(succ.research_history[0], 1)
        self.assertEqual(fail.research_history[0], 0)

    def test_research_cost_matches_the_component_formula(self):
        """payment / (2 * Research_ability) and labor * Research_ability."""
        after = self.dp.state_transition(self.act, self.s0,
                                         research_success=1, permit_granted=0)
        expected_coin = self.s0.coin - self.dp.payment / (2 * self.dp.research_ability)
        expected_labor = self.s0.labor + self.dp.l_research * self.dp.research_ability
        self.assertAlmostEqual(after.coin, expected_coin, delta=TOL)
        self.assertAlmostEqual(after.labor, expected_labor, delta=TOL)


class TestTransitionIsPure(unittest.TestCase):
    """No hidden randomness left in state_transition.

    It used to call random.random() to decide whether a move yielded a permit,
    including while the transition matrix was being assembled -- so the solver
    saw one unseeded coin flip per (state, action) pair, baked in with
    probability 1.0.
    """

    def setUp(self):
        self.dp = DPImpl(load_config(CONFIG))
        self.s0 = start_state(self.dp)

    def test_repeated_calls_agree(self):
        move = Action(build=0, green=0, research=0, move=1)
        for permit in (0, 1):
            results = {
                state_key(self.dp.state_transition(
                    move, self.s0, research_success=1, permit_granted=permit))
                for _ in range(200)
            }
            self.assertEqual(len(results), 1,
                             f"state_transition is not deterministic (permit={permit})")

    def test_permit_argument_is_what_grants_the_certificate(self):
        move = Action(build=0, green=0, research=0, move=1)
        got = self.dp.state_transition(move, self.s0,
                                       research_success=1, permit_granted=1)
        missed = self.dp.state_transition(move, self.s0,
                                          research_success=1, permit_granted=0)
        self.assertEqual(got.on_certificate, 1)
        self.assertEqual(missed.on_certificate, 0)

    def test_permit_probability_is_derived_not_guessed(self):
        self.assertAlmostEqual(
            self.dp.p_permit,
            self.dp.max_greenbudget / self.dp.worldsize,
            delta=TOL)
        self.assertAlmostEqual(
            self.dp.p_research_success, 1.0 - self.dp.failrate, delta=TOL)


class TestOptimalPolicyMeasurement(unittest.TestCase):
    """The statistics that replace "200 of 4032 states"."""

    @classmethod
    def setUpClass(cls):
        cls.ex, _ = make(FAST_T)
        cls.ex.enumerate_reachable()
        cls.ex.solve()

    def test_optimal_policy_has_zero_regret_against_itself(self):
        """Sanity check on the regret machinery before it judges a network."""
        def optimal(s: State) -> int:
            return self.ex.policy[s.timestep][state_key(s)]

        r = self.ex.regret(optimal)
        self.assertLess(abs(r["abs_regret"]), 1e-9, r)
        self.assertLess(abs(r["rel_regret"]), 1e-9, r)

    def test_agreement_with_itself_is_one(self):
        def optimal(s: State) -> int:
            return self.ex.policy[s.timestep][state_key(s)]

        a = self.ex.agreement(optimal)
        self.assertAlmostEqual(a["agree_unweighted"], 1.0, delta=1e-12)
        self.assertAlmostEqual(a["agree_occupancy_weighted"], 1.0, delta=1e-12)
        self.assertEqual(a["n_disagreements"], 0.0)

    def test_occupancy_is_a_distribution_per_timestep(self):
        occ = self.ex.occupancy(policy=self.ex.policy)
        for t, layer_occ in enumerate(occ):
            total = sum(layer_occ.values())
            self.assertAlmostEqual(total, 1.0, delta=1e-9,
                                   msg=f"occupancy at t={t} sums to {total}")

    def test_a_worse_policy_has_positive_regret(self):
        """Guards against a regret function that always returns zero."""
        def always_idle(s: State) -> int:
            return 0  # Action(0, 0, 0, 0)

        r = self.ex.regret(always_idle)
        self.assertGreater(r["abs_regret"], 0.0, r)
        self.assertTrue(math.isfinite(r["rel_regret"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
