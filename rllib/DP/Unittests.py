import unittest
from pathlib import Path
from typing import Dict, Any, List, Tuple

from rllib.DP.DynamicProgram import DPImpl, Action, State


class TestDP(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = {'env': {'n_agents': 1, 'world_size': [18, 18], 'episode_length': 2, 'period': 1,
                       'flatten_observations': True, 'flatten_masks': True, 'scenario_name': 'Carbon/Carbon_env',
                       'components': [{'CarbonRedistribution': {'planner_mode': 'active', 'fixed_punishment': 5,
                                                                'total_idx': 2, 'max_year_percent': 25,
                                                                'years_predefined': 'None',
                                                                'agents_predefined': 'None'}}, {
                                          'Carbon_component': {'payment': 2, 'require_Carbon_idx': 1,
                                                               'lowest_rate': 0.02, 'research_setting': ['e^-', 0.2],
                                                               'random_fails': 0.1, 'delay': 1, 'forget': 1}}, {
                                          'Carbon_auction': {'max_bid_ask': 20, 'max_num_orders': 5,
                                                             'order_duration': 10}},
                                      {'Gather': {'collect_labor': 1, 'collect_cost_coin': 1}}],
                       'dense_log_frequency': 1, 'isoelastic_eta': 0.23, 'energy_cost': 0.1, 'total_idx': None,
                       'energy_warmup_constant': 1, 'energy_warmup_method': 'auto', 'starting_agent_coin': 0,
                       'mobile_coefficient': 0.2},
               }
        self.dp = DPImpl(cfg)

    # research_history has length max(delay, forget) + 1 == 2 for this config
    # (Produce_and_Invest allocates [0] * (max(delay, forget) + 1) and indexes
    # [delay] unguarded). These expectations carried length-1 tuples from
    # before that fix, so all five were red on arrival.

    def test_build(self):
        self.assertEqual(self.dp.state_transition(Action(1, 0, 0, 0), State(0, 0, 0, 0, 0, (), 0, 0, 0)),
                         State(2.0, 0.0, 0, 0, 1, (0, 0), 0, 0, 1))

    def test_green(self):
        self.assertEqual(self.dp.state_transition(Action(0, 1, 0, 0), State(0, 0, 0, 0, 0, (), 0, 0, 0)),
                         State(0.0, 1.0, 0, 0, 0, (0, 0), 0, 0, 1))

    def test_green2(self):
        # when on certificate
        self.assertEqual(self.dp.state_transition(Action(0, 1, 0, 0), State(0, 0, 0, 0, 0, (), 0, 1, 0)),
                         State(-1.0, 2.0, 0, 0, 1, (0, 0), 1.0, 0, 1))

    def test_move(self):
        self.assertEqual(self.dp.state_transition(Action(0, 0, 0, 1), State(0, 0, 0, 0, 0, (), 0, 0, 0)),
                         State(0.0, 1.0, 0, 0, 1, (0, 0), 0, 0, 1))

    def test_all_at_once(self):
        """Certificates EXPIRE at the year boundary (the decided regime).

        The expected coin of 11.906 was written against the cumulative regime,
        where `carbon += start_idx` let the incoming carbon of -2 survive into
        the yearly settlement and trigger the punishment. Expiry is now
        consistent across all three tiers -- DPImpl.state_transition
        (`carbon = start_idx`) and both branches of
        CarbonRedistribution.component_step, which assign rather than
        accumulate -- so the -2 is discarded, no punishment fires, and coin
        stays at 21.0.

        A negative balance is still punished; that happens one step earlier,
        at timestep % period == 0, before the new allocation lands.
        """
        self.assertEqual(self.dp.state_transition(Action(1, 1, 1, 1), State(20, -2, 0, 0, 2, (1, 1), 0, 0, 0)),
                         State(21.0, 0.18126924692201818, 1, 1, 5, (1, 1), 0, 0, 1))


def load_config(path: Path) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    unittest.main()
