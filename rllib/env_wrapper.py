import os
import pickle
import random
import warnings
import sys
from pprint import pformat

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")
    )
)

import numpy as np
from Carbon_simulator import foundation
from gymnasium import spaces
from gymnasium.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import logging

_BIG_NUMBER = 1e20


def recursive_list_to_np_array(d):
    if isinstance(d, dict):
        new_d = {}
        for k, v in d.items():
            if isinstance(v, list):
                new_d[k] = np.array(v)
            elif isinstance(v, dict):
                new_d[k] = recursive_list_to_np_array(v)
            elif isinstance(v, (float, int, np.floating, np.integer)):
                new_d[k] = np.array([v])
            elif isinstance(v, np.ndarray):
                new_d[k] = v
            else:
                raise AssertionError
        return new_d
    raise AssertionError

def pretty_print(dictionary):
    for key in dictionary:
        print("{:15s}: {}".format(key, dictionary[key].shape))
    print("\n")


class RLlibEnvWrapper(MultiAgentEnv):
    """
    Environment wrapper for RLlib. It sub-classes MultiAgentEnv.
    This wrapper adds the action and observation space to the environment,
    and adapts the reset and step functions to run with RLlib.
    """

    def __init__(self, env_config, verbose=False):
        super().__init__()
        self.env_config_dict = env_config["env_config_dict"]

        import signal
        import faulthandler
        faulthandler.register(signal.SIGUSR1)

        # Adding env id in the case of multiple environments
        if hasattr(env_config, "worker_index"):
            self.env_id = (
                env_config["num_envs_per_worker"] * (env_config.worker_index - 1)
            ) + env_config.vector_index
        else:
            self.env_id = None

        self.env = foundation.make_env_instance(**self.env_config_dict)
        self.verbose = verbose
        self.sample_agent_idx = str(self.env.all_agents[0].idx)

        obs = self.env.reset()

        self.observation_space = self._dict_to_spaces_dict(obs["0"])
        self.observation_space_pl = self._dict_to_spaces_dict(obs["p"])

        self._agent_ids = list(obs.keys())

        if self.env.world.agents[0].multi_action_mode:
            self.action_space = spaces.MultiDiscrete(
                self.env.get_agent(self.sample_agent_idx).action_spaces
            )
            '''self.action_space.dtype = np.int64
            self.action_space.nvec = self.action_space.nvec.astype(np.int64)'''

        else:
            self.action_space = spaces.Discrete(
                self.env.get_agent(self.sample_agent_idx).action_spaces
            )
            '''self.action_space.dtype = np.int64'''

        if self.env.world.planner.multi_action_mode:
            self.action_space_pl = spaces.MultiDiscrete(
                self.env.get_agent("p").action_spaces
            )
            '''self.action_space_pl.dtype = np.int64
            self.action_space_pl.nvec = self.action_space_pl.nvec.astype(np.int64)'''

        else:
            self.action_space_pl = spaces.Discrete(
                self.env.get_agent("p").action_spaces
            )
            '''self.action_space_pl.dtype = np.int64'''

        self._seed = None
        if True:

            # Configure a new logger for this specific purpose
            new_logger = logging.getLogger("EnvWrapperLogger")
            new_logger.setLevel(logging.INFO)

            # Create a file handler for the new log file
            file_handler = logging.FileHandler("env_wrapper_new.log")
            file_handler.setLevel(logging.INFO)

            # Add a formatter to the file handler
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            # Add the file handler to the logger
            new_logger.addHandler(file_handler)

            # Log the information
            new_logger.info("[EnvWrapper] Spaces")
            new_logger.info("[EnvWrapper] Obs (a)")
            pretty_print(self.observation_space)
            new_logger.info("[EnvWrapper] Obs (p)")
            new_logger.info("[EnvWrapper] Obs (p) %s", pformat(self.observation_space_pl))
            new_logger.info("[EnvWrapper] Action (a) %s", pformat(self.action_space))
            new_logger.info("[EnvWrapper] Action (p) %s", pformat(self.action_space_pl))

        self._spaces_in_preferred_format = True

    def _dict_to_spaces_dict(self, obs):
        dict_of_spaces = {}
        for k, v in obs.items():

            # list of lists are listified np arrays
            _v = v
            if isinstance(v, list):
                _v = np.array(v)
            elif isinstance(v, (int, float, np.floating, np.integer)):
                _v = np.array([v])

            # assign Space
            if isinstance(_v, np.ndarray):
                if np.issubdtype(_v.dtype, np.integer):
                    x_min, x_max = np.iinfo(_v.dtype).min, np.iinfo(_v.dtype).max
                elif np.issubdtype(_v.dtype, np.floating):
                    x_min, x_max = -1e10, 1e10  # conservative float bounds
                else:
                    raise TypeError(f"Unsupported dtype: {_v.dtype}")

                # Warnings for extreme values
                if np.max(_v) > x_max:
                    warnings.warn("Input is too large!")
                if np.min(_v) < x_min:
                    warnings.warn("Input is too small!")
                box = spaces.Box(low=x_min, high=x_max, shape=_v.shape, dtype=_v.dtype)
                low_high_valid = (box.low < 0).all() and (box.high > 0).all()

                # This loop avoids issues with overflow to make sure low/high are good.
                while not low_high_valid:
                    x_min = x_min // 2
                    x_max = x_max // 2
                    box = spaces.Box(low=x_min, high=x_max, shape=_v.shape, dtype=_v.dtype)
                    low_high_valid = (box.low < 0).all() and (box.high > 0).all()

                dict_of_spaces[k] = box

            elif isinstance(_v, dict):
                dict_of_spaces[k] = self._dict_to_spaces_dict(_v)
            else:
                raise TypeError
        return spaces.Dict(dict_of_spaces)

    @property
    def pickle_file(self):
        if self.env_id is None:
            return "game_object.pkl"
        return "game_object_{:03d}.pkl".format(self.env_id)

    def save_game_object(self, save_dir):
        assert os.path.isdir(save_dir)
        path = os.path.join(save_dir, self.pickle_file)
        with open(path, "wb") as F:
            pickle.dump(self.env, F)

    def load_game_object(self, save_dir):
        assert os.path.isdir(save_dir)
        path = os.path.join(save_dir, self.pickle_file)
        with open(path, "rb") as F:
            self.env = pickle.load(F)

    @property
    def n_agents(self):
        return self.env.n_agents

    @property
    def summary(self):
        last_completion_metrics = self.env.previous_episode_metrics
        if last_completion_metrics is None:
            return {}
        last_completion_metrics["completions"] = int(self.env._completions)
        return last_completion_metrics

    def get_seed(self):
        return int(self._seed)

    def seed(self, seed):
        # Using the seeding utility from OpenAI Gym
        # https://github.com/openai/gym/blob/master/gym/utils/seeding.py
        _, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as an uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31

        if self.verbose:
            print(
                "[EnvWrapper] twisting seed {} -> {} -> {} (final)".format(
                    seed, seed1, seed2
                )
            )

        seed = int(seed2)
        np.random.seed(seed2)
        random.seed(seed2)
        self._seed = seed2

    def reset(self, seed=None, options=None, *args, **kwargs):
        # Use old `seed()` method.
        if seed is not None:
            self.env.seed(seed)
        # Options are ignored
        obs = self.env.reset(*args, **kwargs)
        infos = {k: {} for k in obs.keys()}
        return recursive_list_to_np_array(obs), infos

    def step(self, action_dict):
        '''assert action_dict.keys(), action_dict
        assert isinstance(action_dict[self.sample_agent_idx], int), action_dict'''
        obs, rewards, terminateds, infos = self.env.step(action_dict)
        assert isinstance(obs[self.sample_agent_idx]["action_mask"], np.ndarray)

        assert not (np.isinf([float(rewards[i]) for i in rewards]).any() and np.isnan([float(rewards[i]) for i in rewards]).any()),\
            (rewards, np.isnan([float(rewards[i]) for i in rewards]))

        # Truncated should always be False by default.
        truncateds = {k: False for k in terminateds.keys()}

        return recursive_list_to_np_array(obs), rewards, terminateds, truncateds, infos

