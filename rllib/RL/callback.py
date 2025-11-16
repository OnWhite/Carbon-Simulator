from typing import Optional

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
import pprint


class ResultInfoMetricsCallback(DefaultCallbacks):

    def __init__(self):
        super().__init__()

    def on_episode_step(
            self, *, worker, base_env, policies, episode: Episode, env_index: Optional[int] = None,
            **kwargs
    ):
        # tracks metrics in STEP_METRICS every step per worker for all agents
        infos = episode._last_infos
        if not infos:
            return
        agent_data = infos.get('agent0')
        state = agent_data.get('state')
        action = agent_data.get('action')
        wid = worker.worker_index

        if state:
            for attr, value in vars(state).items():
                episode.hist_data.setdefault(f"worker_{wid}/state_{attr}_ts", []).append(value)
        if action:
            for attr, value in vars(action).items():
                episode.hist_data.setdefault(f"worker_{wid}/action_{attr}_ts", []).append(value)

    def on_episode_end(
            self, *, worker, base_env, policies, episode: Episode, env_index: Optional[int] = None,
            **kwargs    ):
        infos = episode._last_infos
        if not infos:
            return
        wid = worker.worker_index
        agent_data = infos.get('agent0')
        action = agent_data.get('action')
        if action:
            for attr, value in vars(action).items():
                episode.custom_metrics[f"worker_{wid}/action_{attr}_ts"]=value
