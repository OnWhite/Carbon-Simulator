from pprint import pprint

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode import Episode


class InfoMetricsCallback(DefaultCallbacks):
    """
    Collects custom metrics from the `info` dict returned by the env.
    – Step metrics: aggregated (avg, median, total) over *all* agents on the worker.
    – Final metrics: last-step values for one tracked agent (default '0').
    """

    STEP_METRICS = {
        # name               extractor – receives the agent_info dict
        "Research_count_1": lambda info: info.get("Research_count", [0, 0])[1],
        "Manufacture_volume": lambda info: info.get("Manufacture_volume"),
        "Carbon_idx": lambda info: info.get("inventory", {}).get("Carbon_idx"),
        # settlement_idx only exists in the special "p" info-dict
        "settlement_idx": lambda info: (
            None if "settlement_idx" not in info
            else float(np.sum(info["settlement_idx"]))
        ),
    }

    FINAL_METRICS = {
        "Coin": lambda info: info.get("inventory", {}).get("Coin"),
        "Labor": lambda info: info.get("endogenous", {}).get("Labor"),
        "Carbon_project": lambda info: info.get("inventory", {}).get("Carbon_project"),
        "Carbon_emission": lambda info: info.get("endogenous", {}).get("Carbon_emission"),
    }

    def __init__(self, worker_id: int = 1):
        super().__init__()
        self.worker_id = worker_id

    def on_episode_step(
            self, *, worker, base_env, policies, episode: Episode, **kwargs
    ):
        # tracs metrics in Step_Metrics every step per worker for all agents
        infos = episode._last_infos
        if not infos:
            return

        wid = worker.worker_index
        pprint(infos.items)
        for agent_id, agent_info in infos.items():
            if agent_id == 'p':
                print(agent_id+" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                pprint(agent_info)
                print(" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                mobile_idx_list = agent_info.get('mobile_idx', [])
                print(f"Full mobile_idx list: {mobile_idx_list}")
                for i in range (0,agent_info.get('mobile_idx', []).length-1):
                    key = f"worker_{wid}/agent_{i}/Certificates_Allocated"
                    episode.user_data.setdefault(key, []).append(mobile_idx_list[i])
                continue
            if not isinstance(agent_info, dict):
                continue
            for name, fn in self.STEP_METRICS.items():
                value = fn(agent_info)
                if value is None:
                    continue
                key = f"worker_{wid}/agent_{agent_id}/{name}"
                episode.user_data.setdefault(key, []).append(value)
    def on_episode_end(
            self, *, worker, base_env, policies, episode: Episode, **kwargs
    ):
        wid = worker.worker_index

        # ---- step metrics: avg / median / total -----------------
        curr_base=""
        arr= []
        arr2 = []
        for key, series in episode.user_data.items():
            if not key.startswith(f"worker_{wid}/") or not series:
                continue

            base = key.split("/", 2)[2] # drop "worker_X/agent_Y/"
            if base != curr_base:
                if base!= "":
                    episode.custom_metrics[f"worker_{wid}/Med_{base}"] = float(np.median(arr))
                    episode.custom_metrics[f"worker_{wid}/Avg_{base}"] = float(np.mean(arr2))
                curr_base = base
            series = np.asarray(series, dtype=float)
            arr.append(float(np.median(series)))
            arr2.append(float(np.avg(series)))

        if wid<=self.worker_id:
            print("--------------------------------------------------------1")
            print("in episode. wid:", wid, "user_data:", episode.user_data)
            for key, series in episode.user_data.items():
                if not key.startswith(f"worker_{wid}/") or not series:
                    continue
                agent, name = key.split("/", 2)[1], key.split("/", 2)[2]
                series = np.asarray(series, dtype=float)
                episode.custom_metrics[f"worker_{wid}/agent_{agent}/Raw_{name}"] = series.tolist()
                episode.custom_metrics[f"worker_{wid}/agent_{agent}/Med_{name}"] = float(np.median(series))

        # ---- final metrics for the tracked worker showing all agents ----------------
        if wid<=self.worker_id:
            print("--------------------------------------------------------2")
            for k, v in episode._last_infos.items():
                if k != 'p':
                    for name, fn in self.FINAL_METRICS.items():
                        val = fn(v)
                        if val is not None:
                            # name pattern: <AgentID>_<Metric>, e.g. 0_Coin
                            episode.custom_metrics[f"worker_{wid}/agent_{k}/{name}"] = float(val)

        for name, fn in self.FINAL_METRICS.items():
            metric = []
            for k, v in episode._last_infos.items():
                if k != 'p':
                    val = fn(v)
                    if val is not None:
                        metric.append(float(val))

            episode.custom_metrics[f"worker_{wid}/Tot_{name}"] = float(np.sum(metric))
            episode.custom_metrics[f"worker_{wid}/Avg_{name}"] = float(np.mean(metric))
            episode.custom_metrics[f"worker_{wid}/Med_{name}"] = float(np.median(metric))
