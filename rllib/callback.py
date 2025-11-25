from typing import Optional

import numpy as np
import wandb
import sys
import atexit
import cProfile
import logging
import socket
from typing import Optional
import os
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode import Episode

PROFILE_DIR = os.environ.get("PROFILE_DIR", "/nas/ucb/sophialudewig/rllib_profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)
logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(message)s")
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)


class ProfilingCallbacks(DefaultCallbacks):
    SNAPSHOT_EVERY = 10  # episodes

    def _ensure_profiler(self, worker):
        if getattr(worker, "_profiler", None) is not None:
            return
        wid, pid, host = worker.worker_index, os.getpid(), socket.gethostname()
        base = f"worker_{wid}_{pid}_{host}"
        prof_path = os.path.join(PROFILE_DIR, f"{base}.prof")
        log_path = os.path.join(PROFILE_DIR, f"{base}.log")
        open(os.path.join(PROFILE_DIR, f"{base}.alive"), "a").close()

        lg = logging.getLogger(base)
        if not any(isinstance(h, logging.FileHandler) for h in lg.handlers):
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            lg.addHandler(fh)
            lg.setLevel(logging.INFO)
            lg.propagate = False

        profiler = cProfile.Profile()
        profiler.enable()
        worker._profiler = profiler
        worker._profiler_path = prof_path
        worker._profiler_logger = lg
        worker._profiler_inited = True
        lg.info(f"Profiling started wid={wid} pid={pid} host={host} -> {prof_path}")

        def _dump_on_exit():
            try:
                profiler.disable()
                profiler.dump_stats(prof_path)
                lg.info("Profile saved on exit")
            except Exception as e:
                lg.exception(f"Exit dump failed: {e}")

        atexit.register(_dump_on_exit)

    # Try to init as early as possible, but always guard.
    def on_worker_init(self, *, worker, **kwargs):
        self._ensure_profiler(worker)

    def on_episode_start(self, *, worker, **kwargs):
        self._ensure_profiler(worker)

    def on_episode_end(self, *, worker, episode, **kwargs):
        self._ensure_profiler(worker)
        # breadcrumb so you can see this on the driver
        episode.custom_metrics[f"profiler_inited/w{worker.worker_index}"] = \
            1.0 if getattr(worker, "_profiler_inited", False) else 0.0
        # periodic snapshot
        if episode.episode_id % self.SNAPSHOT_EVERY == 0:
            try:
                worker._profiler.disable()
                worker._profiler.dump_stats(worker._profiler_path)
                worker._profiler.enable()
                worker._profiler_logger.info(f"Snapshot at episode {episode.episode_id}")
            except Exception as e:
                getattr(worker, "_profiler_logger", logging.getLogger(__name__)).exception(
                    f"Snapshot failed: {e}"
                )

    # Extra safety: fires on sampling/learning paths even if no episodes finish.
    def on_sample_end(self, *, worker, samples, **kwargs):
        self._ensure_profiler(worker)


def get_gini(endowments):
    n_agents = len(endowments)

    if n_agents < 30:  # Slower. Accurate for all n.
        diff_ij = np.abs(
            endowments.reshape((n_agents, 1)) - endowments.reshape((1, n_agents))
        )
        diff = np.sum(diff_ij)
        norm = 2 * n_agents * endowments.sum(axis=0)
        unscaled_gini = diff / (norm + 1e-10)
        gini = unscaled_gini / ((n_agents - 1) / n_agents)
        return gini

    # Much faster. Slightly overestimated for low n.
    s_endows = np.sort(endowments)
    return 1 - (2 / (n_agents + 1)) * np.sum(
        np.cumsum(s_endows) / (np.sum(s_endows) + 1e-10)
    )


def _env_id_of(episode):
    return getattr(episode, "env_id", getattr(episode, "_env_id", 0))


class InfoMetricsCallback(DefaultCallbacks):
    """
    Collects custom metrics from the `info` dict returned by the env.
    – Step metrics: aggregated (avg, median, total) over *all* agents on the worker.
    – Final metrics: last-step values (episode-final snapshot) per agent and totals.
    """

    STEP_METRICS = {
        # name               extractor – receives the agent_info dict
        "Reward": lambda info: info.get("endogenous", {}).get("Reward", 0.0),
        "Research_count": lambda info: info.get("Research_count", [0, 0])[1],
        "Manufacture_volume": lambda info: info.get("Manufacture_volume"),
        "Carbon_idx": lambda info: info.get("inventory", {}).get("Carbon_idx"),
        "Emission_rate": lambda info: info.get("Carbon_emission_rate"),
        "CoinEndowment": lambda info: info.get("endogenous", {}).get("CoinEndowment", 0.0),
        "Coin": lambda info: info.get("inventory", {}).get("Coin", 0.0),
        "Building_count": lambda info: info.get("Build", 0.0),
        "Power_efficiency": lambda info: info.get("Power_efficiency"),
        "Green_rate": lambda info: info.get("Green_rate"),
        "Startidx": lambda info: info.get("inventory", {}).get("Startidx"),
        "LaborUtility": lambda info: info.get("endogenous", {}).get("LaborUtility", 0.0),
        "CoinUtility": lambda info: info.get("endogenous", {}).get("CoinUtility", 0.0),
        "CurrentUtility": lambda info: info.get("endogenous", {}).get("CurrentUtility", 0.0),
        "PastUtility": lambda info: info.get("endogenous", {}).get("PastUtility", 0.0),
        "Research_ability": lambda info: info.get("Research_ability", 0.0),
        "MoveLabor": lambda info: info.get("MoveLabor", 0.0),

    }

    FINAL_METRICS = {
        "Costs": lambda info: info.get("endogenous", {}).get("Costs"),
        "Revenue": lambda info: info.get("endogenous", {}).get("Revenue"),
        "Coin": lambda info: info.get("inventory", {}).get("Coin"),
        "Labor": lambda info: info.get("endogenous", {}).get("Labor"),
        "Carbon_project": lambda info: info.get("inventory", {}).get("Carbon_project"),
        "Carbon_emission": lambda info: info.get("endogenous", {}).get("Carbon_emission"),
        "Punishment": lambda info: info.get("Cum_Punishment"),
        "Labor_Cost": lambda info: info.get("endogenous", {}).get("LaborCost"),
        "Power_efficiency": lambda info: info.get("Power_efficiency"),
        "Green_rate": lambda info: info.get("Green_project"),
        "CoinEndowment": lambda info: info.get("endogenous", {}).get("CoinEndowment", 0.0),
        "Reward": lambda info: info.get("endogenous", {}).get("CurrentUtility", 0.0),
        "Building_count": lambda info: info.get("Build", 0.0),
        "BidCost": lambda info: info.get("BidCost", 0.0),
        "BidIncome": lambda info: info.get("BidIncome", 0.0),
        "Research_ability": lambda info: info.get("Research_ability", 0.0),
        "MoveLabor": lambda info: info.get("MoveLabor", 0.0),
        "Carbon_project_it": lambda info: info.get("Carbon_project_it", 0.0),
        "BidLabor": lambda info: info.get("BidLabor", 0.0),
        "ResearchCount": lambda info: info.get("ResearchCount", 0.0),
        "Debuff": lambda info: info.get("Debuff", 0.0),
    }

    def __init__(self, worker_id: int = 1):
        super().__init__()
        self.worker_id = worker_id

    def on_episode_step(
            self, *, worker, base_env, policies, episode: Episode, **kwargs
    ):
        # tracks metrics in STEP_METRICS every step per worker for all agents
        infos = episode._last_infos
        if not infos:
            return
        wid = worker.worker_index

        for agent_id, agent_info in infos.items():
            if agent_id == 'p':
                punishment = agent_info.get("punishment", [])
                episode.hist_data.setdefault(f"worker_{wid}/punishment", []).append(punishment)
                mobile_idx_list = agent_info.get("mobile_idx", [])
                for i, v in enumerate(mobile_idx_list):
                    episode.user_data.setdefault(
                        f"worker_{wid}/agent_{i}/Certificates_Allocated", []
                    ).append(v)
                if "settlement_idx" in agent_info:
                    overdraft = float(np.sum(agent_info["settlement_idx"]))
                    episode.user_data.setdefault(
                        f"worker_{wid}/agent_p/Index_Overdraft", []
                    ).append(overdraft)
                continue
            if not isinstance(agent_info, dict):
                continue
            for name, fn in self.STEP_METRICS.items():
                value = fn(agent_info)
                # (keep step collection minimal; profit is computed at episode end)
                if value is None:
                    continue
                key = f"worker_{wid}/agent_{agent_id}/{name}"
                episode.user_data.setdefault(key, []).append(value)

    def on_episode_end(
            self, *, worker, base_env, policies, episode: Episode, **kwargs
    ):
        wid = worker.worker_index
        eid = _env_id_of(episode)

        # ---- step metrics -----------------
        items = sorted(
            episode.user_data.items(),
            key=lambda kv: (kv[0].split("/")[1], kv[0].split("/", 2)[2])
        )
        for key, series in items:
            if not key.startswith(f"worker_{wid}/") or not series:
                continue

            base = key.split("/", 2)[2]  # drop "worker_X/agent_Y/" and just get the metric name

            episode.custom_metrics[f"worker_{wid}/Med_{base}"] = float(np.median(series))
            episode.custom_metrics[f"worker_{wid}/Avg_{base}"] = float(np.mean(series))
            episode.custom_metrics[f"worker_{wid}/Tot_{base}"] = float(sum(series))

        # ---- per-agent FINAL snapshot & episode totals (Revenue, Costs, Profit, Margin) ----
        if wid <= self.worker_id and eid == 0:

            for key, series in episode.user_data.items():
                if not key.startswith(f"worker_{wid}/") or not series:
                    continue
                agent, name = key.split("/", 2)[1], key.split("/", 2)[2]
                if agent=='__common__':
                    continue
                series = np.asarray(series, dtype=float)
                episode.custom_metrics[f"worker_{wid}/{agent}/Tot_{name}"] = float(np.sum(series))

        # ---- FINAL distribution stats and Gini for Coin ---
        for name, fn in self.FINAL_METRICS.items():
            metric = []
            for k, v in episode._last_infos.items():
                if k != 'p' and k!='__common__':
                    valf = fn(v)
                    if valf is not None:
                        metric.append(float(valf))
            if not metric:
                continue
            episode.custom_metrics[f"worker_{wid}/Total_{name}_final"] = float(np.sum(metric))
            if name == "Coin":
                episode.custom_metrics[f"worker_{wid}/Gini_idx_final"] = get_gini(np.array(metric, float))

class ResultInfoMetricsCallback(DefaultCallbacks):
    """
    Collects custom metrics from the `info` dict returned by the env.
    – Step metrics: aggregated (avg, median, total) over *all* agents on the worker.
    – Final metrics: last-step values (episode-final snapshot) per agent and totals.
    """
    STEP_METRICS = {
        # name               extractor – receives the agent_info dict
        "Reward": lambda info: info.get("endogenous", {}).get("Reward", -42),
        "Research_count": lambda info: info.get("Research_count", [0, 0])[1],
        "Manufacture_volume": lambda info: info.get("Manufacture_volume",-42),
        "Cum_Punishment": lambda info: info.get("Cum_Punishment",-42),
        "Tot_Move": lambda info: info.get("Move", -42),
        "Tot_Carbon_project": lambda info: info.get("inventory", {}).get("Carbon_project",-42),
        "Carbon_idx": lambda info: info.get("inventory", {}).get("Carbon_idx",-42),
        "Emission_rate": lambda info: info.get("Carbon_emission_rate",-42),
        "CoinEndowment": lambda info: info.get("endogenous", {}).get("CoinEndowment", -42),
        "Coin": lambda info: info.get("inventory", {}).get("Coin", -42),
        "Tot_Build": lambda info: info.get("Build", -42),
        "Power_efficiency": lambda info: info.get("Power_efficiency",-42),
        "Green_rate": lambda info: info.get("Green_rate",-42),
        "Startidx": lambda info: info.get("inventory", {}).get("Startidx",-42),
        "LaborUtility": lambda info: info.get("endogenous", {}).get("LaborUtility", -42),
        "CoinUtility": lambda info: info.get("endogenous", {}).get("CoinUtility", -42),
        "CurrentUtility": lambda info: info.get("endogenous", {}).get("CurrentUtility", -42),
        "PastUtility": lambda info: info.get("endogenous", {}).get("PastUtility", -42),
        "Research_ability": lambda info: info.get("Research_ability", -42),
        "MoveLabor": lambda info: info.get("MoveLabor", -42),

    }

    def __init__(self, worker_id: int = 1):
        super().__init__()
        self.worker_id = worker_id

    def on_episode_step(
            self, *, worker, base_env, policies, episode: Episode, env_index: Optional[int] = None,
 **kwargs
    ):
        # tracks metrics in STEP_METRICS every step per worker for all agents
        infos = episode._last_infos
        if not infos:
            return
        wid = worker.worker_index
        # Access the unwrapped environment
        env = base_env.get_sub_environments()[env_index]

        # If env is still wrapped, unwrap it
        if hasattr(env, 'env'):
            env = env.env

        # Now access world
        world = env.world
        maps = world.maps
        #to be continued
        """ path = f"/nas/ucb/sophialudewig/Minimalist/rllib/worker_{wid}_episode_info.log"
        with open(path, "a") as fh:
            # Get specific landmark maps
            for landmark_name in maps.keys():
                landmark_map = maps.get(landmark_name)
                fh.write(np.array2string(landmark_map))"""
        for agent_id, agent_info in infos.items():
            if agent_id == 'p':
                punishment = agent_info.get("punishment", [])
                episode.hist_data.setdefault(f"worker_{wid}/punishment_ts", []).append(punishment)
                mobile_idx_list = agent_info.get("mobile_idx", [])
                for i, v in enumerate(mobile_idx_list):
                    episode.hist_data.setdefault(f"worker_{wid}/agent_{i}/Certificates_Allocated_ts", []
                                                 ).append(v)
                if "settlement_idx" in agent_info:
                    overdraft = float(np.sum(agent_info["settlement_idx"]))
                    episode.hist_data.setdefault(
                        f"worker_{wid}/agent_p/Index_Overdraft_ts", []
                    ).append(overdraft)
                    for i, v in enumerate(agent_info["settlement_idx"]):
                        episode.hist_data.setdefault(
                            f"worker_{wid}/agent_{i}/Index_Overdraft_ts", []
                        ).append(v)
                continue
            if not isinstance(agent_info, dict):
                continue

            for name, fn in self.STEP_METRICS.items():
                value = fn(agent_info)
                # (keep step collection minimal; profit is computed at episode end)
                if value is None or agent_id=='__common__':
                    continue
                key = f"worker_{wid}/agent_{agent_id}/{name}_ts"
                episode.hist_data.setdefault(key, []).append(value)


    def on_episode_end(
            self, *, worker, base_env, policies, episode: Episode, env_index: Optional[int] = None, **kwargs
    ):
        """Copy hist_data into custom_metrics so evaluation can see it."""
        wid = worker.worker_index

        # Copy all hist_data into custom_metrics for evaluation reporting
        for key, series in episode.hist_data.items():
            if not series or not key.startswith(f"worker_{wid}/"):
                continue

            # Store the full time series in custom_metrics
            # RLlib will include these in evaluation results
            episode.custom_metrics[key] = series

