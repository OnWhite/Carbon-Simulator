import numpy as np
from ray.rllib.evaluation.episode import Episode
import sys
import os
import atexit
import cProfile
import logging
import socket
from ray.rllib.algorithms.callbacks import DefaultCallbacks

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
        "Research_count_1": lambda info: info.get("Research_count", [0, 0])[1],
        "Manufacture_volume": lambda info: info.get("Manufacture_volume"),
        "Carbon_idx": lambda info: info.get("inventory", {}).get("Carbon_idx"),
        "Emission_rate": lambda info: info.get("Carbon_emission_rate"),
        "Power_efficiency": lambda info: info.get("Power_efficiency"),
        "Green_rate": lambda info: info.get("Green_rate"),
        "Startidx": lambda info: info.get("inventory", {}).get("Startidx"),
        "LaborUtility": lambda info: info.get("endogenous", {}).get("LaborUtility", 0.0),
        "CoinUtility": lambda info: info.get("endogenous", {}).get("CoinUtility", 0.0),
        "CurrentUtility": lambda info: info.get("endogenous", {}).get("CurrentUtility", 0.0),
        "PastUtility": lambda info: info.get("endogenous", {}).get("PastUtility", 0.0),
        "CoinEndowment": lambda info: info.get("endogenous", {}).get("CoinEndowment", 0.0),
        "Building_count": lambda info: info.get("endogenous", {}).get("Build", 0.0),
        "Researchability": lambda info: info.get("endogenous", {}).get("Researchability", 0.0),
    }

    FINAL_METRICS = {
        "Costs": lambda info: info.get("endogenous", {}).get("Costs"),
        "Revenue": lambda info: info.get("endogenous", {}).get("Revenue"),
        "Coin": lambda info: info.get("inventory", {}).get("Coin"),
        "Labor": lambda info: info.get("endogenous", {}).get("Labor"),
        "Carbon_project": lambda info: info.get("inventory", {}).get("Carbon_project"),
        "Carbon_emission": lambda info: info.get("endogenous", {}).get("Carbon_emission"),
        "Punishment": lambda info: info.get("endogenous", {}).get("Punishment"),
        "Labor_Cost": lambda info: info.get("endogenous", {}).get("LaborCost"),
        "Power_efficiency": lambda info: info.get("Power_efficiency"),
        "Green_rate": lambda info: info.get("Green_rate"),
        "Reward": lambda info: info.get("endogenous", {}).get("CurrentUtility", 0.0),
        "Building_count": lambda info: info.get("endogenous", {}).get("Build", 0.0),
        "BidCost": lambda info: info.get("endogenous", {}).get("BidCost", 0.0),
        "BidIncome": lambda info: info.get("endogenous", {}).get("BidIncome", 0.0),
        "Researchability": lambda info: info.get("endogenous", {}).get("Researchability", 0.0),
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

        # ---- step metrics: avg / median -----------------
        curr_base = ""
        arr = []
        arr2 = []
        arr_cert_sum = []
        arr_idx_sum = []
        items = sorted(
            episode.user_data.items(),
            key=lambda kv: (kv[0].split("/")[1], kv[0].split("/", 2)[2])
        )
        for key, series in items:
            if not key.startswith(f"worker_{wid}/") or not series:
                continue

            base = key.split("/", 2)[2]  # drop "worker_X/agent_Y/"
            if base != curr_base:
                if curr_base != "":
                    episode.custom_metrics[f"worker_{wid}/Med_{curr_base}"] = float(np.median(arr))
                    episode.custom_metrics[f"worker_{wid}/Avg_{curr_base}"] = float(np.mean(arr2))
                arr = []
                arr2 = []
                curr_base = base
            series = np.asarray(series, dtype=float)
            arr.append(float(np.median(series)))
            if base == "Certificates_Allocated":
                arr_cert_sum.append(float(np.sum(series)))
            elif base == "Carbon_idx":
                arr_idx_sum.append(float(np.sum(series)))
            arr2.append(float(np.mean(series)))
        if curr_base is not None and arr:
            episode.custom_metrics[f"worker_{wid}/Med_{curr_base}"] = float(np.median(arr))
            episode.custom_metrics[f"worker_{wid}/Avg_{curr_base}"] = float(np.mean(arr2))

        # Derived per-worker metric
        episode.custom_metrics[f"worker_{wid}/Remaining_Manufacturing_Potential"] = (
            float(np.sum(arr_idx_sum) / np.sum(arr_cert_sum)) if np.sum(arr_cert_sum) != 0 else 0.0
        )

        # ---- per-agent FINAL snapshot & episode totals (Revenue, Costs, Profit, Margin) ----
        if wid <= self.worker_id and eid == 0:
            val = {}
            val1 = {}
            startidx = {}
            emissionrate = {}
            manufacturevolume = {}
            total_optimal_capacity = 0.0
            for key, series in episode.user_data.items():
                if not key.startswith(f"worker_{wid}/") or not series:
                    continue
                agent, name = key.split("/", 2)[1], key.split("/", 2)[2]
                series = np.asarray(series, dtype=float)
                episode.custom_metrics[f"worker_{wid}/{agent}/Med_{name}"] = float(np.median(series))
                if name == "Certificates_Allocated" and agent != "p":
                    val[agent] = float(np.sum(series))
                elif name == "Carbon_idx" and agent != "p":
                    val1[agent] = float(np.sum(series))
                elif name == "RewardPlanner" and agent == "p":
                    episode.custom_metrics[f"worker_{wid}/PlannerReward_final"] = float(np.sum(series))
                elif name == "Startidx" and agent != "p":
                    startidx[agent] = float(np.average(series))
                elif name == "Emission_rate" and agent != "p":
                    episode.custom_metrics[f"worker_{wid}/{agent}/Avg_Emission_rate"] = float(np.average(series))
                    emissionrate[agent] = float(np.average(series))
                elif name == "Manufacture_volume" and agent != "p":
                    manufacturevolume[agent] = float(np.average(series))
            optimal_capacity = 0.0
            for agent, value in val.items():
                episode.custom_metrics[f"worker_{wid}/{agent}/Remaining_Manufacturing_Potential"] = (
                    float(val1[agent] / value) if value != 0 and agent in val1 else 0.0
                )
                optimal_capacity = float(startidx[agent] / (emissionrate[agent] * manufacturevolume[
                    agent]) if agent in startidx and agent in emissionrate and agent in manufacturevolume else 0.0)
                episode.custom_metrics[f"worker_{wid}/{agent}/Optimal_Production_Capacity"] = (
                    optimal_capacity
                )
                episode.custom_metrics[f"worker_{wid}/{agent}/emmissionrate"] = (emissionrate[agent])
                episode.custom_metrics[f"worker_{wid}/{agent}/manufacturevolume"] = (manufacturevolume[agent])
                episode.custom_metrics[f"worker_{wid}/{agent}/startidx"] = (startidx[agent])
                total_optimal_capacity += optimal_capacity
            episode.custom_metrics[f"worker_{wid}/Total_Optimal_Production_Capacity"] = float(total_optimal_capacity)
        tot_rev = 0.0
        tot_prf = 0.0
        tot_cost = 0.0
        tot_coinlaborcost = 0.0
        tot_pun = 0.0
        for k, info in episode._last_infos.items():
            if k == 'p' or not isinstance(info, dict):
                continue
            inv = info.get("inventory", {}) or {}
            endo = info.get("endogenous", {}) or {}

            rev = float(endo.get("Revenue", 0.0) or 0.0)
            cst = float(endo.get("Costs", 0.0) or 0.0)
            prf = rev - cst
            coin = inv.get("Coin", None)
            lc = endo.get("LaborCost", None)
            pun = endo.get("Punishment", None)
            if wid <= self.worker_id and eid == 0:
                base = f"worker_{wid}/agent_{k}"
                episode.custom_metrics[f"{base}/Revenue_final"] = rev
                episode.custom_metrics[f"{base}/Costs_final"] = cst
                episode.custom_metrics[f"{base}/Profit_final"] = prf
                episode.custom_metrics[f"{base}/ProfitMargin_final"] = (prf / rev) if rev != 0 else 0.0

                # other finals
                mv = info.get("Manufacture_volume", None)
                if mv is not None:
                    episode.custom_metrics[f"{base}/Manufacture_volume_final"] = float(mv)
                if coin is not None:
                    episode.custom_metrics[f"{base}/Coin_final"] = float(coin)
                carbon_idx = inv.get("Carbon_idx", None)
                if carbon_idx is not None:
                    episode.custom_metrics[f"{base}/Carbon_idx_final"] = float(carbon_idx)
                labor = endo.get("Labor", None)
                if labor is not None:
                    episode.custom_metrics[f"{base}/Labor_final"] = float(labor)
                ce = endo.get("Carbon_emission", None)
                if ce is not None:
                    episode.custom_metrics[f"{base}/Carbon_emission_final"] = float(ce)
                if pun is not None:
                    episode.custom_metrics[f"{base}/Punishment_final"] = float(pun)
                if lc is not None:
                    episode.custom_metrics[f"{base}/Labor_Cost_final"] = float(lc)
                if coin is not None and lc is not None:
                    episode.custom_metrics[f"{base}/Coin-LaborCost_final"] = float(coin - lc)
                    episode.custom_metrics[f"{base}/Profit-(Coin-LaborCost)_final"] = prf - float(coin - lc)
            tot_rev += rev
            tot_prf += prf
            tot_cost += cst
            tot_pun += pun if pun is not None else 0.0
            tot_coinlaborcost += float(coin - lc) if coin is not None and lc is not None else 0.0

        episode.custom_metrics[f"worker_{wid}/Episode_Revenue_final"] = tot_rev
        episode.custom_metrics[f"worker_{wid}/Episode_Profit_final"] = tot_prf
        episode.custom_metrics[f"worker_{wid}/Episode_Cost_final"] = tot_cost
        episode.custom_metrics[f"worker_{wid}/Episode_Coin-LaborCost_final"] = tot_coinlaborcost
        episode.custom_metrics[f"worker_{wid}/Episode_Profit-(Coin-LaborCost)_final"] = tot_prf - tot_coinlaborcost
        episode.custom_metrics[f"worker_{wid}/Episode_Punishment_final"] = tot_pun
        episode.custom_metrics[f"worker_{wid}/Episode_ProfitMargin_final"] = (
            float(tot_prf / tot_rev) if tot_rev != 0 else 0.0
        )
        # ---- FINAL distribution stats (Avg/Med) and Gini for Coin ----
        for name, fn in self.FINAL_METRICS.items():
            metric = []
            for k, v in episode._last_infos.items():
                if k != 'p':
                    valf = fn(v)
                    if valf is not None:
                        metric.append(float(valf))
            if not metric:
                continue
            episode.custom_metrics[f"worker_{wid}/Avg_{name}_final"] = float(np.mean(metric))
            episode.custom_metrics[f"worker_{wid}/Total_{name}_final"] = float(np.sum(metric))
            if name == "Coin":
                episode.custom_metrics[f"worker_{wid}/Gini_idx_final"] = get_gini(np.array(metric, float))
