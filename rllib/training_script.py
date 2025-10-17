import argparse
import json
import logging
import os
import ray
import sys
import time
from callback import InfoMetricsCallback, ProfilingCallbacks
import matplotlib.pyplot as plt
import numpy as np
import wandb
import shutil
wandb.login(key="2c1f8f77f938086f691891b269af9d5e4925c425")
from torch_models import ConvRnn

from ray import train
import utils.saving as saving
import yaml
from env_wrapper import RLlibEnvWrapper
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import NoopLogger, pretty_print
import pathlib
BASE = "/scratch/$USER"  # or an absolute path you own, e.g. "/nas/ucb/yourid/ray"
BASE = os.path.expandvars(BASE)
TMP_DIR   = os.path.join(BASE, "ray_tmp")
SPILL_DIR = os.path.join(BASE, "ray_spill")

pathlib.Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(SPILL_DIR).mkdir(parents=True, exist_ok=True)

# Make Ray use these instead of /tmp
os.environ["RAY_TMPDIR"] = TMP_DIR
# Optional but nice: also steer Python temp
os.environ["TMPDIR"] = TMP_DIR

logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(message)s")
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_dir", type=str, default='exp', help="Path to the directory for this run."
    )

    args = parser.parse_args()
    run_directory = args.run_dir

    config_path = os.path.join(args.run_dir, "config.yaml")
    assert os.path.isdir(args.run_dir)
    assert os.path.isfile(config_path)

    with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)

    return run_directory, run_configuration


def build_trainer(run_configuration, tune_params=None):
    """Finalize the trainer config by combining the sub-configs."""
    trainer_config = run_configuration.get("trainer")
    if tune_params:
        trainer_config.update(tune_params)
    # === Env ===
    env_config = {
        "env_config_dict": run_configuration.get("env"),
        "num_envs_per_worker": trainer_config.get("num_envs_per_worker"),
    }

    # === Seed ===
    if trainer_config["seed"] is None:
        try:
            start_seed = int(run_configuration["metadata"]["launch_time"])
        except KeyError:
            start_seed = int(time.time())
    else:
        start_seed = int(trainer_config["seed"])

    final_seed = int(start_seed % (2 ** 16)) * 1000
    logger.info("seed (final): %s", final_seed)

    # === Multiagent Policies ===
    dummy_env = RLlibEnvWrapper(env_config)

    # Policy tuples for agent/planner policy types
    agent_policy_tuple = (
        None,
        dummy_env.observation_space,
        dummy_env.action_space,
        run_configuration.get("agent_policy"),
    )
    planner_policy_tuple = (
        None,
        dummy_env.observation_space_pl,
        dummy_env.action_space_pl,
        run_configuration.get("planner_policy"),
    )

    policies = {"a": agent_policy_tuple, "p": planner_policy_tuple}

    # Which policies to train
    if run_configuration["general"]["train_planner"] and not run_configuration["general"]["fix_mobile"]:
        policies_to_train = ["a", "p"]
    elif not run_configuration["general"]["train_planner"] and not run_configuration["general"]["fix_mobile"]:
        policies_to_train = ["a"]
    elif run_configuration["general"]["train_planner"] and run_configuration["general"]["fix_mobile"]:
        policies_to_train = ["p"]
    else:
        raise ValueError("must train one agent")

    # === Finalize and create ===
    trainer_config.update(
        {
            "env_config": env_config,
            "seed": final_seed,
            "multiagent": {
                "policies": policies,
                "policies_to_train": policies_to_train,
                "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: "a" if str(agent_id).isdigit() else "p",
            },
            "metrics_smoothing_episodes": trainer_config.get("num_workers")
                                          * trainer_config.get("num_envs_per_worker"),
        }
    )

    def logger_creator(config):
        return NoopLogger({}, "/tmp")

    from ray.rllib.algorithms.callbacks import MultiCallbacks

    ppo_trainer = PPOConfig().update_from_dict(trainer_config).callbacks(
            lambda: ProfilingCallbacks()
        ).reporting(keep_per_episode_custom_metrics=False,
                      metrics_num_episodes_for_smoothing=1).build(
        env=RLlibEnvWrapper, logger_creator=logger_creator)
    return ppo_trainer


def set_up_dirs_and_maybe_restore(run_directory, run_configuration, trainer_obj):
    # === Set up Logging & Saving, or Restore ===
    # All model parameters are always specified in the settings YAML.
    # We do NOT overwrite / reload settings from the previous checkpoint dir.
    # 1.0.0. For new runs, the only object that will be loaded from the checkpoint dir
    #    are model weights.
    # 2. For crashed and restarted runs, load_snapshot will reload the full state of
    #    the Trainer(s), including metadata, optimizer, and models.
    (
        dense_log_directory,
        ckpt_directory,
        restore_from_crashed_run,
    ) = saving.fill_out_run_dir(run_directory)

    # If this is a starting from a crashed run, restore the last trainer snapshot
    if restore_from_crashed_run:
        logger.info(
            "ckpt_dir already exists! Planning to restore using latest snapshot from "
            "earlier (crashed) run with the same ckpt_dir %s",
            ckpt_directory,
        )

        at_loads_a_ok = saving.load_snapshot(
            trainer_obj, run_directory, load_latest=True
        )

        # at this point, we need at least one good ckpt restored
        if not at_loads_a_ok:
            logger.fatal(
                "restore_from_crashed_run -> restore_run_dir %s, but no good ckpts "
                "found/loaded!",
                run_directory,
            )
            sys.exit()

        # === Trainer-specific counters ===
        training_step_last_ckpt = (
            int(trainer_obj._timesteps_total) if trainer_obj._timesteps_total else 0
        )
        epis_last_ckpt = (
            int(trainer_obj._episodes_total) if trainer_obj._episodes_total else 0
        )

    else:
        logger.info("Not restoring trainer...")
        # === Trainer-specific counters ===
        training_step_last_ckpt = 0
        epis_last_ckpt = 0

        # For new runs, load only torch checkpoint weights
        starting_weights_path_agents = run_configuration["general"].get(
            "restore_weights_agents", ""
        )
        if starting_weights_path_agents:
            logger.info("Restoring agents weights...")
            saving.load_model_weights(trainer_obj, starting_weights_path_agents)
        else:
            logger.info("Starting with fresh agent weights.")

        starting_weights_path_planner = run_configuration["general"].get(
            "restore_weights_planner", ""
        )
        if starting_weights_path_planner:
            logger.info("Restoring planner weights...")
            saving.load_model_weights(trainer_obj, starting_weights_path_planner)
        else:
            logger.info("Starting with fresh planner weights.")

    return (
        dense_log_directory,
        ckpt_directory,
        restore_from_crashed_run,
        training_step_last_ckpt,
        epis_last_ckpt,
    )


def maybe_store_dense_log(
        trainer_obj, result_dict, dense_log_freq, dense_log_directory, trainer_step_last_ckpt
):
    if result_dict["episodes_this_iter"] > 0 and dense_log_freq > 0:
        training_iteration = result_dict["training_iteration"]

        if training_iteration == 1 or training_iteration - trainer_step_last_ckpt >= dense_log_freq:
            log_dir = os.path.join(
                dense_log_directory,
                "logs_{:06d}".format(result_dict["training_iteration"]),
            )
            trainer_step_last_ckpt = int(training_iteration)
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            saving.write_dense_logs(trainer_obj, log_dir)
            logger.info(">> Wrote dense logs to: %s", log_dir)

    return trainer_step_last_ckpt

def maybe_save(trainer_obj, result_dict, ckpt_freq, ckpt_directory, trainer_step_last_ckpt):
    training_iteration = result_dict["training_iteration"]

    # Check if saving this iteration
    if (
            result_dict["episodes_this_iter"] > 0
    ):  # Don't save if midway through an episode.

        if ckpt_freq > 0:
            if training_iteration - trainer_step_last_ckpt >= ckpt_freq:
                # saving.save_snapshot(trainer_obj, ckpt_directory, suffix="")
                saving.save_model_weights(
                    trainer_obj, ckpt_directory, training_iteration, suffix="agent"
                )
                saving.save_model_weights(
                    trainer_obj, ckpt_directory, training_iteration, suffix="planner"
                )

                trainer_step_last_ckpt = int(training_iteration)

                logger.info("Checkpoint saved @ step %d", training_iteration)

    return trainer_step_last_ckpt

def plot_reward(run_directory, reward_a, reward_p):
    np_dir = run_directory + "/reward_a.npy"
    np.save(np_dir, np.array(reward_a))

    np_dir = run_directory + "/reward_p.npy"
    np.save(np_dir, np.array(reward_p))

    fig1 = plt.figure()
    plt.plot(range(len(reward_a)), reward_a)
    fig_dir = run_directory + "/reward_a.jpg"
    fig1.savefig(fig_dir)
    plt.close()

    fig2 = plt.figure()
    plt.plot(range(len(reward_a)), reward_p)
    fig_dir = run_directory + "/reward_p.jpg"
    fig2.savefig(fig_dir)
    plt.close()

def tune_train(config, run_dir="exp", run_config=None):
    run_config["trainer"].update(config)
    trainer = build_trainer(run_config)
    while True:
        result = trainer.train()
        agent_reward = result.get('policy_reward_mean', {}).get('a', 0)
        train.report({
            "agent_reward": agent_reward,
        })
def log_custom_metrics(result):
    """Format RLlib custom metrics for W&B with media types."""
    metrics = {}
    cm = result.get("custom_metrics", {})

    for key, val in cm.items():
        if val is None:
            continue

        # Lists/arrays -> media
        if isinstance(val, (list, np.ndarray)):
            arr = np.asarray(val)
            if arr.ndim == 1:
                # Recreates the automatic histogram panel
                metrics[f"hist/{key}"] = wandb.Histogram(arr)
            elif arr.ndim == 2:
                # Heat map as an image
                fig, ax = plt.subplots()
                ax.imshow(arr, aspect="auto")
                ax.set_title(key)
                fig.tight_layout()
                metrics[f"heatmap/{key}"] = wandb.Image(fig)
                plt.close(fig)
            # Higher dims: skip or reduce as needed
            continue

        # Scalars stay scalars
        if isinstance(val, (np.floating, np.integer)):
            val = val.item()
        metrics[key] = val

    return metrics


def create_unique_temp_dir():
    """Create a unique temp directory for this run"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Use a shorter base path but still unique per run
    temp_dir = f"/tmp/ray_{timestamp}"
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


if __name__ == "__main__":
    try:
        # Process the args first
        run_dir, run_config = process_args()

        # Initialize Ray with temp directory
        ray.init(
            log_to_driver=True,
            include_dashboard=False,
            object_store_memory=8 * 1024 ** 3,
            _temp_dir = TMP_DIR,  # belt-and-suspenders with RAY_TMPDIR
            profile=True,  # Add this line
            _system_config = {
                "object_spilling_config": json.dumps({
                    "type": "filesystem",
                    "params": {"directory_path": SPILL_DIR}
                })}
        )


        fh = logging.FileHandler(run_dir+"/train.log")
        logger.addHandler(fh)
        # Initialize W&B
        wandb.init(
            project="Minimal_Testing",
            name="gettingmetricsright",
            config=run_config, #{'env': {'n_agents': 5, 'world_size': [40, 40], 'episode_length': 500, 'period': 50, 'multi_action_mode_agents': False, 'multi_action_mode_planner': True, 'flatten_observations': True, 'flatten_masks': True, 'scenario_name': 'Carbon/Carbon_env', 'components': [{'CarbonTaxation': {'planner_mode': 'active', 'total_idx': 200, 'max_year_percent': 25, 'years_predefined': 'flat', 'agents_predefined': 'grandfathering_ml'}}, {'Carbon_component': {'payment': 10, 'require_Carbon_idx': 1, 'lowest_rate': 0.02, 'research_setting': ['e^-', 0.1], 'random_fails': 0.3, 'delay': 5, 'forget': 25}}, {'Carbon_auction': {'max_bid_ask': 20, 'max_num_orders': 5, 'order_duration': 10}}, {'Gather': {'collect_labor': 30, 'collect_cost_coin': 10}}], 'dense_log_frequency': 20, 'isoelastic_eta': 0.23, 'energy_cost': 0.1, 'energy_warmup_constant': 10000, 'energy_warmup_method': 'auto', 'starting_agent_coin': 20, 'mobile_coefficient': 20}, 'general': {'ckpt_frequency_steps': 500, 'cpus': 8, 'episodes': 50000, 'gpus': 0, 'restore_weights_agents': '', 'restore_weights_planner': '', 'train_planner': False, 'fix_mobile': False, 'dense_log_frequency': 250}, 'agent_policy': {'clip_param': 0.3, 'entropy_coeff': 0.025, 'entropy_coeff_schedule': None, 'gamma': 0.998, 'grad_clip': 10.0, 'kl_coeff': 0.0, 'kl_target': 0.01, 'lambda': 0.98, 'lr': 5e-05, 'lr_schedule': None, 'use_gae': True, 'vf_clip_param': 50.0, 'vf_loss_coeff': 0.05, 'vf_share_layers': False, 'model': {'custom_model': 'Conv_Rnn', 'custom_model_config': {'input_emb_vocab': 20, 'idx_emb_dim': 5, 'num_conv': 2, 'num_fc': 2, 'cell_size': 128}, 'max_seq_len': 50}}, 'planner_policy': {'clip_param': 0.3, 'entropy_coeff': 0.125, 'entropy_coeff_schedule': [[0, 2.0], [50000000, 0.125]], 'gamma': 0.998, 'grad_clip': 10.0, 'kl_coeff': 0.0, 'kl_target': 0.01, 'lambda': 0.98, 'lr': 1e-05, 'lr_schedule': None, 'use_gae': True, 'vf_clip_param': 50.0, 'vf_loss_coeff': 0.05, 'vf_share_layers': False, 'model': {'custom_model': 'Conv_Rnn', 'custom_model_config': {'input_emb_vocab': 20, 'idx_emb_dim': 5, 'num_conv': 2, 'num_fc': 2, 'cell_size': 256}, 'max_seq_len': 100}}, 'trainer': {'batch_mode': 'truncate_episodes', 'env_config': None, 'multiagent': None, 'seed': 22635000, 'num_gpus': 0, 'num_envs_per_worker': 2, 'num_sgd_iter': 1, 'num_workers': 7, 'shuffle_sequences': True, 'sgd_minibatch_size': 1000, 'train_batch_size': 3500, 'observation_filter': 'NoFilter', 'rollout_fragment_length': 250}}
            dir=run_dir #'/Users/work/PycharmProjects/Carbon-Simulator/rllib/exp/defuat'
        )


        # Create a trainer object
        trainer = build_trainer(run_config)

        # Set up directories for logging and saving. Restore if this has already been
        # done (indicating that we're restarting a crashed run). Or, if appropriate,
        # load in starting model weights for the agent and/or planner.
        (
            dense_log_dir,
            ckpt_dir,
            restore_from_crashed_run,
            step_last_ckpt,
            num_parallel_episodes_done,
        ) = set_up_dirs_and_maybe_restore(run_dir, run_config, trainer)

        # ======================
        # === Start training ===
        # ======================
        dense_log_frequency = run_config["general"].get("dense_log_frequency", 0)
        ckpt_frequency = run_config["general"].get("ckpt_frequency_steps", 0)
        global_step = int(step_last_ckpt)
        step_last_log = 0

        reward_result_a, reward_result_p = [], []

        if False:
            search_space = {
                "lr": tune.loguniform(1e-5, 5e-4),
                "entropy_coeff": tune.uniform(0.005, 0.01),  # Add entropy decay schedule if possible
                "num_sgd_iter": tune.choice([5, 10]),
                "grad_clip": tune.uniform(0.5, 3.0),
                "vf_loss_coeff": tune.uniform(0.05, 0.1),
                "clip_param": tune.uniform(0.1, 0.2),  # Smaller for more stable updates
                "lambda": tune.uniform(0.95, 0.99),
            }

            algo = OptunaSearch(
                metric="agent_reward",
                mode="max"
            )
            scheduler = ASHAScheduler(
                metric="agent_reward",
                mode="max",
                max_t=500,
                grace_period=100,  # Evaluate very early
                reduction_factor=2,
            )
            pgf = PlacementGroupFactory(
                [{"CPU": 4, "GPU": 1.0}] + [{"CPU": 4}] * 7
            )

            tune.run(
                tune.with_parameters(tune_train, run_dir=run_dir, run_config=run_config),
                resources_per_trial=pgf,
                config=search_space,
                num_samples=5,
                max_concurrent_trials=1,
                search_alg=algo,
                scheduler=scheduler,
                local_dir=os.path.abspath(os.path.join(run_dir, "tune_results")),
                name="hyperparam_tuning",
            )
        elif True:

            while num_parallel_episodes_done < run_config["general"]["episodes"]:

                # Training
                result = trainer.train()
                # Get formatted metrics
                metrics = log_custom_metrics(result)
                wandb.log({
                    "iteration": result["training_iteration"],
                    "timesteps_total": result["timesteps_total"],
                    "episodes_total": result["episodes_total"],
                    "reward/agent": result.get("policy_reward_mean", {}).get("a", 0),
                    "reward/planner": result.get("policy_reward_mean", {}).get("p", 0),
                    **metrics
                }, step=result["episodes_total"])  # <-- add step to align by episode

                # === Counters++ ===
                num_parallel_episodes_done = result["episodes_total"]
                global_step = result["timesteps_total"]
                curr_iter = result["training_iteration"]
                if num_parallel_episodes_done % run_config["general"]["episodes"]/50 == 0:
                    logger.info(
                        "Iter %d: episodes this-iter %d total %d step -> %d/%d episodes done",
                        curr_iter,
                        result["episodes_this_iter"],
                        global_step,
                        num_parallel_episodes_done,
                        run_config["general"]["episodes"],
                    )

                if curr_iter == 1 or result["episodes_this_iter"] > 0:
                    logger.info(pretty_print(result))

                reward_result_a.append(result.get('policy_reward_mean')["a"] if result.get('policy_reward_mean') else 0)
                reward_result_p.append(result.get('policy_reward_mean')["p"] if result.get('policy_reward_mean') else 0)
                plot_reward(run_dir, reward_result_a, reward_result_p)

                # === Dense logging ===
                #step_last_log = maybe_store_dense_log(trainer, result, dense_log_frequency, dense_log_dir,
                #                                     step_last_log)

                # === Saving ===
                step_last_ckpt = maybe_save(
                    trainer, result, ckpt_frequency, ckpt_dir, step_last_ckpt
                )
            # Finish up
            logger.info("Completing! Saving final snapshot...\n\n")
            # saving.save_snapshot(trainer, ckpt_dir)
            saving.save_model_weights(trainer, ckpt_dir, global_step, suffix="agent")
            saving.save_model_weights(trainer, ckpt_dir, global_step, suffix="planner")
            logger.info("Final snapshot saved! All done.")
    finally:
        #ray.timeline(os.path.join(run_dir, "timeline.json"))
        ray.shutdown()
        wandb.finish()