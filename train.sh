#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --qos=high
#SBATCH --time=168:00:00
#SBATCH --job-name=ray_train
cd /nas/ucb/sophialudewig/Carbon-Simulator-minimal/Carbon-Simulator

source .venv/bin/activate

PYTHONPATH=. python3 rllib/training_script.py --run_dir rllib/exp/pl1