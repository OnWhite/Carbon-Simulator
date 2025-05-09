#!/bin/bash
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --qos=high
#SBATCH --time=168:00:00
#SBATCH --job-name=ray_train

cd /nas/ucb/sophialudewig/Carbon-Simulator-minimal/Carbon-Simulator
source .venv/bin/activate

# Start Ray head node (use appropriate IP if running multi-node; here it's single-node)
ray start --head --num-cpus=32 --num-gpus=2

# Run your training script
python3 rllib/training_script.py --run_dir rllib/exp/defuat

# Optional: Stop Ray after the job completes
ray stop
