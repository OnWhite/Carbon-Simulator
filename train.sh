#!/bin/bash

# Configuration Parameters
CPUS_TO_USE=32
GPUS_NEEDED=1
MAX_ALLOWED_MEMORY_USED=500  # MiB

cd /nas/ucb/sophialudewig/Carbon-Simulator-minimal/Carbon-Simulator
source .venv/bin/activate

# Find GPUs with 0% utilization and less than MAX_ALLOWED_MEMORY_USED MiB used
AVAILABLE_GPUS=$(gpustat --json | python3 -c "
import sys, json
data = json.load(sys.stdin)
gpus = data['gpus']
free_gpus = [
    str(g['index']) for g in gpus
    if g['utilization.gpu'] == 0 and g['memory.used'] <= ${MAX_ALLOWED_MEMORY_USED}
]
if len(free_gpus) >= ${GPUS_NEEDED}:
    print(','.join(free_gpus[:${GPUS_NEEDED}]))
")

if [ -z \"$AVAILABLE_GPUS\" ]; then
    echo 'Not enough sufficiently free GPUs available. Exiting...'
    exit 1
fi

echo \"Using free GPUs: $AVAILABLE_GPUS\"

# Limit CPU usage
export OMP_NUM_THREADS=$CPUS_TO_USE
export MKL_NUM_THREADS=$CPUS_TO_USE
export NUMEXPR_NUM_THREADS=$CPUS_TO_USE
export PYTHONPATH=.

# Start Ray
ray start --head --num-cpus=$CPUS_TO_USE --num-gpus=$GPUS_NEEDED

# Run the training script
CUDA_VISIBLE_DEVICES=$AVAILABLE_GPUS python3 rllib/training_script.py --run_dir rllib/exp/defuat

# Stop Ray after the job
ray stop
