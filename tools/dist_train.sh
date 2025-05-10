#!/usr/bin/env bash

CONFIG=$1
# GPUS variable can represent the number of GPUs for --gpus argument in train.py
# Default to 1 GPU if not specified by the second script argument.
GPUS=${2:-1}
# PORT is no longer needed for non-distributed launch

echo "Running in WSL-compatible mode (launcher: none, GPUs: $GPUS)."

# Set PYTHONPATH and execute train.py directly.
# --launcher none indicates non-distributed training.
# Pass the GPUS variable to the --gpus argument of train.py.
# Pass any additional arguments (from the 3rd one onwards) to train.py.
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/train.py $CONFIG --launcher none --gpus $GPUS ${@:3} --deterministic
