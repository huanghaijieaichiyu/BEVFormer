#!/usr/bin/env bash

# 使用方法：
# bash tools/train_distill.sh [配置文件] [GPU数量]
# 例如：bash tools/train_distill.sh projects/configs/bevformer_distill/bevformer_pico_from_tiny_distill.py 1  (单GPU)
# 例如：bash tools/train_distill.sh projects/configs/bevformer_distill/bevformer_pico_from_tiny_distill.py 8  (8 GPU, 如果环境支持分布式)

CONFIG=$1
GPUS=${2:-1} # 默认为1个GPU

# 检查是否在WSL环境或者是否明确要求非分布式启动
# 如果GPUS=1，则总是使用 --launcher none
# 如果GPUS>1，在WSL中也使用 --launcher none (train.py内部应能处理多GPU的DataParallel)
# 或者如果希望在原生Linux环境中使用torch.distributed.launch，可以添加一个判断或新的脚本

LAUNCHER="none"
DIST_ARGS=""

# 对于多GPU，如果不是WSL或者用户希望强制使用分布式，可以取消下面的注释
# if [ "$GPUS" -gt 1 ] && [ -z "$WSL_DISTRO_NAME" ]; then
# LAUNCHER="pytorch"
# PORT=${PORT:-29500}
# DIST_ARGS="--nproc_per_node=$GPUS --master_port=$PORT"
# fi

# 当前统一使用 launcher none，适用于WSL和单机多卡非严格分布式场景
# train.py 脚本需要能够通过 cfg.gpu_ids 和 len(cfg.gpu_ids) 来正确处理单卡或多卡(DataParallel)

echo "Running distillation training with launcher: $LAUNCHER, GPUs: $GPUS"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/train.py $CONFIG --launcher $LAUNCHER --gpus $GPUS ${@:3} --deterministic

# 如果希望对多GPU严格使用 torch.distributed.launch，可以像原始脚本那样：
# if [ "$GPUS" -gt 1 ]; then
#     echo "Running distributed distillation training with launcher: pytorch, GPUs: $GPUS"
#     PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#     python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=${PORT:-29500} \
#         $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic
# else
#     echo "Running distillation training with launcher: none, GPUs: $GPUS"
#     PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#     python $(dirname "$0")/train.py $CONFIG --launcher none --gpus $GPUS ${@:3} --deterministic
# fi 