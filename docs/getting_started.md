# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

## BEVFormer Modernized Scripts (New ✨)

We provide modernized scripts with enhanced features for easier training and evaluation.

## 1. Unified CLI (main.py)
You can now use `main.py` as a unified entry point into the project.
```bash
# See all available commands
python main.py --help

# Show project structure, available configs, and check environment
python main.py info
```

## 2. Training (tools/dist_train.sh)

The new training script automatically detects GPUs and includes environment checks.

**Standard Training:**
```bash
./tools/dist_train.sh projects/configs/bevformer/bevformer_base.py
```
*(Optionally specify GPUs explicitly: `./tools/dist_train.sh [CONFIG] [GPUS]`)*

**Auto-Resume Training:**
Automatically find and resume from the latest checkpoint in your `work_dirs`.
```bash
./tools/dist_train.sh projects/configs/bevformer/bevformer_base.py --auto-resume
```

**Weights & Biases Logging:**
```bash
./tools/dist_train.sh projects/configs/bevformer/bevformer_base.py --wandb --wandb-project bevformer_v1
```

## 3. Evaluation (tools/dist_test.sh)

The evaluation script now supports flexible metrics, FPS statistics, and exporting results to JSON.

**Standard Evaluation:**
```bash
./tools/dist_test.sh projects/configs/bevformer/bevformer_base.py ./ckpts/bevformer_base_epoch_24.pth
```
*(By default, this evaluates the `bbox` metric for 3D detection).*

**Save Evaluation Results to JSON:**
Automatically generates a structured JSON file with metrics and inference speeds.
```bash
./tools/dist_test.sh projects/configs/bevformer/bevformer_base.py ./ckpts/bevformer_base_epoch_24.pth --save-results-dir ./eval_results
```

**Faster Inference (Fuse Conv-BN):**
```bash
./tools/dist_test.sh projects/configs/bevformer/bevformer_base.py ./ckpts/bevformer_base_epoch_24.pth --fuse-conv-bn
```

---

## Using FP16 to train the model

If you need multi-gpu distributed FP16 training with PyTorch Launcher, you can use the legacy script:

```bash
./tools/fp16/dist_train.sh ./projects/configs/bevformer_fp16/bevformer_tiny_fp16.py 8
```

## Visualization 

see [visual.py](../tools/analysis_tools/visual.py)