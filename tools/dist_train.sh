#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  BEVFormer Training Launcher (Modernized)
#  Features: Environment checks, auto GPU detection, colored output,
#            auto-resume, log tee, TensorBoard hints
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Colors ─────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ $1${NC}"; }
warn()  { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠ $1${NC}"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')] ✗ $1${NC}"; }
header() {
    echo ""
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# ─── Usage ──────────────────────────────────────────────────────────
usage() {
    echo ""
    echo -e "${BOLD}BEVFormer Training Launcher${NC}"
    echo ""
    echo -e "${CYAN}Usage:${NC}"
    echo "  ./tools/dist_train.sh <CONFIG> [GPUS] [OPTIONS...]"
    echo ""
    echo -e "${CYAN}Arguments:${NC}"
    echo "  CONFIG        Path to config file (required)"
    echo "  GPUS          Number of GPUs (default: auto-detect, fallback: 1)"
    echo ""
    echo -e "${CYAN}Options (passed to train.py):${NC}"
    echo "  --auto-resume         Auto-resume from latest checkpoint"
    echo "  --wandb               Enable WandB logging"
    echo "  --wandb-project NAME  WandB project name"
    echo "  --work-dir DIR        Custom work directory"
    echo "  --no-validate         Disable validation during training"
    echo "  --cfg-options K=V     Override config options"
    echo "  --seed N              Random seed (default: 0)"
    echo "  --autoscale-lr        Auto-scale learning rate by GPU count"
    echo ""
    echo -e "${CYAN}Examples:${NC}"
    echo "  ./tools/dist_train.sh projects/configs/bevformer/bevformer_tiny.py"
    echo "  ./tools/dist_train.sh projects/configs/bevformer/bevformer_tiny.py 1 --auto-resume"
    echo "  ./tools/dist_train.sh projects/configs/bevformer/bevformer_tiny.py 1 --wandb"
    echo ""
    exit 0
}

# ─── Parse Arguments ───────────────────────────────────────────────
if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
fi

CONFIG=$1
shift

# Check if second argument is a number (GPUS)
if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    GPUS=$1
    shift
else
    # Auto-detect GPU count
    if command -v nvidia-smi &> /dev/null; then
        GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        if [ "$GPUS" -eq 0 ]; then
            GPUS=1
        fi
    else
        GPUS=1
    fi
fi

EXTRA_ARGS="$@"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ─── Environment Checks ────────────────────────────────────────────
header "BEVFormer Training Launcher"

# Check config file
if [ ! -f "$CONFIG" ]; then
    error "Config file not found: $CONFIG"
    echo ""
    echo -e "${DIM}Available configs:${NC}"
    find projects/configs -name "*.py" -not -path "*/_base_/*" -not -path "*__pycache__*" 2>/dev/null | sort | head -20
    echo ""
    exit 1
fi

# Check Python
if ! command -v python &> /dev/null; then
    error "Python not found. Please install Python 3.7+."
    exit 1
fi

PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
info "Python: $PYTHON_VERSION"

# Check CUDA
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    info "PyTorch: $TORCH_VERSION (CUDA $CUDA_VERSION)"
else
    warn "CUDA is not available. Training will be slow."
fi

# Check key dependencies
for pkg in mmcv mmdet mmdet3d mmseg; do
    if python -c "import $pkg" 2>/dev/null; then
        PKG_VER=$(python -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "?")
        info "$pkg: $PKG_VER"
    else
        error "$pkg not installed!"
        exit 1
    fi
done

# ─── Training Configuration Preview ────────────────────────────────
echo ""
info "Config:       $CONFIG"
info "GPUs:         $GPUS"
info "Extra args:   ${EXTRA_ARGS:-none}"

# Extract work_dir from config
CONFIG_NAME=$(basename "$CONFIG" .py)
WORK_DIR="./work_dirs/${CONFIG_NAME}"

# Check for existing checkpoints (resume hint)
if [ -d "$WORK_DIR" ]; then
    LATEST_CKPT=$(find "$WORK_DIR" -name "*.pth" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -n "$LATEST_CKPT" ]; then
        warn "Existing checkpoint found: $LATEST_CKPT"
        echo -e "      ${DIM}Use --auto-resume to continue training${NC}"
    fi
fi

# ─── Launch Training ───────────────────────────────────────────────
header "Launching Training"

info "Mode: Single-process (launcher: none, GPUs: $GPUS)"

# Create log directory
mkdir -p "$WORK_DIR"
LOG_FILE="$WORK_DIR/train_$(date '+%Y%m%d_%H%M%S').log"

echo -e "${DIM}Log file: $LOG_FILE${NC}"
echo ""

# Run training
PYTHONPATH="${PROJECT_ROOT}":${PYTHONPATH:-} \
python "${SCRIPT_DIR}/train.py" \
    "$CONFIG" \
    --launcher none \
    --gpus "$GPUS" \
    --deterministic \
    $EXTRA_ARGS \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# ─── Post-Training ─────────────────────────────────────────────────
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    header "Training Complete"
    info "Results saved to: $WORK_DIR"
    info "Log file: $LOG_FILE"
    echo ""
    echo -e "${CYAN}📊 View TensorBoard:${NC}"
    echo -e "   tensorboard --logdir=$WORK_DIR --port=6006"
    echo ""
    echo -e "${CYAN}🔍 Evaluate model:${NC}"
    LATEST=$(find "$WORK_DIR" -name "*.pth" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -n "$LATEST" ]; then
        echo -e "   ./tools/dist_test.sh $CONFIG $LATEST"
    fi
    echo ""
else
    error "Training failed with exit code: $EXIT_CODE"
    error "Check log file: $LOG_FILE"
    exit $EXIT_CODE
fi
