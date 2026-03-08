#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  BEVFormer Evaluation Launcher (Modernized)
#  Features: Flexible eval metrics, checkpoint validation,
#            result saving, colored output, FPS reporting
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Colors ─────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

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
    echo -e "${BOLD}BEVFormer Evaluation Launcher${NC}"
    echo ""
    echo -e "${CYAN}Usage:${NC}"
    echo "  ./tools/dist_test.sh <CONFIG> <CHECKPOINT> [OPTIONS...]"
    echo ""
    echo -e "${CYAN}Arguments:${NC}"
    echo "  CONFIG        Path to config file (required)"
    echo "  CHECKPOINT    Path to checkpoint file (required)"
    echo ""
    echo -e "${CYAN}Options (passed to test.py):${NC}"
    echo "  --eval METRICS        Evaluation metrics (default: bbox)"
    echo "  --out FILE            Save predictions to pickle file"
    echo "  --save-results-dir DIR  Save JSON evaluation results"
    echo "  --fuse-conv-bn        Fuse conv-bn for faster inference"
    echo "  --show-dir DIR        Save visualization results"
    echo "  --cfg-options K=V     Override config options"
    echo ""
    echo -e "${CYAN}Examples:${NC}"
    echo "  ./tools/dist_test.sh configs/bevformer_tiny.py ckpts/model.pth"
    echo "  ./tools/dist_test.sh configs/bevformer_tiny.py ckpts/model.pth --eval bbox --fuse-conv-bn"
    echo "  ./tools/dist_test.sh configs/bevformer_tiny.py ckpts/model.pth --save-results-dir ./results"
    echo "  ./tools/dist_test.sh configs/bevformer_tiny.py ckpts/model.pth --out predictions.pkl"
    echo ""
    exit 0
}

# ─── Parse Arguments ───────────────────────────────────────────────
if [ $# -lt 2 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
fi

CONFIG=$1
CHECKPOINT=$2
shift 2

# Check if --eval is specified in remaining args; if not, add default
HAS_EVAL=false
for arg in "$@"; do
    if [ "$arg" = "--eval" ]; then
        HAS_EVAL=true
        break
    fi
done

if [ "$HAS_EVAL" = false ]; then
    EXTRA_ARGS="--eval bbox $@"
else
    EXTRA_ARGS="$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ─── Validation ─────────────────────────────────────────────────────
header "BEVFormer Evaluation Launcher"

# Check config file
if [ ! -f "$CONFIG" ]; then
    error "Config file not found: $CONFIG"
    echo ""
    echo -e "${DIM}Available configs:${NC}"
    find projects/configs -name "*.py" -not -path "*/_base_/*" -not -path "*__pycache__*" 2>/dev/null | sort | head -20
    echo ""
    exit 1
fi

# Check checkpoint file
if [ ! -f "$CHECKPOINT" ]; then
    error "Checkpoint file not found: $CHECKPOINT"
    echo ""
    # Suggest available checkpoints
    echo -e "${DIM}Available checkpoints:${NC}"
    find . -name "*.pth" -not -path "./.git/*" 2>/dev/null | sort | head -20
    echo ""
    exit 1
fi

# File size info
CKPT_SIZE=$(du -h "$CHECKPOINT" 2>/dev/null | cut -f1)

info "Config:     $CONFIG"
info "Checkpoint: $CHECKPOINT ($CKPT_SIZE)"
info "Eval args:  ${EXTRA_ARGS}"

# ─── Launch Evaluation ──────────────────────────────────────────────
header "Running Evaluation"

PYTHONPATH="${PROJECT_ROOT}":${PYTHONPATH:-} \
python "${SCRIPT_DIR}/test.py" \
    "$CONFIG" \
    "$CHECKPOINT" \
    $EXTRA_ARGS

EXIT_CODE=$?

# ─── Post-Evaluation ───────────────────────────────────────────────
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    info "Evaluation complete!"
else
    error "Evaluation failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi