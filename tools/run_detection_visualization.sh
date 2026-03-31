#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

if [ -f "tools/comparison_env.sh" ]; then
    # shellcheck disable=SC1091
    source tools/comparison_env.sh
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "[ERROR] Neither python3 nor python is available in PATH."
        exit 1
    fi
fi

CONFIG="${CONFIG:-projects/configs/bevformer/bevformer_tiny.py}"
CHECKPOINT="${CHECKPOINT:-ckpts/bevformer_tiny_epoch_24.pth}"
DATAROOT_DEFAULT="${NM_ROOT:-datasets/nuscenes}"
NUSC_VERSION_DEFAULT="${NUSC_VERSION:-v1.0-mini}"

RESULTS=""
DATAROOT="${DATAROOT_DEFAULT}"
NUSC_VERSION="${NUSC_VERSION_DEFAULT}"
OUT_DIR="runs/visual_detection"
NUM_SAMPLES=10
SCORE_THRESH=0.2
RUN_TEST=0
SAVE_PANELS=0
WITH_MAP=1
EXTRA_TEST_ARGS=()

usage() {
    cat <<EOF
Usage:
  bash tools/run_detection_visualization.sh [options]

Options:
  --results PATH         Visualize an existing results_nusc.json
  --run-test             Run tools/test.py first, then visualize the newest results_nusc.json
  --config PATH          Model config path (default: ${CONFIG})
  --checkpoint PATH      Checkpoint path (default: ${CHECKPOINT})
  --dataroot PATH        NuScenes dataroot (default: ${DATAROOT_DEFAULT})
  --nusc-version VER     NuScenes version (default: ${NUSC_VERSION_DEFAULT})
  --out-dir DIR          Output visualization directory (default: ${OUT_DIR})
  --num-samples N        Number of samples to visualize (default: ${NUM_SAMPLES})
  --score-thresh X       Prediction score threshold (default: ${SCORE_THRESH})
  --sample-tokens SPEC   Comma-separated tokens or path to token list file
  --save-panels          Keep camera/lidar/map panel images in each sample directory
  --no-map               Disable static map rendering
  --                    Extra args passed through to tools/test.py when using --run-test

Examples:
  bash tools/run_detection_visualization.sh --results test/normal_detection/results_nusc.json
  bash tools/run_detection_visualization.sh --run-test --num-samples 5 -- --cfg-options data.test.samples_per_gpu=1
EOF
}

SAMPLE_TOKENS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --results)
            RESULTS="$2"
            shift 2
            ;;
        --run-test)
            RUN_TEST=1
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --dataroot)
            DATAROOT="$2"
            shift 2
            ;;
        --nusc-version)
            NUSC_VERSION="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --score-thresh)
            SCORE_THRESH="$2"
            shift 2
            ;;
        --sample-tokens)
            SAMPLE_TOKENS="$2"
            shift 2
            ;;
        --save-panels)
            SAVE_PANELS=1
            shift
            ;;
        --no-map)
            WITH_MAP=0
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_TEST_ARGS+=("$@")
            break
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

find_latest_results() {
    find test -name 'results_nusc.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}'
}

if [ "${RUN_TEST}" -eq 1 ]; then
    if [ ! -f "${CONFIG}" ]; then
        echo "[ERROR] Config not found: ${CONFIG}"
        exit 1
    fi
    if [ ! -f "${CHECKPOINT}" ]; then
        echo "[ERROR] Checkpoint not found: ${CHECKPOINT}"
        exit 1
    fi

    echo "============================================================"
    echo "  Running BEVFormer test before visualization"
    echo "============================================================"
    echo "  Config:     ${CONFIG}"
    echo "  Checkpoint: ${CHECKPOINT}"
    echo "  Dataroot:   ${DATAROOT}"
    echo "  Version:    ${NUSC_VERSION}"
    echo "============================================================"

    "${PYTHON_BIN}" tools/test.py "${CONFIG}" "${CHECKPOINT}" \
        --eval bbox \
        --cfg-options \
            "data.test.data_root=${DATAROOT}/" \
            "data.test.ann_file=${DATAROOT}/nuscenes_infos_temporal_val.pkl" \
        "${EXTRA_TEST_ARGS[@]}"

    RESULTS="$(find_latest_results)"
    if [ -z "${RESULTS}" ]; then
        echo "[ERROR] No results_nusc.json was found after running tools/test.py."
        exit 1
    fi
fi

if [ -z "${RESULTS}" ]; then
    RESULTS="$(find_latest_results)"
fi

if [ -z "${RESULTS}" ] || [ ! -f "${RESULTS}" ]; then
    echo "[ERROR] Could not locate a valid results_nusc.json."
    echo "        Use --results PATH or run with --run-test."
    exit 1
fi

VIS_ARGS=(
    --results "${RESULTS}"
    --dataroot "${DATAROOT}"
    --nusc-version "${NUSC_VERSION}"
    --out-dir "${OUT_DIR}"
    --num-samples "${NUM_SAMPLES}"
    --score-thresh "${SCORE_THRESH}"
)

if [ -n "${SAMPLE_TOKENS}" ]; then
    VIS_ARGS+=(--sample-tokens "${SAMPLE_TOKENS}")
fi
if [ "${SAVE_PANELS}" -eq 1 ]; then
    VIS_ARGS+=(--save-panels)
fi
if [ "${WITH_MAP}" -eq 0 ]; then
    VIS_ARGS+=(--no-map)
fi

echo "============================================================"
echo "  Rendering BEVFormer detection outputs"
echo "============================================================"
echo "  Results:    ${RESULTS}"
echo "  Dataroot:   ${DATAROOT}"
echo "  Version:    ${NUSC_VERSION}"
echo "  Output Dir: ${OUT_DIR}"
echo "============================================================"

"${PYTHON_BIN}" tools/analysis_tools/visual.py "${VIS_ARGS[@]}"

echo ""
echo "Done. Combined figures are under ${OUT_DIR}/sample_*/combined.png"
