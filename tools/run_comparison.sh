#!/bin/bash
# =============================================================================
# 低照度 vs 正常检测对比可视化 - 完整流程 (支持断点续跑)
# =============================================================================
# 每个步骤都有完成标记检查，已完成的步骤自动跳过
# 如需强制重跑某步骤，删除对应的 .done 标记文件即可
#
# 标记文件位置: .pipeline_status/
#   step0_data_prep.done  - 数据准备完成
#   step1_ll_inference.done - 低照度推理完成
#   step2_nm_inference.done - 正常推理完成
#   step3_visualization.done - 可视化完成
#
# 用法:
#   bash tools/run_comparison.sh          # 正常运行（自动跳过已完成步骤）
#   bash tools/run_comparison.sh --clean  # 清除所有标记，全部重跑
#   bash tools/run_comparison.sh --status # 查看当前进度
# =============================================================================

set -e

# ── 环境检查 ──
if ! command -v python &> /dev/null; then
    echo "[错误] 找不到 python 命令！"
    echo "  请确认您已经激活了含有 BEVFormer 相关依赖的虚拟环境！"
    echo "  例如: conda activate bevformer"
    exit 1
fi

# ── 加载配置 ──
source tools/comparison_env.sh

CONFIG="projects/configs/bevformer/bevformer_tiny.py"
CHECKPOINT="ckpts/bevformer_tiny_epoch_24.pth"

# ── 状态管理 ──
STATUS_DIR=".pipeline_status"
mkdir -p "${STATUS_DIR}"

mark_done()  { touch "${STATUS_DIR}/$1.done"; }
is_done()    { [ -f "${STATUS_DIR}/$1.done" ]; }
clear_step() { rm -f "${STATUS_DIR}/$1.done"; }

# ── 命令行参数 ──
if [ "$1" = "--clean" ]; then
    echo "清除所有任务标记，将重新运行所有步骤..."
    rm -rf "${STATUS_DIR}"
    mkdir -p "${STATUS_DIR}"
    echo "已清除。"
fi

if [ "$1" = "--status" ]; then
    echo "============================================================"
    echo "  流程进度状态"
    echo "============================================================"
    for step in step0_data_prep step1_ll_inference step2_nm_inference step3_visualization step4_domain_analysis step5_metrics_comparison; do
        if is_done "$step"; then
            status="✅ 已完成"
            ts=$(stat -c '%y' "${STATUS_DIR}/${step}.done" 2>/dev/null | cut -d. -f1)
            echo "  ${step}: ${status} (${ts})"
        else
            echo "  ${step}: ⬜ 未完成"
        fi
    done
    echo "============================================================"
    echo ""
    echo "提示: 运行 'bash tools/run_comparison.sh --clean' 可清除所有标记重跑"
    echo "提示: 删除单个标记文件可重跑对应步骤，如:"
    echo "      rm ${STATUS_DIR}/step1_ll_inference.done"
    exit 0
fi

# ── 开始 ──
echo "============================================================"
echo "  BEVFormer 低照度 vs 正常检测 对比可视化流程"
echo "============================================================"
echo "  Low-Light Root: ${LL_ROOT}"
echo "  Normal Root:    ${NM_ROOT}"
echo "  NuScenes Ver:   ${NUSC_VERSION}"
echo "  Checkpoint:     ${CHECKPOINT}"
echo "============================================================"

# 显示当前进度
TOTAL_STEPS=5
DONE_COUNT=0
for step in step0_data_prep step1_ll_inference step2_nm_inference step3_visualization step4_domain_analysis step5_metrics_comparison; do
    is_done "$step" && DONE_COUNT=$((DONE_COUNT + 1))
done
echo "  当前进度: ${DONE_COUNT}/${TOTAL_STEPS} 步骤已完成"
echo "============================================================"

# ═════════════════════════════════════════════════════════════════════════
# Step 0: 准备低照度数据集
# ═════════════════════════════════════════════════════════════════════════
echo ""
if is_done "step0_data_prep"; then
    echo "[Step 0/4] ✅ 数据准备已完成，跳过"
else
    echo "[Step 0/4] 准备低照度数据集元数据..."
    echo ""

    # 低照度数据集只有修改后的 samples/ 图像，
    # 其他元数据共享原始数据集，需要创建 symlinks

    LINK_ITEMS="v1.0-mini sweeps maps lidarseg"
    for item in ${LINK_ITEMS}; do
        if [ ! -e "${LL_ROOT}/${item}" ] && [ -e "${NM_ROOT}/${item}" ]; then
            echo "  创建符号链接: ${LL_ROOT}/${item} -> ${NM_ROOT}/${item}"
            ln -sf "${NM_ROOT}/${item}" "${LL_ROOT}/${item}"
        elif [ -e "${LL_ROOT}/${item}" ]; then
            echo "  已存在: ${LL_ROOT}/${item}"
        else
            echo "  [警告] 源不存在: ${NM_ROOT}/${item}"
        fi
    done

    # LIDAR_TOP 和 RADAR
    if [ ! -e "${LL_ROOT}/samples/LIDAR_TOP" ] && [ -e "${NM_ROOT}/samples/LIDAR_TOP" ]; then
        echo "  创建符号链接: samples/LIDAR_TOP"
        ln -sf "${NM_ROOT}/samples/LIDAR_TOP" "${LL_ROOT}/samples/LIDAR_TOP"
    fi
    for radar in RADAR_FRONT RADAR_FRONT_LEFT RADAR_FRONT_RIGHT RADAR_BACK_LEFT RADAR_BACK_RIGHT; do
        if [ ! -e "${LL_ROOT}/samples/${radar}" ] && [ -e "${NM_ROOT}/samples/${radar}" ]; then
            ln -sf "${NM_ROOT}/samples/${radar}" "${LL_ROOT}/samples/${radar}"
        fi
    done

    # can_bus
    CANBUS_PATH="${NM_ROOT}"
    if [ -d "${NM_ROOT}/can_bus" ]; then
        CANBUS_PATH="${NM_ROOT}"
        [ ! -e "${LL_ROOT}/can_bus" ] && ln -sf "${NM_ROOT}/can_bus" "${LL_ROOT}/can_bus"
    elif [ -d "datasets/can_bus" ]; then
        CANBUS_PATH="datasets"
        [ ! -e "${LL_ROOT}/can_bus" ] && ln -sf "$(pwd)/datasets/can_bus" "${LL_ROOT}/can_bus"
    fi

    echo "  元数据链接完成"

    # 生成低照度 pkl
    if [ ! -f "${LL_ROOT}/nuscenes_infos_temporal_val.pkl" ]; then
        echo "  生成低照度数据集 pkl ..."
        python tools/create_data.py nuscenes \
            --root-path "${LL_ROOT}" \
            --canbus "${CANBUS_PATH}" \
            --out-dir "${LL_ROOT}" \
            --extra-tag nuscenes \
            --version "${NUSC_VERSION}"
        echo "  低照度 pkl 完成"
    else
        echo "  低照度 pkl 已存在"
    fi

    # 生成正常 pkl
    if [ ! -f "${NM_ROOT}/nuscenes_infos_temporal_val.pkl" ]; then
        echo "  生成正常数据集 pkl ..."
        NM_CANBUS="${NM_ROOT}"
        [ -d "datasets/can_bus" ] && [ ! -d "${NM_ROOT}/can_bus" ] && NM_CANBUS="datasets"
        python tools/create_data.py nuscenes \
            --root-path "${NM_ROOT}" \
            --canbus "${NM_CANBUS}" \
            --out-dir "${NM_ROOT}" \
            --extra-tag nuscenes \
            --version "${NUSC_VERSION}"
        echo "  正常 pkl 完成"
    else
        echo "  正常 pkl 已存在"
    fi

    mark_done "step0_data_prep"
    echo "  [Step 0] ✅ 完成"
fi

# ═════════════════════════════════════════════════════════════════════════
# Step 1: 低照度数据集推理
# ═════════════════════════════════════════════════════════════════════════
echo ""
if is_done "step1_ll_inference"; then
    echo "[Step 1/4] ✅ 低照度推理已完成，跳过"
    echo "  结果文件: test/lowlight_detection/results_nusc.json"
else
    echo "[Step 1/4] 在低照度数据集上运行推理..."
    echo "  Dataset: ${LL_ROOT}"
    echo ""

    python tools/test.py ${CONFIG} ${CHECKPOINT} \
        --eval bbox \
        --cfg-options \
            "data.test.data_root=${LL_ROOT}/" \
            "data.test.ann_file=${LL_ROOT}/nuscenes_infos_temporal_val.pkl"

    LL_RESULT=$(find test/bevformer_tiny -name 'results_nusc.json' -type f -printf '%T@ %p\n' | sort -n | tail -1 | awk '{print $2}')
    mkdir -p test/lowlight_detection
    cp "${LL_RESULT}" test/lowlight_detection/results_nusc.json

    # 同时复制 metrics 如果存在
    LL_METRICS_DIR=$(dirname "${LL_RESULT}")
    [ -f "${LL_METRICS_DIR}/metrics_summary.json" ] && \
        cp "${LL_METRICS_DIR}/metrics_summary.json" test/lowlight_detection/
    [ -f "${LL_METRICS_DIR}/metrics_details.json" ] && \
        cp "${LL_METRICS_DIR}/metrics_details.json" test/lowlight_detection/

    mark_done "step1_ll_inference"
    echo "  [Step 1] ✅ 完成: test/lowlight_detection/results_nusc.json"
fi

# ═════════════════════════════════════════════════════════════════════════
# Step 2: 正常数据集推理
# ═════════════════════════════════════════════════════════════════════════
echo ""
if is_done "step2_nm_inference"; then
    echo "[Step 2/4] ✅ 正常推理已完成，跳过"
    echo "  结果文件: test/normal_detection/results_nusc.json"
else
    echo "[Step 2/4] 在正常数据集上运行推理..."
    echo "  Dataset: ${NM_ROOT}"
    echo ""

    python tools/test.py ${CONFIG} ${CHECKPOINT} \
        --eval bbox \
        --cfg-options \
            "data.test.data_root=${NM_ROOT}/" \
            "data.test.ann_file=${NM_ROOT}/nuscenes_infos_temporal_val.pkl"

    NM_RESULT=$(find test/bevformer_tiny -name 'results_nusc.json' -type f -printf '%T@ %p\n' | sort -n | tail -1 | awk '{print $2}')
    mkdir -p test/normal_detection
    cp "${NM_RESULT}" test/normal_detection/results_nusc.json

    NM_METRICS_DIR=$(dirname "${NM_RESULT}")
    [ -f "${NM_METRICS_DIR}/metrics_summary.json" ] && \
        cp "${NM_METRICS_DIR}/metrics_summary.json" test/normal_detection/
    [ -f "${NM_METRICS_DIR}/metrics_details.json" ] && \
        cp "${NM_METRICS_DIR}/metrics_details.json" test/normal_detection/

    mark_done "step2_nm_inference"
    echo "  [Step 2] ✅ 完成: test/normal_detection/results_nusc.json"
fi

# ═════════════════════════════════════════════════════════════════════════
# Step 3: 生成对比可视化
# ═════════════════════════════════════════════════════════════════════════
echo ""
if is_done "step3_visualization"; then
    echo "[Step 3/4] ✅ 可视化已完成，跳过"
    echo "  输出目录: runs/visual_comparison/"
else
    echo "[Step 3/4] 生成论文对比可视化图..."
    echo ""

    # 检查依赖的推理结果是否存在
    if [ ! -f "test/lowlight_detection/results_nusc.json" ]; then
        echo "  [错误] 低照度推理结果不存在！请先完成 Step 1"
        echo "  提示: rm ${STATUS_DIR}/step1_ll_inference.done && bash tools/run_comparison.sh"
        exit 1
    fi
    if [ ! -f "test/normal_detection/results_nusc.json" ]; then
        echo "  [错误] 正常推理结果不存在！请先完成 Step 2"
        echo "  提示: rm ${STATUS_DIR}/step2_nm_inference.done && bash tools/run_comparison.sh"
        exit 1
    fi

    # 单视角 CAM_FRONT
    python tools/analysis_tools/visualize_comparison.py \
        --lowlight-results  test/lowlight_detection/results_nusc.json \
        --normal-results    test/normal_detection/results_nusc.json \
        --lowlight-dataroot "${LL_ROOT}" \
        --normal-dataroot   "${NM_ROOT}" \
        --nusc-version      "${NUSC_VERSION}" \
        --out-dir           runs/visual_comparison/front_only \
        --num-samples       10 \
        --score-thresh      0.25 \
        --front-only

    # 全部6相机
    python tools/analysis_tools/visualize_comparison.py \
        --lowlight-results  test/lowlight_detection/results_nusc.json \
        --normal-results    test/normal_detection/results_nusc.json \
        --lowlight-dataroot "${LL_ROOT}" \
        --normal-dataroot   "${NM_ROOT}" \
        --nusc-version      "${NUSC_VERSION}" \
        --out-dir           runs/visual_comparison/all_cams \
        --num-samples       5 \
        --score-thresh      0.25 \
        --all-cams

    mark_done "step3_visualization"
    echo "  [Step 3] ✅ 完成"
fi

# ═════════════════════════════════════════════════════════════════════════
# Step 4: 图像域分析 (直方图 + t-SNE)
# ═════════════════════════════════════════════════════════════════════════
echo ""
if is_done "step4_domain_analysis"; then
    echo "[Step 4/6] ✅ 图像域分析已完成，跳过"
    echo "  输出目录: runs/domain_analysis/"
else
    echo "[Step 4/6] 生成图像域分析图 (直方图 + t-SNE)..."
    echo ""

    python tools/analysis_tools/analyze_image_domains.py \
        --lowlight-dataroot "${LL_ROOT}" \
        --normal-dataroot   "${NM_ROOT}" \
        --nusc-version      "${NUSC_VERSION}" \
        --out-dir           runs/domain_analysis \
        --num-normal        20 \
        --num-night         20 \
        --camera            CAM_FRONT

    mark_done "step4_domain_analysis"
    echo "  [Step 4] ✅ 完成"
fi

# ═════════════════════════════════════════════════════════════════════════
# Step 5: 学术指标对比与可视化 (mAP, NDS 等)
# ═════════════════════════════════════════════════════════════════════════
echo ""
if is_done "step5_metrics_comparison"; then
    echo "[Step 5/6] ✅ 学术指标对比可视化已完成，跳过"
    echo "  输出目录: runs/metrics_comparison/"
else
    echo "[Step 5/6] 生成详细学术指标对比图表 (mAP, NDS, mATE等)..."
    echo ""
    
    if [ ! -f "test/lowlight_detection/metrics_summary.json" ] || [ ! -f "test/normal_detection/metrics_summary.json" ]; then
        echo "  [警告] metrics_summary.json 缺失，无法生成学术指标图表。"
        echo "  请确认 Step 1 和 Step 2 正常生成了性能评估文件。"
    else
        python tools/analysis_tools/compare_metrics.py \
            --ll-metrics test/lowlight_detection/metrics_summary.json \
            --nm-metrics test/normal_detection/metrics_summary.json \
            --out-dir runs/metrics_comparison
            
        mark_done "step5_metrics_comparison"
        echo "  [Step 5] ✅ 完成"
    fi
fi

# ═════════════════════════════════════════════════════════════════════════
# Step 6: 部署 Web 查看器
# ═════════════════════════════════════════════════════════════════════════
echo ""
echo "[Step 6/6] 部署交互式 Web 查看器..."

# 复制 viewer.html 到输出目录
VIEWER_SRC="tools/analysis_tools/viewer.html"
for vis_dir in runs/visual_comparison/front_only runs/visual_comparison/all_cams; do
    if [ -d "${vis_dir}" ]; then
        cp "${VIEWER_SRC}" "${vis_dir}/index.html"
        echo "  已部署: ${vis_dir}/index.html"
    fi
done

echo ""
echo "============================================================"
echo "  🎉 全部完成!"
echo "============================================================"
echo ""
echo "  📁 检测对比图:"
echo "    单视角: runs/visual_comparison/front_only/"
echo "    全视角: runs/visual_comparison/all_cams/"
echo ""
echo "  📊 图像域分析 (直方图/t-SNE):"
echo "    runs/domain_analysis/"
echo "    ├── histogram_grayscale.pdf  — 灰度直方图"
echo "    ├── histogram_rgb.pdf        — RGB通道直方图"
echo "    ├── tsne_comparison.pdf      — t-SNE散点图"
echo "    ├── tsne_density.pdf         — t-SNE+密度等高线"
echo "    ├── combined_analysis.pdf    — 2×2组合图"
echo "    └── statistics.json          — 统计数据"
echo ""
echo "  📈 学术指标对比 (mAP, NDS):"
echo "    runs/metrics_comparison/"
echo "    ├── metrics_comparison_table.csv — 详细指标对比表格"
echo "    ├── metrics_comparison_table.md  — Markdown版对应表格"
echo "    ├── radar_global_metrics.pdf     — 全局指标雷达图"
echo "    └── bar_per_class_map.pdf        — 各类别mAP柱状图"
echo ""
echo "  🌐 启动 Web 查看器:"
echo "    cd runs/visual_comparison/front_only && python -m http.server 8080"
echo "    然后打开: http://localhost:8080"
echo ""
echo "  🔧 管理命令:"
echo "    查看进度:     bash tools/run_comparison.sh --status"
echo "    全部重跑:     bash tools/run_comparison.sh --clean"
echo "    重跑某步骤:   rm .pipeline_status/stepN_xxx.done"
echo "============================================================"
