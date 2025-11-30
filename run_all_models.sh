#!/bin/bash
##############################################################################
# VMEvalKit - Unified Open-Source Models Inference Runner
# 
# This script runs all open-source models on the complete dataset
# - 1,643 questions across 11 task types
# - 16 open-source models (excludes wan-2.1-vace-1.3b - doesn't exist)
# - Total: ~26,288 video generations (1,643 × 16)
#
# Hardware: 8x NVIDIA H200 (140GB VRAM each)
# Resume: Automatically skips completed tasks
#
# Usage:
#   ./run_all_models.sh              # Sequential (one at a time)
#   ./run_all_models.sh --parallel   # Parallel (use all GPUs)
##############################################################################

# Configuration
OUTPUT_DIR="data/outputs/pilot_experiment"
LOG_DIR="logs/opensource_inference"
mkdir -p "$LOG_DIR"

# Working models (wan-2.1-vace-1.3b removed - doesn't exist on HF)
MODELS=(
    "ltx-video"
    "ltx-video-13b-distilled"
    "svd"
    "hunyuan-video-i2v"
    "videocrafter2-512"
    "dynamicrafter-256"
    "dynamicrafter-512"
    "dynamicrafter-1024"
    "wan"
    "wan-2.1-flf2v-720p"
    "wan-2.2-i2v-a14b"
    "wan-2.1-i2v-480p"
    "wan-2.1-i2v-720p"
    "wan-2.2-ti2v-5b"
    "wan-2.1-vace-14b"
    "morphic-frames-to-video"
)

# Parse arguments
PARALLEL=false
if [ "$1" = "--parallel" ]; then
    PARALLEL=true
fi

echo "════════════════════════════════════════════════════════════════"
echo "         VMEvalKit Open-Source Models Inference Runner"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  • Models: ${#MODELS[@]}"
echo "  • Questions: 1,643"
echo "  • Total: ~26,288 generations"
if [ "$PARALLEL" = true ]; then
    echo "  • Mode: PARALLEL (all GPUs)"
else
    echo "  • Mode: SEQUENTIAL (one model at a time)"
fi
echo "  • Output: ${OUTPUT_DIR}"
echo ""

# Activate venv
cd /home/hokindeng/VMEvalKit
source venv/bin/activate

if [ "$PARALLEL" = true ]; then
    echo "🚀 Starting PARALLEL execution..."
    echo "  Wave 1: 5 lightweight models on GPUs 0-4"
    echo "  Wave 2: 4 medium models on GPUs 0-3"
    echo "  Wave 3: 3 heavy models on GPUs 0-2"
    echo "  Wave 4: 4 very heavy models - 2 at a time"
    echo ""
    
    # Wave 1: Lightweight (5 models)
    echo "🌊 Wave 1: Lightweight models..."
    CUDA_VISIBLE_DEVICES=0 python examples/generate_videos.py --model ltx-video --all-tasks > "${LOG_DIR}/ltx-video.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python examples/generate_videos.py --model ltx-video-13b-distilled --all-tasks > "${LOG_DIR}/ltx-video-13b.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=2 python examples/generate_videos.py --model videocrafter2-512 --all-tasks > "${LOG_DIR}/videocrafter.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=3 python examples/generate_videos.py --model dynamicrafter-256 --all-tasks > "${LOG_DIR}/dynamicrafter-256.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=4 python examples/generate_videos.py --model svd --all-tasks > "${LOG_DIR}/svd.log" 2>&1 &
    wait
    echo "✅ Wave 1 done!"
    
    # Wave 2: Medium (4 models)  
    echo "🌊 Wave 2: Medium models..."
    CUDA_VISIBLE_DEVICES=0 python examples/generate_videos.py --model dynamicrafter-512 --all-tasks > "${LOG_DIR}/dynamicrafter-512.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python examples/generate_videos.py --model dynamicrafter-1024 --all-tasks > "${LOG_DIR}/dynamicrafter-1024.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=2 python examples/generate_videos.py --model hunyuan-video-i2v --all-tasks > "${LOG_DIR}/hunyuan.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=3 python examples/generate_videos.py --model wan-2.1-i2v-480p --all-tasks > "${LOG_DIR}/wan-480p.log" 2>&1 &
    wait
    echo "✅ Wave 2 done!"
    
    # Wave 3: Heavy (3 models)
    echo "🌊 Wave 3: Heavy models..."
    CUDA_VISIBLE_DEVICES=0 python examples/generate_videos.py --model wan-2.2-ti2v-5b --all-tasks > "${LOG_DIR}/wan-ti2v.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python examples/generate_videos.py --model wan-2.1-i2v-720p --all-tasks > "${LOG_DIR}/wan-720p.log" 2>&1 &
    wait
    echo "✅ Wave 3 done!"
    
    # Wave 4: Very heavy (4 models, 2 at a time)
    echo "🌊 Wave 4: Very heavy models (48GB each, 2 at a time)..."
    CUDA_VISIBLE_DEVICES=0 python examples/generate_videos.py --model wan --all-tasks > "${LOG_DIR}/wan.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python examples/generate_videos.py --model wan-2.1-flf2v-720p --all-tasks > "${LOG_DIR}/wan-flf2v.log" 2>&1 &
    wait
    
    CUDA_VISIBLE_DEVICES=0 python examples/generate_videos.py --model wan-2.2-i2v-a14b --all-tasks > "${LOG_DIR}/wan-a14b.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python examples/generate_videos.py --model wan-2.1-vace-14b --all-tasks > "${LOG_DIR}/wan-vace.log" 2>&1 &
    wait
    echo "✅ Wave 4 done!"
    
    # Wave 5: Morphic (all GPUs)
    echo "🌊 Wave 5: Morphic (distributed across all 8 GPUs)..."
    python examples/generate_videos.py --model morphic-frames-to-video --all-tasks > "${LOG_DIR}/morphic.log" 2>&1
    echo "✅ Wave 5 done!"
    
else
    echo "🚀 Starting SEQUENTIAL execution..."
    echo ""
    
    COMPLETED=0
    FAILED=0
    
    for i in "${!MODELS[@]}"; do
        MODEL="${MODELS[$i]}"
        NUM=$((i + 1))
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Model ${NUM}/${#MODELS[@]}: ${MODEL}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Assign GPU (rotate)
        GPU=$((i % 8))
        
        # Run
        CUDA_VISIBLE_DEVICES=${GPU} python examples/generate_videos.py \
            --model "${MODEL}" \
            --all-tasks \
            2>&1 | tee "${LOG_DIR}/${MODEL}_gpu${GPU}.log"
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            COMPLETED=$((COMPLETED + 1))
            echo "✅ Completed: ${MODEL}"
        else
            FAILED=$((FAILED + 1))
            echo "❌ Failed: ${MODEL}"
        fi
        
        echo "Progress: ${COMPLETED} done, ${FAILED} failed, $((${#MODELS[@]} - NUM)) remaining"
    done
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "                    ✅ ALL DONE!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Outputs: ${OUTPUT_DIR}"
echo "Logs: ${LOG_DIR}"

