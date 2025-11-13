#!/bin/bash

# CAM batch processing for NBA videos
# Processes only videos where split=test (from CSV)
# Usage: ./mvit_app.sh [GPU_ID]

# GPU selection parameter (default: 0)
GPU_ID=${1:-0}

BASE_DIR="/fs/scratch/PAS3184/v3/appearance"
SCRIPT_DIR="/users/PAS2099/clydewu117/nba_reid"
OUTPUT_BASE="/fs/scratch/PAS3184/baicheng_cam"

# ==== NEW: CSV file containing split information ====
CSV_FILE="/fs/scratch/PAS3184/v3/train_test_split.csv"   # ← 已更新为你提供的路径

# Set GPU device
export CUDA_VISIBLE_DEVICES=$GPU_ID

# List of checkpoints (保持原样)
CHECKPOINTS=(
    "/fs/scratch/PAS3184/baicheng_ckpt/new2/appearance_k400_16frames_beginning_nosplitsampling_drop02/best_model.pth"
    "/fs/scratch/PAS3184/baicheng_ckpt/new2/color_appearance_k400_16frames_beginning_nosplitsampling_drop03/best_model.pth"
)

CHECKPOINT_NAMES=(
    "appearance_k400_16frames_beginning_nosplitsampling_drop02"
    "color_appearance_k400_16frames_beginning_nosplitsampling_drop03"
)

# ==== NEW: 从 CSV 读取 split=test 的视频和玩家 ====

# 获取所有 split=test 的 video_path
mapfile -t TEST_VIDEO_PATHS < <(awk -F',' '$1=="test" {print $6}' "$CSV_FILE")

# 获取 split=test 的 identity_folder（作为玩家）
mapfile -t PLAYERS < <(awk -F',' '$1=="test" {print $3}' "$CSV_FILE" | sort -u)

# CAM methods (保持原样)
CAM_METHODS=("originalcam" "scorecam")


echo "=========================================="
echo "CAM Batch Processing - Test Split Only"
echo "GPU: $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo "Players in test split: ${#PLAYERS[@]}"
echo "Total test videos: ${#TEST_VIDEO_PATHS[@]}"
echo "CAM methods: ${#CAM_METHODS[@]}"
echo "Output: $OUTPUT_BASE"
echo "=========================================="
echo ""

mkdir -p "$OUTPUT_BASE"

total_runs=0
success_count=0
fail_count=0

# Loop through checkpoints
for ckpt_idx in "${!CHECKPOINTS[@]}"; do
    CHECKPOINT="${CHECKPOINTS[$ckpt_idx]}"
    CHECKPOINT_NAME="${CHECKPOINT_NAMES[$ckpt_idx]}"

    echo "========================================"
    echo "CHECKPOINT [$((ckpt_idx+1))/${#CHECKPOINTS[@]}]: $CHECKPOINT_NAME"
    echo "========================================"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "ERROR: Checkpoint not found: $CHECKPOINT"
        echo ""
        continue
    fi

    CHECKPOINT_OUTPUT_DIR="$OUTPUT_BASE/$CHECKPOINT_NAME"
    mkdir -p "$CHECKPOINT_OUTPUT_DIR"

    # Loop players
    for player in "${PLAYERS[@]}"; do
        echo "----------------------------------------"
        echo "Processing player: $player"
        echo "----------------------------------------"

        PLAYER_DIR="$BASE_DIR/$player/freethrow"
        PLAYER_OUTPUT_DIR="$CHECKPOINT_OUTPUT_DIR/$player"
        mkdir -p "$PLAYER_OUTPUT_DIR"

        # 遍历所有 test 视频，只处理属于该 player 的
        for VIDEO_PATH in "${TEST_VIDEO_PATHS[@]}"; do

            # 检查此视频是否属于该 player
            if [[ "$VIDEO_PATH" != *"/$player/"* ]]; then
                continue
            fi

            VIDEO_NAME=$(basename "$VIDEO_PATH")
            VIDEO_NUM="${VIDEO_NAME%.*}"  # 009.mp4 → 009

            echo "  Test Video: $VIDEO_PATH"

            # Loop CAM methods
            for method in "${CAM_METHODS[@]}"; do

                if [ "$method" == "originalcam" ]; then
                    METHOD_FLAG="originalcam"
                    EXTRA_FLAGS=""
                    METHOD_NAME="originalcam"
                elif [ "$method" == "scorecam" ]; then
                    METHOD_FLAG="scorecam"
                    EXTRA_FLAGS=""
                    METHOD_NAME="scorecam"
                fi

                OUTPUT_DIR="$PLAYER_OUTPUT_DIR/video_${VIDEO_NUM}_${METHOD_NAME}"

                echo "    Method: $METHOD_NAME"

                python3 "$SCRIPT_DIR/run_gradcam_mvit.py" \
                    --video "$VIDEO_PATH" \
                    --checkpoint "$CHECKPOINT" \
                    --output "$OUTPUT_DIR" \
                    --method "$METHOD_FLAG" \
                    --frames 16 \
                    --device cuda \
                    $EXTRA_FLAGS

                if [ $? -eq 0 ]; then
                    echo "    ✓ Success: $OUTPUT_DIR"
                    ((success_count++))
                else
                    echo "    ✗ Failed: $METHOD_NAME"
                    ((fail_count++))
                fi

                ((total_runs++))
            done

            echo ""
        done

        echo "Completed player: $player"
        echo ""
    done

    echo "Completed checkpoint: $CHECKPOINT_NAME"
    echo ""
done

echo "=========================================="
echo "Batch processing complete!"
echo "=========================================="
echo "Total runs: $total_runs"
echo "  Success: $success_count"
echo "  Failed:  $fail_count"
echo "Output directory: $OUTPUT_BASE"
echo "=========================================="
