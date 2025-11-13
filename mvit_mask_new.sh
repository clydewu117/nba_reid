#!/bin/bash

# CAM batch processing for NBA videos
# Only process videos where split=test from CSV
# Usage: ./mvit_app.sh [GPU_ID]

# GPU selection parameter (default: 0)
GPU_ID=${1:-0}

BASE_DIR="/fs/scratch/PAS3184/v3/appearance"
SCRIPT_DIR="/users/PAS2099/clydewu117/nba_reid"
OUTPUT_BASE="/fs/scratch/PAS3184/baicheng_cam"

# NEW: CSV containing split info
CSV_FILE="/fs/scratch/PAS3184/v3/train_test_split.csv"

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Checkpoints
CHECKPOINTS=(
    "/fs/scratch/PAS3184/baicheng_ckpt/new2/mask_k400_16frames_beginning_nosplitsampling_drop02/best_model.pth"
    "/fs/scratch/PAS3184/baicheng_ckpt/new2/mask_k400_16frames_beginning_splitsampling_drop02/best_model.pth"
)

CHECKPOINT_NAMES=(
    "mask_k400_16frames_beginning_nosplitsampling_drop02"
    "mask_k400_16frames_beginning_splitsampling_drop02"
)

# NEW: read PLAYERS from CSV (only split=test)
mapfile -t PLAYERS < <(awk -F',' '$1=="test" {print $3}' "$CSV_FILE" | sort -u)

# NEW: read VIDEOS from CSV (only split=test)
mapfile -t TEST_VIDEO_PATHS < <(awk -F',' '$1=="test" {print $6}' "$CSV_FILE")

# CAM methods
CAM_METHODS=("originalcam" "scorecam")

echo "=========================================="
echo "CAM Batch Processing - Test Split Only"
echo "GPU: $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo "Players(test): ${#PLAYERS[@]}"
echo "Videos(test): ${#TEST_VIDEO_PATHS[@]}"
echo "CAM methods: ${#CAM_METHODS[@]} (originalcam, scorecam)"
echo "Output: $OUTPUT_BASE"
echo "=========================================="
echo ""

mkdir -p "$OUTPUT_BASE"

# counters
total_runs=0
success_count=0
fail_count=0

# Loop checkpoints
for ckpt_idx in "${!CHECKPOINTS[@]}"; do
    CHECKPOINT="${CHECKPOINTS[$ckpt_idx]}"
    CHECKPOINT_NAME="${CHECKPOINT_NAMES[$ckpt_idx]}"

    echo "========================================"
    echo "CHECKPOINT [$((ckpt_idx+1))/${#CHECKPOINTS[@]}]: $CHECKPOINT_NAME"
    echo "========================================"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "ERROR: Checkpoint not found: $CHECKPOINT"
        echo "Skipping..."
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

        # NEW: loop all test video paths belonging to this player
        for VIDEO_PATH in "${TEST_VIDEO_PATHS[@]}"; do

            # Filter video-paths by the player name
            if [[ "$VIDEO_PATH" != *"/$player/"* ]]; then
                continue
            fi

            # Extract video number
            VIDEO_NAME=$(basename "$VIDEO_PATH")
            VIDEO_NUM="${VIDEO_NAME%.*}"

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
                elif [ "$method" == "scorecam_finer" ]; then
                    METHOD_FLAG="scorecam"
                    EXTRA_FLAGS="--finer"
                    METHOD_NAME="scorecam_finer"
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
echo "  Failed: $fail_count"
echo "Output directory: $OUTPUT_BASE"
echo "=========================================="
