#!/bin/bash

# CAM batch processing for NBA videos using UniFormerV2
# Processes players with mask videos, 3 checkpoints, 2 CAM methods
# Usage: ./uni_mask.sh [GPU_ID]
# Example: ./uni_mask.sh 0  (use GPU 0)
#          ./uni_mask.sh     (use default GPU 0)

# GPU selection parameter (default: 0)
GPU_ID=${1:-0}

BASE_DIR="/home/zhang.13617/Desktop/zhang.13617/NBA/mask"
SCRIPT_DIR="/home/zhang.13617/Desktop/clean"
OUTPUT_BASE="/home/zhang.13617/Desktop/zhang.13617/NBA/uni_full"

# Set GPU device
export CUDA_VISIBLE_DEVICES=$GPU_ID

# List of 3 checkpoints for testing (mask models)
CHECKPOINTS=(
    "/home/zhang.13617/Desktop/zhang.13617/NBA/ckpt/uni/mask_k400_16frames_beginning_nosplitsampling/best_model.pth"
)

# Checkpoint names for output directories
CHECKPOINT_NAMES=(
    "mask_k400_16frames_beginning_nosplitsampling"
)

# List of players for testing
PLAYERS=(
    "Aaron Gordon"
    "Alex_Caruso"
    "Al_Horford"
    "Devin Booker"
    "Anthony Davis"
    "Bradley Beal"
    "Chris Paul"
    "Damian Lillard"
    "DeMar DeRozan"
    "Draymond Green"
    "Jimmy Butler"
    "Kawhi Leonard"
    "Kevin Durant"
    "Klay Thompson"
    "LeBron James"
)

# CAM methods to test
# Method 1: OriginalCAM (gradient-free)
# Method 2: ScoreCAM (gradient-free, perturbation-based)
CAM_METHODS=("originalcam" "scorecam")

echo "=========================================="
echo "UniFormerV2 CAM Batch Processing"
echo "GPU: $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo "Players: ${#PLAYERS[@]}"
echo "Videos per player: 10"
echo "CAM methods: ${#CAM_METHODS[@]} (originalcam, scorecam)"
echo "Output: $OUTPUT_BASE"
echo "=========================================="
echo ""

# Create output base directory if it doesn't exist
mkdir -p "$OUTPUT_BASE"

# Counter for total processed videos
total_runs=0
success_count=0
fail_count=0

# Loop through each checkpoint
for ckpt_idx in "${!CHECKPOINTS[@]}"; do
    CHECKPOINT="${CHECKPOINTS[$ckpt_idx]}"
    CHECKPOINT_NAME="${CHECKPOINT_NAMES[$ckpt_idx]}"

    echo "========================================"
    echo "CHECKPOINT [$((ckpt_idx+1))/${#CHECKPOINTS[@]}]: $CHECKPOINT_NAME"
    echo "========================================"

    # Check if checkpoint exists
    if [ ! -f "$CHECKPOINT" ]; then
        echo "ERROR: Checkpoint not found: $CHECKPOINT"
        echo "Skipping this checkpoint"
        echo ""
        continue
    fi

    # Get config file from checkpoint directory
    CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
    CONFIG_PATH="$CHECKPOINT_DIR/config.yaml"

    if [ ! -f "$CONFIG_PATH" ]; then
        echo "ERROR: Config not found: $CONFIG_PATH"
        echo "Skipping this checkpoint"
        echo ""
        continue
    fi

    # Create output directory for this checkpoint
    CHECKPOINT_OUTPUT_DIR="$OUTPUT_BASE/$CHECKPOINT_NAME"
    mkdir -p "$CHECKPOINT_OUTPUT_DIR"

    # Loop through each player
    for player in "${PLAYERS[@]}"; do
        echo "----------------------------------------"
        echo "Processing player: $player"
        echo "----------------------------------------"

        PLAYER_DIR="$BASE_DIR/$player/freethrow"

        # Check if player directory exists
        if [ ! -d "$PLAYER_DIR" ]; then
            echo "WARNING: Directory not found: $PLAYER_DIR"
            echo "Skipping $player"
            echo ""
            continue
        fi

        # Create output directory for this player
        PLAYER_OUTPUT_DIR="$CHECKPOINT_OUTPUT_DIR/$player"
        mkdir -p "$PLAYER_OUTPUT_DIR"

        # Process 10 videos (000.mp4 to 009.mp4)
        for i in {0..9}; do
            VIDEO_NUM=$(printf "%03d" $i)
            VIDEO_PATH="$PLAYER_DIR/${VIDEO_NUM}.mp4"

            # Check if video exists
            if [ ! -f "$VIDEO_PATH" ]; then
                echo "  WARNING: Video not found: $VIDEO_PATH"
                continue
            fi

            echo "  Video [${i}/9]: $VIDEO_PATH"

            # Loop through each CAM method
            for method in "${CAM_METHODS[@]}"; do
                # Determine method parameters
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

                # Create output directory for this specific video and method
                OUTPUT_DIR="$PLAYER_OUTPUT_DIR/video_${VIDEO_NUM}_${METHOD_NAME}"

                echo "    Method: $METHOD_NAME"

                # Run the UniFormerV2 CAM script (using checkpoint's config.yaml)
                python3 "$SCRIPT_DIR/run_gradcam_uniformer.py" \
                    --video "$VIDEO_PATH" \
                    --checkpoint "$CHECKPOINT" \
                    --config "$CONFIG_PATH" \
                    --output "$OUTPUT_DIR" \
                    --method "$METHOD_FLAG" \
                    --frames 16 \
                    --device cuda \
                    --scorecam_batch_size 64 \
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
echo "Total runs: $total_runs (1 checkpoint × 15 players × 10 videos × 2 methods = 300 runs)"
echo "  Success: $success_count"
echo "  Failed: $fail_count"
echo "Output directory: $OUTPUT_BASE"
echo "=========================================="
