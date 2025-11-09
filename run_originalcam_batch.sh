#!/bin/bash

# Original CAM batch processing for NBA videos
# Processes 50 players, 6 videos each (freethrow)
# Usage: ./run_originalcam_batch.sh [GPU_ID]
# Example: ./run_originalcam_batch.sh 0  (use GPU 0)
#          ./run_originalcam_batch.sh     (use default GPU 0)

# GPU selection parameter (default: 0)
GPU_ID=${1:-0}

BASE_DIR="/home/zhang.13617/Desktop/zhang.13617/NBA/app"
SCRIPT_DIR="/home/zhang.13617/Desktop/clean"
CHECKPOINT="/home/zhang.13617/Desktop/zhang.13617/NBA/ckpt/mvit_app_model.pth"
OUTPUT_BASE="/home/zhang.13617/Desktop/zhang.13617/NBA/visual"

# Set GPU device
export CUDA_VISIBLE_DEVICES=$GPU_ID

# List of 50 players
PLAYERS=(
    "Aaron Gordon"
    "Alex_Caruso"
    "Al_Horford"
    "Alperen_Sengun"
    "Andrew Wiggins"
    "Anthony Edwards"
    "Austin Reaves"
    "Bam Adebayo"
    "Bobby_Portis"
    "Bogdan_Bogdanović"
    "Brandon_Miller"
    "Brook_Lopez"
    "Cade_Cunningham"
    "Cameron_Johnson"
    "Chet_Holmgren"
    "Clint_Capela"
    "Coby_White"
    "Cole_Anthony"
    "Collin_Sexton"
    "Damian_Lillard"
    "Daniel_Gafford"
    "Dante_DiVincenzo"
    "Darius_Garland"
    "Dejounte_Murray"
    "Dereck_Lively_II"
    "Derrick_White"
    "Desmond_Bane"
    "Devin_Booker"
    "De'Aaron_Fox"
    "Dillon_Brooks"
    "Domantas_Sabonis"
    "Donovan_Mitchell"
    "Donte_DiVincenzo"
    "Draymond_Green"
    "Evan_Mobley"
    "Franz_Wagner"
    "Giannis_Antetokounmpo"
    "Gradey_Dick"
    "Grayson_Allen"
    "Jaime_Jaquez_Jr"
    "Jalen_Brunson"
    "Jalen_Duren"
    "Jalen_Green"
    "Jalen_Johnson"
    "Jalen_Suggs"
    "Jalen_Williams"
    "Jaren_Jackson_Jr"
    "Jarrett_Allen"
    "Jaylen_Brown"
    "Jayson_Tatum"
)

echo "=========================================="
echo "Original CAM Batch Processing"
echo "GPU: $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "Players: ${#PLAYERS[@]}"
echo "Videos per player: 6"
echo "Output: $OUTPUT_BASE"
echo "=========================================="
echo ""

# Create output base directory if it doesn't exist
mkdir -p "$OUTPUT_BASE"

# Counter for total processed videos
total_videos=0
success_count=0
fail_count=0

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
    PLAYER_OUTPUT_DIR="$OUTPUT_BASE/$player"
    mkdir -p "$PLAYER_OUTPUT_DIR"

    # Process 6 videos (000.mp4 to 005.mp4)
    for i in {0..5}; do
        VIDEO_NUM=$(printf "%03d" $i)
        VIDEO_PATH="$PLAYER_DIR/${VIDEO_NUM}.mp4"

        # Check if video exists
        if [ ! -f "$VIDEO_PATH" ]; then
            echo "  WARNING: Video not found: $VIDEO_PATH"
            continue
        fi

        # Create output directory for this specific video
        OUTPUT_DIR="$PLAYER_OUTPUT_DIR/video_${VIDEO_NUM}"

        echo "  [${i}/5] Processing: $VIDEO_PATH"

        # Run the CAM script with Original CAM method (default)
        python3 "$SCRIPT_DIR/run_gradcam_mvit.py" \
            --video "$VIDEO_PATH" \
            --checkpoint "$CHECKPOINT" \
            --output "$OUTPUT_DIR" \
            --method originalcam \
            --frames 16 \
            --device cuda

        if [ $? -eq 0 ]; then
            echo "  ✓ Success: $OUTPUT_DIR"
            ((success_count++))
        else
            echo "  ✗ Failed: $VIDEO_PATH"
            ((fail_count++))
        fi

        ((total_videos++))
        echo ""
    done

    echo "Completed player: $player"
    echo ""
done

echo "=========================================="
echo "Batch processing complete!"
echo "=========================================="
echo "Total videos processed: $total_videos"
echo "  Success: $success_count"
echo "  Failed: $fail_count"
echo "Output directory: $OUTPUT_BASE"
echo "=========================================="
