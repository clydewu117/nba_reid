python batch_videomaev2_masked_cam.py \
    --csv /fs/scratch/PAS3184/v3/train_test_split_selected.csv \
    --appearance-config /users/PAS2985/cz2128/ReID/nba_reid/outputs/v3/VideoMAEv2_appearance_beginning_nosplitsampling-2/config.yaml \
    --mask-config /users/PAS2985/cz2128/ReID/nba_reid/outputs/v3/VideoMAEv2_mask_beginning_nosplitsampling/config.yaml \
    --appearance-checkpoints /users/PAS2985/cz2128/ReID/nba_reid/outputs/v3/VideoMAEv2_appearance_beginning_nosplitsampling-2/VideoMAEv2_appearance_beginning_nosplitsampling.pth \
    --mask-checkpoints /users/PAS2985/cz2128/ReID/nba_reid/outputs/v3/VideoMAEv2_mask_beginning_nosplitsampling/VideoMAEv2_mask_beginning_nosplitsampling.pth \
    --output-root /fs/scratch/PAS3184/v3_cam \
    --methods originalcam \
    --sampling uniform