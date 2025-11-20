#!/bin/bash
#SBATCH --account=PAS2099
#SBATCH --job-name=mvitv2
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=%j_%x.slurm.out

# activate conda env
source /users/PAS2099/clydewu117/anaconda3/etc/profile.d/conda.sh
conda activate mvitv2

cd /users/PAS2099/clydewu117/nba_reid

python train_test.py --config-file exp/new4/color_mask_k400_16frames_beginning_splitsampling_drop03