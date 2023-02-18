#!/bin/bash
#SBATCH --job-name=save_images
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 80
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@300
#SBATCH --partition=devlab,learnlab
#SBATCH --output=slurm_logs/save_images-%j.out
#SBATCH --error=slurm_logs/save_images-%j.err
#SBATCH --requeue

. /public/apps/anaconda3/5.0.1/etc/profile.d/conda.sh
conda deactivate
conda activate eaif

cd /private/home/$USER/eai-foundations/tools/

python save_images_from_videos.py --store_images --num_processes 75 --saving_fps 0.05
# python youtube_downloader.py --num_processes 10 --dont_store_images --download_subset $SLURM_ARRAY_TASK_ID