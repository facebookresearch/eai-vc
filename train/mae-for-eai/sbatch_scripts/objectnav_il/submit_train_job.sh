#!/bin/bash
#SBATCH --job-name=mae_onav
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 6
#SBATCH --ntasks-per-node 1
#SBATCH --signal=USR1@1000
#SBATCH --partition=short
#SBATCH --constraint=a40
#SBATCH --exclude=chappie,robby
#SBATCH --output=slurm_logs/ddpil-%j.out
#SBATCH --error=slurm_logs/ddpil-%j.err
#SBATCH --requeue

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate eai

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

config=$1

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_10k"
TENSORBOARD_DIR="tb/objectnav_il/overfitting/vit/seed_2/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/overfitting/vit/seed_2/"
INFLECTION_COEF=3.234951275740812
set -x

echo "In ObjectNav IL DDP"
srun python -u -m run \
--exp-config $config \
--run-type train \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
CHECKPOINT_INTERVAL 100 \
NUM_UPDATES 5000 \
NUM_PROCESSES 4 \
LOG_INTERVAL 1 \
IL.BehaviorCloning.num_steps 64 \
IL.BehaviorCloning.num_mini_batch 2 \
TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF \
TASK_CONFIG.DATASET.SPLIT "sample" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.DATASET.TYPE "ObjectNav-v2" \
TASK_CONFIG.DATASET.MAX_EPISODE_STEPS 700 \
TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR', 'DEMONSTRATION_SENSOR', 'INFLECTION_WEIGHT_SENSOR']" \
MODEL.RGB_ENCODER.backbone "vit_small_patch16" \
MODEL.RGB_ENCODER.pretrained_encoder "data/visual_encoders/mae_vit_small_decoder_large_HGPS_RE10K_100.pth"
