#!/bin/bash
#SBATCH --nodes=1
#SBATCH --array=0-511
#SBATCH --cpus-per-task=1
#SBATCH --partition=learnlab
#SBATCH --output=/fsx-omnivore/rgirdhar/data/audioset/slurm/%A_%a.out
#SBATCH --err=/fsx-omnivore/rgirdhar/data/audioset/slurm/%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --signal=USR1@600
#SBATCH --open-mode=append
#SBATCH --ntasks-per-node=1

conda activate /private/home/rgirdhar/.conda/envs/omnirg
# If the path to the --output or --err above does not exist,
# the sbatch job will get submitted but die immidiately without any error
# Can not specify --mem in AWS since the SLURM is configured that way
srun --label \
    bash /data/home/rgirdhar/Work/FB/2021/003_JointImVid/omnivision/projects/omnivore/extra_scripts/resize_audioset.sh ${SLURM_ARRAY_TASK_ID} ${SLURM_ARRAY_TASK_COUNT}
