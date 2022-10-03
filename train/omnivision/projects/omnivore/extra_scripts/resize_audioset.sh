#!/bin/bash

# Before running, activate a conda environment that has ffmpeg
# conda activate omnirg

# Run this script with the sbatch script which will run a slurm array to do this
# in parallel
task_id=$1
num_tasks=$2

# Running 1 job since the sbatch script will run a ton of jobs
# also removed the "&" from the ffmpeg command, so it will run this in series
# that will help ensure the checks are completed that the video was correclty
# rendered. And I leave the parallelization to the sbatch script that is running
# a lot of 1 CPU jobs.
num_procs=1  # Run this many in parallel at max
small_side=288
max_tries=5
indir="/datasets01/audioset/042319/data/eval_segments/video/"
# This path must be updated in the sbatch script as well if running using sbatch for correct
# out/err log locations (else the sbatch scripts will keep dying without complaining)
outdir="/fsx-omnivore/rgirdhar/data/audioset/eval_segments/video_mp4-288p/"

cd $indir || exit
all_videos=$(find . -iname "*.mp4")
all_videos=( $all_videos )  # to array
cd -

# Subselect the videos to process for this task
num_videos=${#all_videos[@]}
# A convoluted way to do ceil https://stackoverflow.com/a/12536521/1492614
(( per_task_videos=(num_videos+num_tasks-1)/num_tasks ))
(( start_pos=per_task_videos*task_id ))
videos=("${all_videos[@]:${start_pos}:${per_task_videos}}")
echo "Selected ${#videos[@]} from original $num_videos videos to run in task $task_id / $num_tasks"

num_jobs="\j"  # The prompt escape for number of jobs currently running
for video in "${videos[@]}"; do
    while (( ${num_jobs@P} >= num_procs )); do
        wait -n
    done
    W=$( ffprobe -v quiet -show_format -show_streams -show_entries stream=width "${indir}/${video}" | grep width )
    W=${W#width=}
    H=$( ffprobe -v quiet -show_format -show_streams -show_entries stream=height "${indir}/${video}" | grep height )
    H=${H#height=}
    # Set the smaller side to small_side
    # from https://superuser.com/a/624564
    if [ $W -gt $H ] && [ $H -gt ${small_side} ]; then
        scale_str="-filter:v scale=-1:${small_side}"
    elif [ $H -gt $W ] && [ $W -gt ${small_side} ]; then
        scale_str="-filter:v scale=${small_side}:-1"
    else
        # The small side is smaller than required size, so don't resize/distort the video
        scale_str=""
    fi
    outfpath=${outdir}/${video}
    try=0
    while [ $try -le $max_tries ]; do
        # TODO Ideally use a lower -crf value for higher quality, such
        # as -crf 17
        # https://trac.ffmpeg.org/wiki/Encode/H.264#:~:text=Choose%20a%20CRF%20value,sane%20range%20is%2017%E2%80%9328.
        ffmpeg -y -i "${indir}/${video}" ${scale_str} "${outfpath}"
        try=$(( $try + 1 ))
        write_errors=$( ffprobe -v error -i "${outfpath}" )
        # If no errors detected by ffprobe, we are done
        if [ -z "$write_errors" ]; then
            echo $outfpath written successfully in $try tries!
            break
        fi
    done
    echo "Converted ${video}"
done
