run_training() {
    SEED=$1
    # create run folder
    RUN_FOLDER="/checkpoint/${USER}/${REPO_NAME}/${EXP_NAME}/${SEED}"
    LOG_DIR="${RUN_FOLDER}/logs"
    CHKP_DIR="${RUN_FOLDER}/chkp"
    VIDEO_DIR="${RUN_FOLDER}/videos"
    CMD_TRAIN_OPTS_FILE="${LOG_DIR}/cmd_opt.txt"

    # Create folders
    mkdir -p ${CHKP_DIR}
    mkdir -p ${LOG_DIR}
    mkdir -p ${VIDEO_DIR}

    if [ -z "${CHKP_NAME}" ]; then
        EVAL_CKPT_PATH_DIR="${CHKP_DIR}"
    else
        EVAL_CKPT_PATH_DIR="${CHKP_DIR}/${CHKP_NAME}"
    fi

    # Write commands to file
    CMD_COMMON_OPTS="--exp-config $EXP_CONFIG_PATH \
        BASE_TASK_CONFIG_PATH $BASE_TASK_CONFIG_PATH \
        EVAL_CKPT_PATH_DIR ${EVAL_CKPT_PATH_DIR} \
        CHECKPOINT_FOLDER ${CHKP_DIR} \
        TENSORBOARD_DIR ${LOG_DIR} \
        VIDEO_DIR ${VIDEO_DIR} \
        RL.DDPPO.pretrained_weights ${REPO_PATH}/data/ddppo-models/${WEIGHTS_NAME} \
        TASK_CONFIG.DATASET.SCENES_DIR ${REPO_PATH}/data/scene_datasets \
        RL.DDPPO.backbone ${BACKBONE} \
        TASK_CONFIG.SEED ${SEED} \
        TOTAL_NUM_STEPS ${NUM_STEPS} \
        VIDEO_OPTION ${VIDEO_OPTION} \
        ${EXTRA_CMDS}"

    CMD_TRAIN_OPTS="${CMD_COMMON_OPTS} \
        TASK_CONFIG.DATASET.SPLIT train \
        TASK_CONFIG.DATASET.DATA_PATH ${REPO_PATH}/data/datasets/pointnav/${ENVIRONMENT}/v1/${SPLIT}/${SPLIT}.json.gz \
        NUM_ENVIRONMENTS ${NUM_ENV} \
        WANDB_NAME ${EXP_NAME} \
        WANDB_MODE ${WANDB_MODE}"

    if [ "$RUN_TRAIN_SCRIPT" = true ]; then
        echo $CMD_TRAIN_OPTS > ${CMD_TRAIN_OPTS_FILE}

        sbatch \
            --export=ALL,CMD_OPTS_FILE=${CMD_TRAIN_OPTS_FILE},MODE='train' \
            --job-name=${EXP_NAME} \
            --output=$LOG_DIR/log.out \
            --error=$LOG_DIR/log.err \
            --partition=$PARTITION \
            --nodes $NODES \
            --time $TIME \
            sbatch_scripts/sbatch_file.sh
    fi

    # Evaluate the model simultanously on the saved checkpoints
    CMD_EVAL_OPTS_FILE="${LOG_DIR}/cmd_eval_opt.txt"

    CMD_EVAL_OPTS="${CMD_COMMON_OPTS} \
        EVAL.SPLIT ${VAL_SPLIT} \
        TASK_CONFIG.DATASET.CONTENT_SCENES [\"*\"] \
        TEST_EPISODE_COUNT ${TEST_EPISODE_COUNT} \
        NUM_ENVIRONMENTS 10 \
        RL.PPO.num_mini_batch 1 \
        TASK_CONFIG.DATASET.DATA_PATH ${REPO_PATH}/data/datasets/pointnav/${ENVIRONMENT}/v1/${VAL_SPLIT}/${VAL_SPLIT}.json.gz \
        TASK_CONFIG.TASK.TOP_DOWN_MAP.MAP_RESOLUTION 1024 \
        WANDB_NAME ${EXP_NAME} \
        WANDB_MODE ${WANDB_MODE}"
    
    # Run evaluation if EVAL_ON_TRAIN is set to True
    if [ "$RUN_EVAL_SCRIPT" = true ]; then
        echo "$CMD_EVAL_OPTS" > ${CMD_EVAL_OPTS_FILE}
        
        sbatch \
            --export=ALL,CMD_OPTS_FILE=${CMD_EVAL_OPTS_FILE},MODE='eval' \
            --job-name=${EXP_NAME} \
            --output=$LOG_DIR/log_${VAL_SPLIT}.out \
            --error=$LOG_DIR/log_${VAL_SPLIT}.err \
            --partition=$PARTITION \
            --nodes 1 \
            --time $TIME \
            sbatch_scripts/sbatch_file.sh
    fi

}
