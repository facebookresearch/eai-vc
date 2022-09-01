#!/bin/bash

ROOT_DIR=$(git rev-parse --show-toplevel)
names=( datasets scene_datasets )

for s in "${names[@]}"
do
  TARGET_PATH="$ROOT_DIR/eval/habitat_eaif/data/$s"
  if [ -L $TARGET_PATH ]; then
    echo "$TARGET_PATH already exists"
  else
    echo "Simlinking to $TARGET_PATH"
    ln -s /checkpoint/yixinlin/eaif/datasets/habitat_task_dataset/$s $TARGET_PATH
  fi
done

