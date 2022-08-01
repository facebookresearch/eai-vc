#!/bin/bash

ROOT_DIR=$(git rev-parse --show-toplevel)
names=( datasets ddppo-models scene_datasets )

for s in "${names[@]}"
do
  TARGET_PATH="$ROOT_DIR/data/$s"
  if [ -L $TARGET_PATH ]; then
    echo "$TARGET_PATH already exists"
  else
    echo "Simlinking to $TARGET_PATH"
    ln -s /private/home/karmeshyadav/mae/mae-for-eai/data/$s $TARGET_PATH
  fi
done

