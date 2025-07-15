#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

# TreeGRPO specific parameters
STEP_LENGTH=256
MAX_RESPONSE_LENGTH=2048
ROLLOUT_N=5

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    algorithm.adv_estimator=tree_grpo \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=${ROLLOUT_N} \
    worker.rollout.step_length=${STEP_LENGTH} \
    trainer.experiment_name=qwen2_5_vl_3b_geo_simple_tree_grpo \
    trainer.n_gpus_per_node=8