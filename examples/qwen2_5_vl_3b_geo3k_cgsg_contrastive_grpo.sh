#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

# Generate timestamp for unique experiment name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="qwen2_5_vl_3b_geo_cgsg_contrastive_grpo_${TIMESTAMP}"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    algorithm.grpo_variant=cgsg \
    algorithm.cgsg_config.num_negatives=4 \
    algorithm.cgsg_config.negative_selection_strategy=lowest_reward \
    algorithm.cgsg_config.loss_type=contrastive_loss \
    algorithm.cgsg_config.reward_threshold=0.99