#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen3-4B  # replace it with your local file path

# Generate timestamp for unique experiment name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="qwen3_4b_math_cgsg_grpo_${TIMESTAMP}"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    algorithm.grpo_variant=cgsg \
    algorithm.cgsg_config.num_negatives=4 \
    algorithm.cgsg_config.negative_selection_strategy=lowest_reward \
    algorithm.cgsg_config.loss_type=normalized_advantage \
    algorithm.cgsg_config.reward_threshold=0.99