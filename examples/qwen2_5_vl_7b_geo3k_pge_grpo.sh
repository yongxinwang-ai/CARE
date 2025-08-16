#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_3b_geo_pge_grpo \
    trainer.n_gpus_per_node=8 \
    algorithm.grpo_variant=pge \
    algorithm.pge_config.num_perturbations=4 \
    algorithm.pge_config.perturbation_methods='["token_substitute", "cot_step_resample"]' \
    algorithm.pge_config.perturbation_strength=0.1