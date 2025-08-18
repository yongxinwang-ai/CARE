#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

# PDB variant with strict SLA mode (lower budgets, higher penalties)
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_geo_pdb_strict_grpo \
    trainer.n_gpus_per_node=8 \
    algorithm.grpo_variant=pdb \
    algorithm.pdb_config.sla_mode=strict \
    algorithm.pdb_config.visual_budget=256 \
    algorithm.pdb_config.text_budget=80 \
    algorithm.pdb_config.lambda_v_init=0.04 \
    algorithm.pdb_config.lambda_t_init=0.02 \
    algorithm.pdb_config.enable_crop=true \
    algorithm.pdb_config.enable_draw=true \
    algorithm.pdb_config.enable_tool=false