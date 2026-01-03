#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="qwen2_5_vl_7b_geo_care_grpo_${TIMESTAMP}"
RUN_DIR="runs/${EXPERIMENT_NAME}"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    algorithm.grpo_variant=care \
    care.K=4 care.M=6 \
    care.neg_scale_s=0.5 care.equalize=true \
    care.rescue.enable=true care.rescue.delta=0.1 \
    care.token_weighting=region_weighted care.gamma_pos=0.005 \
    rgr.enable=true rgr.template=structured \
    rgr.max_critique_tokens=64 \
    rgr.sampling.temperature=0.6 rgr.sampling.top_p=0.95 rgr.sampling.max_tokens=512 \
    care.instrument.enable=true \
    care.instrument.care_jsonl=${RUN_DIR}/care_events.jsonl \
    care.instrument.rgr_jsonl=${RUN_DIR}/rgr_events.jsonl
