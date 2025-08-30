#!/bin/bash
#SBATCH --time=14-00:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=qwen2_5_vl_3b_geo3k_pge_grpo.log


conda activate easy_r1

# bash examples/qwen2_5_vl_7b_geo3k_cgsg_contrastive_grpo.sh
# bash examples/qwen2_5_vl_3b_geo3k_dapo.sh
# bash examples/qwen2_5_vl_7b_geo3k_cgsg_contrastive_grpo.sh
# bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
bash examples/qwen2_5_vl_3b_geo3k_pge_grpo.sh