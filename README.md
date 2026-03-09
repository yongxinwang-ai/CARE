# CARE: Contrastive Anchored-REflection for Verifiable Multimodal Reasoning

📑 <a href="https://arxiv.org/pdf/2512.19554">Paper</a> | 🤗 <a href="https://huggingface.co/YongxinWang">Hugging Face</a>


## News

2026.2 Our paper is accepted by **CVPR 2026**! See you in Denver!
2026.1 Code is released.

## Method overview

Given a multimodal prompt x = <image(s), question>:

1. Sample a group of rollouts (size G).
2. Verify each rollout with a programmatic verifier (exact match / format checks).
3. If there is at least one verified-correct rollout:
   - Choose the anchor as the shortest verified-correct rationale.
   - Select hard negatives closest to the anchor in rationale space (cosine distance over pooled hidden states).
   - Compute within-subgroup normalized advantages and down-weight negatives.
   - Optionally run RGR: repair one hard negative once, re-verify, and replace if it becomes correct.
4. If all rollouts are incorrect:
   - Apply an all-negative rescue pseudo-contrast to avoid stalled gradients.

CARE reshapes selection and advantages only. The reward function itself is unchanged.

---

## Implementation map (this repo)

- CARE subgrouping + advantages: `verl/algorithms/adv_estimators/care.py`
- Cosine hard negatives: `verl/algorithms/neg_selectors/cosine_hardneg.py`
- Region-weighted token advantages: `verl/algorithms/losses/region_weighted_tokens.py`
- RGR hook (training-only): `verl/hooks/rgr.py`
- Tag parsing for <think>/<answer>: `verl/utils/response_tags.py`
- Config surface: `examples/config.yaml` (`care.*`, `rgr.*`)
- Qwen2.5-VL CARE script: `examples/qwen2_5_vl_7b_geo3k_care_grpo.sh`

---

## Installation

```bash
git clone https://github.com/yongxinwang-ai/CARE.git
cd CARE

# (recommended) create env
conda create -n care python=3.10 -y
conda activate care

pip install -e .
```

---

## Quickstart

### CARE (Qwen2.5-VL 7B, Geometry3K)

```bash
bash examples/qwen2_5_vl_7b_geo3k_care_grpo.sh
```

### CARE with overrides

```bash
python3 -m verl.trainer.main \
  config=examples/config.yaml \
  data.train_files=hiyouga/geometry3k@train \
  data.val_files=hiyouga/geometry3k@test \
  worker.actor.model.model_path=Qwen/Qwen2.5-VL-7B-Instruct \
  algorithm.grpo_variant=care \
  care.K=4 care.M=6 \
  care.neg_scale_s=0.5 care.equalize=true \
  care.rescue.enable=true care.rescue.delta=0.1 \
  care.token_weighting=region_weighted care.gamma_pos=0.005 \
  rgr.enable=true rgr.template=structured
```

---

## CARE configuration

CARE is exposed via `algorithm.grpo_variant=care` and the `care.*` / `rgr.*` sections in
`examples/config.yaml`. Defaults match the roadmap:

- Rollouts per prompt: G = worker.rollout.n (default 8 in examples)
- Hard-negative subgroup size: care.K = 4
- Negative preselect size: care.M = 6
- Negative scaling: care.neg_scale_s = 0.5
- Reflected-failure scaling: rgr.s_refl = care.neg_scale_s / 2 (if not set)
- All-negative rescue magnitude: care.rescue.delta = 0.1
- Positive rationale token weight (region-weighting): care.gamma_pos = 0.005


---

## Citation

If you use this code, please cite the CARE paper:

```bibtex
@article{wang2025care,
  title   = {CARE What Fails: Contrastive Anchored-REflection for Verifiable Multimodal Reasoning},
  author  = {Wang, Yongxin and Yang, Zhicheng and Cao, Meng and Han, Mingfei and Lin, Haokun and Zhu, Yingying and Chang, Xiaojun and Liang, Xiaodan},
  year    = {2025},
  note    = {arXiv preprint (add identifier once available)}
}
```

---

## License

Apache-2.0. See `LICENSE`.
