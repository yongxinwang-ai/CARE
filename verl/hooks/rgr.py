import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from ..protocol import DataProto
from ..trainer.config import RgrCfg
from ..utils.response_tags import extract_region_masks
from ..algorithms.adv_estimators.care import CareGroupInfo


def _decode_response_text(
    tokenizer: PreTrainedTokenizer, response_ids: torch.Tensor, response_mask: torch.Tensor
) -> str:
    resp_len = int(response_mask.sum().item())
    if resp_len <= 0:
        return ""
    ids = response_ids[:resp_len]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _extract_raw_prompt_ids(prompts: torch.Tensor, pad_token_id: int) -> List[int]:
    arr = prompts.detach().cpu().tolist()
    i = 0
    while i < len(arr) and arr[i] == pad_token_id:
        i += 1
    return arr[i:]


def _build_repair_cue(tokenizer: PreTrainedTokenizer, max_tokens: int) -> str:
    cue = "\n<repair>\n- Mistake:\n- Fix:\n- Corrected final answer:\n</repair>\n"
    if max_tokens <= 0:
        return ""
    cue_ids = tokenizer.encode(cue, add_special_tokens=False)[:max_tokens]
    if not cue_ids:
        return ""
    return tokenizer.decode(cue_ids, skip_special_tokens=True)


def _insert_repair_cue(text: str, cue: str) -> str:
    if not text:
        if cue:
            return f"<think>{cue}</think>\n<answer>"
        return "<answer>"

    if "<think>" in text:
        if "</think>" in text:
            head, tail = text.split("</think>", 1)
            text = f"{head}{cue}</think>{tail}"
        else:
            text = f"{text}{cue}"
    else:
        if cue:
            text = f"<think>{cue}</think>\n{text}"

    if "<answer>" in text:
        prefix, _ = text.split("<answer>", 1)
        return f"{prefix}<answer>"
    return f"{text.rstrip()}\n<answer>"


def apply_rgr_care(
    batch: DataProto,
    tokenizer: PreTrainedTokenizer,
    llm_generate_fn: Callable[[DataProto], DataProto],
    sampling_params: Dict[str, Any],
    cfg: RgrCfg,
    group_infos: List[CareGroupInfo],
) -> Tuple[Optional[DataProto], List[Dict[str, Any]], List[Tuple[int, int, int]]]:
    """
    Run RGR for CARE groups and return appended samples plus replacement mapping.
    Returns (appended_dp, events, replacements), where replacements are tuples:
    (group_id, original_neg_idx, appended_local_idx).
    """
    if not cfg or not cfg.enable:
        return None, [], []

    if "token_level_scores" not in batch.batch:
        return None, [], []

    pad_token_id = batch.meta_info.get("pad_token_id", getattr(tokenizer, "pad_token_id", 0))
    appended_blocks: List[DataProto] = []
    events: List[Dict[str, Any]] = []
    replacements: List[Tuple[int, int, int]] = []
    appended_count = 0

    for info in group_infos:
        if not info.has_positive or not info.neg_indices:
            continue
        neg_idx = info.neg_indices[0]

        neg_prompt_tokens = batch.batch["prompts"][neg_idx]
        neg_resp_tokens = batch.batch["responses"][neg_idx]
        neg_resp_mask = batch.batch["response_mask"][neg_idx]
        neg_text = _decode_response_text(tokenizer, neg_resp_tokens, neg_resp_mask)

        cue_text = _build_repair_cue(tokenizer, cfg.max_critique_tokens)
        revised_text = _insert_repair_cue(neg_text, cue_text)
        revised_ids = tokenizer.encode(revised_text, add_special_tokens=False)

        base_raw_prompt = _extract_raw_prompt_ids(neg_prompt_tokens, pad_token_id)
        new_raw_prompt = base_raw_prompt + revised_ids

        prompt_batch = TensorDict(
            {
                "input_ids": neg_prompt_tokens.unsqueeze(0),
                "attention_mask": batch.batch["attention_mask"][neg_idx][..., : neg_prompt_tokens.shape[-1]].unsqueeze(0),
                "position_ids": batch.batch["position_ids"][neg_idx][..., : neg_prompt_tokens.shape[-1]].unsqueeze(0),
            },
            batch_size=(1,),
        )

        non_tensors: Dict[str, Any] = {"raw_prompt_ids": np.array([new_raw_prompt], dtype=object)}
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensors["multi_modal_data"] = np.array([batch.non_tensor_batch["multi_modal_data"][neg_idx]], dtype=object)

        meta = {
            "min_pixels": batch.meta_info.get("min_pixels"),
            "max_pixels": batch.meta_info.get("max_pixels"),
            "video_fps": batch.meta_info.get("video_fps"),
            **(sampling_params or {}),
            "n": 1,
        }
        prompt_dp = DataProto(batch=prompt_batch, non_tensor_batch=non_tensors, meta_info=meta)
        resampled = llm_generate_fn(prompt_dp)

        append_non_tensor: Dict[str, np.ndarray] = {}
        for key, arr in batch.non_tensor_batch.items():
            value = arr[neg_idx]
            append_non_tensor[key] = np.array([value], dtype=object)

        for k, v in append_non_tensor.items():
            if k in resampled.non_tensor_batch:
                continue
            resampled.non_tensor_batch[k] = v

        appended_blocks.append(resampled)
        replacements.append((info.group_id, neg_idx, appended_count))
        appended_count += 1

        events.append(
            {
                "group": info.group_id,
                "group_uid": info.uid,
                "triggered": True,
                "passed": False,
                "replaced_original_failure": True,
                "s_refl_used": None,
                "template": cfg.template,
                "critique_len": None,
                "final_len": None,
                "_neg_idx": neg_idx,
                "_append_idx": appended_count - 1,
            }
        )

    if not appended_blocks:
        return None, events, []

    appended = DataProto.concat(appended_blocks)
    return appended, events, replacements


def process_rgr_events(
    events: List[Dict[str, Any]],
    appended: DataProto,
    tokenizer: PreTrainedTokenizer,
    rewards: torch.Tensor,
    cfg: RgrCfg,
    *,
    log_path: Optional[str] = None,
) -> None:
    if not events:
        return
    if log_path is None:
        return
    if appended is None or len(appended) == 0:
        return
    if "responses" not in appended.batch or "response_mask" not in appended.batch:
        return

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    think_mask, answer_mask, think_len, answer_len = extract_region_masks(
        appended.batch["responses"], appended.batch["response_mask"], tokenizer
    )

    for event in events:
        append_idx = event.get("_append_idx")
        if append_idx is None:
            continue
        idx = int(append_idx)
        passed = bool(rewards[idx].item() > 0)
        event["passed"] = passed
        event["critique_len"] = int(think_len[idx].item())
        event["final_len"] = int(answer_len[idx].item())
        if cfg.s_refl is not None:
            event["s_refl_used"] = float(cfg.s_refl)

        record = {k: v for k, v in event.items() if not k.startswith("_")}
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
