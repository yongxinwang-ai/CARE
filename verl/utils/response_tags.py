from typing import Iterable, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizer


def _find_subsequence(seq: List[int], pattern: List[int], start: int = 0) -> Optional[int]:
    if not pattern or not seq:
        return None
    plen = len(pattern)
    last = len(seq) - plen
    for i in range(start, last + 1):
        if seq[i : i + plen] == pattern:
            return i
    return None


def _token_variants(tokenizer: PreTrainedTokenizer, tag: str) -> List[List[int]]:
    variants = [tag, f" {tag}", f"\n{tag}", f"\n\n{tag}"]
    token_variants: List[List[int]] = []
    for variant in variants:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            token_variants.append(ids)
    return token_variants


def find_tag_span(
    response_ids: List[int],
    valid_len: int,
    start_variants: Iterable[List[int]],
    end_variants: Iterable[List[int]],
) -> Optional[Tuple[int, int]]:
    seq = response_ids[:valid_len]
    best_start = None
    best_start_len = 0
    for variant in start_variants:
        pos = _find_subsequence(seq, variant, start=0)
        if pos is not None and (best_start is None or pos < best_start):
            best_start = pos
            best_start_len = len(variant)

    if best_start is None:
        return None

    search_from = best_start + best_start_len
    best_end = None
    for variant in end_variants:
        pos = _find_subsequence(seq, variant, start=search_from)
        if pos is not None and (best_end is None or pos < best_end):
            best_end = pos

    end_pos = best_end if best_end is not None else valid_len
    if end_pos < search_from:
        return None
    return search_from, end_pos


def extract_region_masks(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    *,
    think_tag: str = "<think>",
    think_end_tag: str = "</think>",
    answer_tag: str = "<answer>",
    answer_end_tag: str = "</answer>",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = responses.device
    batch_size, response_len = responses.shape

    think_start_variants = _token_variants(tokenizer, think_tag)
    think_end_variants = _token_variants(tokenizer, think_end_tag)
    answer_start_variants = _token_variants(tokenizer, answer_tag)
    answer_end_variants = _token_variants(tokenizer, answer_end_tag)

    think_mask = torch.zeros((batch_size, response_len), dtype=torch.bool, device=device)
    answer_mask = torch.zeros((batch_size, response_len), dtype=torch.bool, device=device)
    think_len = torch.zeros((batch_size,), dtype=torch.long, device=device)
    answer_len = torch.zeros((batch_size,), dtype=torch.long, device=device)

    responses_cpu = responses.detach().cpu().tolist()
    response_mask_cpu = response_mask.detach().cpu().tolist()

    for i in range(batch_size):
        valid_len = int(sum(response_mask_cpu[i]))
        if valid_len <= 0:
            continue
        seq = responses_cpu[i]

        think_span = find_tag_span(seq, valid_len, think_start_variants, think_end_variants)
        if think_span is not None:
            start, end = think_span
            if end > start:
                think_mask[i, start:end] = True
                think_len[i] = end - start

        answer_span = find_tag_span(seq, valid_len, answer_start_variants, answer_end_variants)
        if answer_span is not None:
            start, end = answer_span
            if end > start:
                answer_mask[i, start:end] = True
                answer_len[i] = end - start

    return think_mask, answer_mask, think_len, answer_len
