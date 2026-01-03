from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..neg_selectors import select_cosine_hard_negatives


@dataclass
class CareGroupInfo:
    uid: str
    indices: List[int]
    group_id: int
    has_positive: bool
    anchor_idx: Optional[int]
    neg_indices: List[int]
    reflected_idx: Optional[int] = None
    reflected_failed: bool = False


def _select_anchor_index(
    positive_indices: List[int], think_len: torch.Tensor, answer_len: torch.Tensor
) -> int:
    best_idx = positive_indices[0]
    best_think = int(think_len[best_idx].item())
    best_answer = int(answer_len[best_idx].item())
    for idx in positive_indices[1:]:
        t_len = int(think_len[idx].item())
        a_len = int(answer_len[idx].item())
        if t_len < best_think or (t_len == best_think and a_len < best_answer):
            best_idx = idx
            best_think = t_len
            best_answer = a_len
    return best_idx


def build_care_group_infos(
    uids: List[str],
    rewards: torch.Tensor,
    think_len: torch.Tensor,
    answer_len: torch.Tensor,
    embeddings: torch.Tensor,
    *,
    k: int,
    m: int,
) -> List[CareGroupInfo]:
    group_to_indices: Dict[str, List[int]] = {}
    for i, uid in enumerate(uids):
        group_to_indices.setdefault(str(uid), []).append(i)

    group_infos: List[CareGroupInfo] = []
    for group_id, (uid, indices) in enumerate(group_to_indices.items()):
        if not indices:
            continue
        group_rewards = rewards[indices]
        pos_mask = group_rewards > 0
        has_positive = bool(torch.any(pos_mask).item())
        anchor_idx = None
        neg_indices: List[int] = []

        if has_positive:
            positives = [indices[i] for i, flag in enumerate(pos_mask.tolist()) if flag]
            anchor_idx = _select_anchor_index(positives, think_len, answer_len)
            failures = [idx for idx in indices if idx not in positives]
            if failures:
                neg_indices = select_cosine_hard_negatives(anchor_idx, failures, embeddings, k, m)

        group_infos.append(
            CareGroupInfo(
                uid=str(uid),
                indices=list(indices),
                group_id=group_id,
                has_positive=has_positive,
                anchor_idx=anchor_idx,
                neg_indices=neg_indices,
            )
        )

    return group_infos


def _select_diverse_negatives(
    candidates: List[int], embeddings: torch.Tensor, anchor_idx: int, k: int
) -> List[int]:
    if k <= 0 or not candidates:
        return []
    with torch.no_grad():
        emb = F.normalize(embeddings.detach(), dim=-1)
        cand_idx = torch.tensor(candidates, device=emb.device, dtype=torch.long)
        cand_emb = emb[cand_idx]
        anchor_emb = emb[anchor_idx]
        dist_anchor = 1.0 - torch.matmul(cand_emb, anchor_emb)
        start = int(torch.argmax(dist_anchor).item())
        selected = [candidates[start]]
        if k == 1:
            return selected
        selected_mask = torch.zeros(len(candidates), device=emb.device, dtype=torch.bool)
        selected_mask[start] = True
        for _ in range(1, k):
            sel_emb = cand_emb[selected_mask]
            distances = 1.0 - torch.matmul(cand_emb, sel_emb.T)
            min_dist, _ = torch.min(distances, dim=1)
            min_dist[selected_mask] = -1.0
            next_idx = int(torch.argmax(min_dist).item())
            selected_mask[next_idx] = True
            selected.append(candidates[next_idx])
    return selected


def compute_care_advantages(
    group_infos: List[CareGroupInfo],
    rewards: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    embeddings: torch.Tensor,
    *,
    k: int,
    neg_scale_s: float,
    eps: float,
    equalize: bool,
    rescue_enable: bool,
    rescue_delta: float,
    s_refl: float,
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    device = rewards.device
    seq_advantages = torch.zeros_like(rewards, dtype=torch.float32, device=device)
    events: List[Dict[str, float]] = []

    seq_logp = (old_log_probs * response_mask).sum(dim=-1)

    for info in group_infos:
        indices = info.indices
        has_positive = info.has_positive
        anchor_idx = info.anchor_idx
        neg_indices = list(info.neg_indices)

        if has_positive and anchor_idx is not None and neg_indices:
            subgroup = [anchor_idx] + neg_indices
            sub_rewards = rewards[subgroup]
            mu = sub_rewards.mean()
            sigma = sub_rewards.std(unbiased=False) + eps
            raw = (sub_rewards - mu) / sigma

            k_prime = len(neg_indices)
            eq = sqrt(k / k_prime) if equalize and k_prime > 0 else 1.0

            for i, idx in enumerate(subgroup):
                value = raw[i].item()
                if idx == anchor_idx:
                    adv_val = value
                elif rewards[idx] <= 0:
                    scale = s_refl if info.reflected_failed and idx == info.reflected_idx else neg_scale_s
                    adv_val = -scale * abs(value)
                else:
                    adv_val = value
                seq_advantages[idx] = adv_val * eq

            events.append(
                {
                    "group": float(info.group_id),
                    "G": float(len(indices)),
                    "K": float(k),
                    "K_prime": float(k_prime),
                    "has_positive": 1.0,
                    "anchor_idx": float(anchor_idx),
                    "neg_scale_s": float(neg_scale_s),
                    "equalize_factor": float(eq),
                    "rescue_triggered": 0.0,
                    "rescue_delta": float(rescue_delta),
                }
            )
            continue

        if not has_positive and rescue_enable:
            failures = indices
            if not failures:
                continue
            group_logp = seq_logp[failures]
            local_best = int(torch.argmax(group_logp).item())
            pseudo_anchor = failures[local_best]
            neg_candidates = [idx for idx in failures if idx != pseudo_anchor]
            k_neg = min(k, len(neg_candidates))
            if k_neg <= 0:
                continue

            neg_indices = _select_diverse_negatives(neg_candidates, embeddings, pseudo_anchor, k_neg)
            subgroup = [pseudo_anchor] + neg_indices

            sub_rewards = torch.zeros(len(subgroup), device=device)
            sub_rewards[0] = rescue_delta
            sub_rewards[1:] = -rescue_delta / k_neg

            mu = sub_rewards.mean()
            sigma = sub_rewards.std(unbiased=False) + eps
            raw = (sub_rewards - mu) / sigma
            eq = sqrt(k / k_neg) if equalize and k_neg > 0 else 1.0

            for i, idx in enumerate(subgroup):
                value = raw[i].item()
                if i == 0:
                    adv_val = value
                else:
                    adv_val = -neg_scale_s * abs(value)
                seq_advantages[idx] = adv_val * eq

            events.append(
                {
                    "group": float(info.group_id),
                    "G": float(len(indices)),
                    "K": float(k),
                    "K_prime": float(k_neg),
                    "has_positive": 0.0,
                    "anchor_idx": float(pseudo_anchor),
                    "neg_scale_s": float(neg_scale_s),
                    "equalize_factor": float(eq),
                    "rescue_triggered": 1.0,
                    "rescue_delta": float(rescue_delta),
                }
            )

    return seq_advantages, events
