import torch


def apply_region_weighted_advantages(
    sequence_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    think_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    rewards: torch.Tensor,
    gamma_pos: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    if response_mask.dim() != 2:
        raise ValueError("response_mask must be 2D (batch_size, response_len)")

    answer_mask = answer_mask.bool()
    think_mask = think_mask.bool()
    response_mask = response_mask.bool()

    missing_answer = answer_mask.sum(dim=-1) == 0
    if torch.any(missing_answer):
        answer_mask = torch.where(missing_answer.unsqueeze(-1), response_mask, answer_mask)

    weights = answer_mask.float()
    pos_mask = rewards > 0
    if gamma_pos > 0:
        weights = weights + think_mask.float() * (gamma_pos * pos_mask.float().unsqueeze(-1))

    weights = weights * response_mask.float()
    denom = weights.sum(dim=-1, keepdim=True).clamp(min=eps)
    token_adv = sequence_advantages.unsqueeze(-1) * weights / denom
    token_adv = token_adv * response_mask.float()
    return token_adv
