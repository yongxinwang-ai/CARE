from typing import List

import torch
import torch.nn.functional as F


def select_cosine_hard_negatives(
    anchor_idx: int,
    failure_indices: List[int],
    embeddings: torch.Tensor,
    k: int,
    m: int,
) -> List[int]:
    if k <= 0 or not failure_indices:
        return []

    if embeddings.dim() != 2:
        raise ValueError("embeddings must be 2D (batch_size, hidden_dim)")

    with torch.no_grad():
        emb = embeddings.detach()
        emb = F.normalize(emb, dim=-1)
        anchor = emb[anchor_idx]

        failure_idx = torch.tensor(failure_indices, device=emb.device, dtype=torch.long)
        failure_emb = emb[failure_idx]
        dcos = 1.0 - torch.matmul(failure_emb, anchor)  # (num_failures,)

        pool_size = min(int(m), len(failure_indices))
        nearest = torch.argsort(dcos)[:pool_size]
        pool_indices = [failure_indices[i] for i in nearest.tolist()]

        k_prime = min(int(k), len(pool_indices))
        if k_prime <= 0:
            return []

        selected = [pool_indices[0]]
        if k_prime == 1:
            return selected

        pool_emb = emb[torch.tensor(pool_indices, device=emb.device, dtype=torch.long)]
        selected_mask = torch.zeros(len(pool_indices), device=emb.device, dtype=torch.bool)
        selected_mask[0] = True

        for _ in range(1, k_prime):
            sel_emb = pool_emb[selected_mask]
            distances = 1.0 - torch.matmul(pool_emb, sel_emb.T)
            min_dist, _ = torch.min(distances, dim=1)
            min_dist[selected_mask] = -1.0
            next_idx = int(torch.argmax(min_dist).item())
            selected_mask[next_idx] = True
            selected.append(pool_indices[next_idx])

    return selected
