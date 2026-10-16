"""Token merging used by RESTORE (same functions as RESTORE's tokenreduction/tore/merge.py)."""
from typing import Callable, Tuple

import torch
import torch.nn.functional as F


def distinctive_anchor_merging(
    metric: torch.Tensor,
    n_merged: int,
    class_token: bool = False,
):
    """
    DPC-style one-shot merging: anchors are the tokens with the highest density x distinctiveness,
    every other token is merged into its most similar anchor.

    Args:
        metric: (B, N, C) — token features
        n_merged: number of anchor (destination) tokens
    Returns:
        merge: callable — x (B, N, C) → merged (B, n_merged, C)
        None
    """
    B, N, C = metric.shape
    n_src = N - n_merged

    if n_merged <= 0:
        raise ValueError("Too many tokens to reduce.")

    with torch.no_grad():
        metric_norm = metric / metric.norm(dim=-1, keepdim=True)
        sim_matrix = metric_norm @ metric_norm.transpose(-1, -2)  # (B, N, N)

        # Step 1: density
        density = sim_matrix.sum(dim=-1)  # (B, N)

        # Step 2: distinctiveness — similarity to the closest denser token
        better_mask = density.unsqueeze(1) < density.unsqueeze(2)
        masked_sim = sim_matrix.clone()
        masked_sim[~better_mask] = -1.0
        max_sim_to_better, _ = masked_sim.max(dim=-1)

        is_global_leader = (max_sim_to_better == -1.0)
        max_sim_to_better[is_global_leader] = 0.0

        diversity_factor = 1.0 - max_sim_to_better
        final_score = density * diversity_factor

        if class_token:
            final_score[..., 0] = float('inf')

        # Step 3: anchors = top n_merged scores
        _, dst_idx = final_score.topk(k=n_merged, dim=-1)
        dst_idx, _ = dst_idx.sort(dim=-1)  # (B, n_merged)

        # Step 4: sources = the rest
        mask = torch.ones(B, N, dtype=torch.bool, device=metric.device)
        mask.scatter_(1, dst_idx, False)

        all_indices = torch.arange(N, device=metric.device).expand(B, N)
        src_idx = all_indices[mask].view(B, n_src)

        # Step 5: assign each source to its most similar anchor
        src_sims = sim_matrix.gather(1, src_idx.unsqueeze(-1).expand(B, n_src, N))
        src_to_dst_sims = src_sims.gather(2, dst_idx.unsqueeze(1).expand(B, n_src, n_merged))
        best_match_rel_idx = src_to_dst_sims.argmax(dim=-1)  # (B, n_src)
        assignment_idx = best_match_rel_idx.unsqueeze(-1)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        n, _, c = x.shape
        dst = x.gather(1, dst_idx.unsqueeze(-1).expand(n, n_merged, c))
        src = x.gather(1, src_idx.unsqueeze(-1).expand(n, n_src, c))

        assignment_mat = F.one_hot(assignment_idx.squeeze(-1), num_classes=n_merged).to(dtype=x.dtype)
        merged_sum = torch.bmm(assignment_mat.transpose(1, 2), src)  # (B, n_merged, C)

        if mode == "sum":
            dst = dst + merged_sum
        elif mode == "mean":
            counts = assignment_mat.transpose(1, 2).sum(dim=-1, keepdim=True)  # (B, n_merged, 1)
            dst = (dst + merged_sum) / (1 + counts)
        elif mode == "amax":
            dst = dst + merged_sum
            dst = dst.clamp(max=1.0)

        return dst

    return merge, None


def merge_wavg(
    merge: Callable,
    x: torch.Tensor,
    size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weighted average merge based on token size. Returns the merged tensor and the new token sizes."""
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size

    return x, size


def merge_source(
    merge: Callable,
    x: torch.Tensor,
    source: torch.Tensor = None
) -> torch.Tensor:
    """Source tracking — adjacency matrix between the initial tokens and the merged groups."""
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source


def sort_rows_by_first_source(merging_pattern: torch.Tensor):
    """
    merging_pattern: (T, N_vis) — row = destination, col = source
    Returns: sorted pattern, permutation (new = old[perm]), first source index of each row
    """
    first_idx = merging_pattern.argmax(dim=1)  # (T,)
    perm = torch.argsort(first_idx, stable=True)  # (T,)
    merged_sorted = merging_pattern[perm]
    return merged_sorted, perm, first_idx
