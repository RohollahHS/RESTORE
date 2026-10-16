"""VisPruner (https://github.com/Theia-4869/VisPruner): attention-important tokens + iteratively de-duplicated diverse tokens."""
import torch

IMPORTANT_RATIO = 0.5  # fraction of the budget picked by attention (VisPruner default)


def vispruner_select(image_tokens, image_attn, n_retain, important_ratio=IMPORTANT_RATIO):
    """
    Phase 1: Top-k by attention score → important tokens (T_imp = n_retain * r)
    Phase 2: From residual, iteratively remove duplicate pairs → diverse tokens (T_div = n_retain - T_imp)

    Args:
        image_tokens: (1, N, D)
        image_attn:   (N,) ViT last-layer attention score per token
        n_retain:     total number of tokens to keep
    Returns:
        retained_tokens: (1, n_retain, D), retained_position: (n_retain,) sorted indices into N
    """
    important_token_num = int(n_retain * important_ratio)
    diverse_token_num = n_retain - important_token_num

    # Phase 1: attention-based important selection
    token_indices = image_attn.argsort(descending=True)  # (N,)
    important_indices = token_indices[:important_token_num]  # (T_imp,)
    residual_indices = token_indices[important_token_num:]   # (N - T_imp,)

    # Phase 2: iterative duplicate removal from residual
    if diverse_token_num > 0:
        image_normalized = image_tokens[0] / (image_tokens[0].norm(dim=-1, keepdim=True) + 1e-6)  # (N, D)

        while True:
            R = residual_indices.shape[0]
            r = min(8, R - diverse_token_num)
            if r <= 0:
                break

            residual_tokens = image_normalized[residual_indices]  # (R, D)
            a = residual_tokens[::2]    # (R//2, D)
            b = residual_tokens[1::2]   # (R//2, D)
            scores = a @ b.T            # (R//2, R//2)
            scores = scores.max(dim=-1).values  # (R//2,)

            # Remove most similar (highest score) — keep the least similar
            distinct_indices = scores.argsort(descending=True)[r:]  # (R//2 - r,)
            residual_indices = torch.cat([
                residual_indices[::2][distinct_indices],
                residual_indices[1::2],
            ], dim=0)

        selected_indices = torch.cat([important_indices, residual_indices], dim=0)
    else:
        selected_indices = important_indices

    # Sort by position for causal mask compatibility
    selected_indices_sorted, _ = torch.sort(selected_indices)
    retained_tokens = image_tokens[:, selected_indices_sorted, :]

    return retained_tokens, selected_indices_sorted
