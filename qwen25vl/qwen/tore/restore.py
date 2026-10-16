"""RESTORE for Qwen2.5-VL: merge the pruned tokens, give them M-RoPE positions, and calibrate the attention.

LLaVA counterpart: tokenreduction/tore/*/llava_arch_restore.py (merging) and
llava/model/modeling_llama_restore.py (calibration) in the RESTORE repository.
"""
import torch

from .merge import distinctive_anchor_merging, merge_source, merge_wavg, sort_rows_by_first_source

GAMMA = 0.5              # share of the token budget selected by the baseline; the rest is merged
CALIBRATION_CONST = 1.0  # bias log(const + 1 - D) = log(2 - D), i.e. c = 2 in the paper


def merge_pruned_tokens(image_tokens, retained_tokens, retained_position, n_merged):
    """
    Merges the tokens the baseline did not keep into n_merged distinctive anchors.

    Args:
        image_tokens:      (1, N_vis, D) all visual tokens
        retained_tokens:   (1, n_retain, D) tokens kept by the baseline
        retained_position: (n_retain,) their indices into N_vis
        n_merged:          number of merged tokens to add
    Returns:
        new_image_tokens: (1, n_retain + n_merged, D) ordered by their first source token
        merging_pattern:  (1, n_retain + n_merged, N_vis) one-hot source map of every new token
        token_size:       (1, n_retain + n_merged, 1) number of source tokens behind each new token
    """
    B_img, N_vis, _ = image_tokens.shape
    n_retain = retained_position.shape[0]
    device = image_tokens.device

    retain_mask = torch.zeros(N_vis, dtype=torch.bool, device=device)
    retain_mask[retained_position] = True
    pruned_indices = torch.arange(N_vis, device=device)[~retain_mask]  # (N_vis - n_retain,)
    image_tokens_to_merge = image_tokens[:, pruned_indices, :]

    # Source tracking pattern of the pruned tokens
    merging_pattern = torch.zeros(B_img, N_vis - n_retain, N_vis, device=device, dtype=image_tokens.dtype)
    merging_pattern[:, torch.arange(pruned_indices.shape[0], device=device), pruned_indices] = 1.0

    merge, _ = distinctive_anchor_merging(metric=image_tokens_to_merge, n_merged=n_merged)
    merging_pattern = merge_source(merge=merge, x=image_tokens_to_merge, source=merging_pattern)
    image_tokens_merged, _ = merge_wavg(merge=merge, x=image_tokens_to_merge)

    merging_pattern[0], perm, _ = sort_rows_by_first_source(merging_pattern[0])
    image_tokens_merged[0] = image_tokens_merged[0][perm]

    # Retained tokens are their own single source
    prune_pattern = torch.zeros(B_img, n_retain, N_vis, device=device, dtype=image_tokens.dtype)
    prune_pattern[:, torch.arange(n_retain, device=device), retained_position] = 1.0

    new_image_tokens = torch.cat([retained_tokens.to(image_tokens_merged.dtype), image_tokens_merged], dim=1)
    merging_pattern = torch.cat([prune_pattern, merging_pattern], dim=1)
    merging_pattern[0], perm, _ = sort_rows_by_first_source(merging_pattern[0])
    new_image_tokens[0] = new_image_tokens[0][perm]

    token_size = merging_pattern.sum(dim=-1, keepdim=True)  # retained = 1, merged = #sources
    return new_image_tokens, merging_pattern, token_size


def compute_merged_positions(position_ids, merging_pattern, size):
    """
    3D M-RoPE position of each new token = mean position of its source tokens.

    Args:
        position_ids:    (3, B, N_vis) positions of the original visual tokens
        merging_pattern: (B, n_new, N_vis)
        size:            (B, n_new, 1)
    Returns:
        merged_position_ids: (3, B, n_new) float positions
    """
    merged_pos_list = []
    for axis in range(position_ids.shape[0]):  # temporal, height, width
        pos_k = position_ids[axis].float()  # (B, N_vis)
        merged_pos = torch.bmm(
            merging_pattern.to(dtype=pos_k.dtype),
            pos_k.unsqueeze(-1)
        ).squeeze(-1)  # (B, n_new)
        merged_pos_list.append(merged_pos / size.squeeze(-1))

    return torch.stack(merged_pos_list, dim=0)


def compute_mrope_decay_table(
    head_dim: int = 128,
    rope_theta: float = 1000000.0,
    mrope_section: list = [16, 24, 24],
    max_dist: int = 4096,
    device: torch.device = None,
):
    """
    Per-axis RoPE decay lookup table: D_k(d) = mean(cos(d * theta_j)) over axis k's frequencies.

    Returns:
        decay_tables: list of 3 tensors, each (max_dist,)
        weights: (3,) — w_k = 2 * s_k / d_h
    """
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))  # (head_dim/2,)
    t = torch.arange(max_dist, device=device).float()  # (max_dist,)

    decay_tables = []
    start = 0
    for s_k in mrope_section:
        freq_slice = inv_freq[start:start + s_k]  # (s_k,)
        freqs = t.unsqueeze(1) * freq_slice.unsqueeze(0)  # (max_dist, s_k)
        decay_tables.append(torch.mean(torch.cos(freqs), dim=-1))  # (max_dist,)
        start += s_k

    weights = torch.tensor([2 * s_k / head_dim for s_k in mrope_section],
                           device=device, dtype=torch.float32)

    return decay_tables, weights


def compute_mrope_calibration(
    position_ids: torch.Tensor,
    token_size: torch.Tensor,
    head_dim: int = 128,
    rope_theta: float = 1000000.0,
    mrope_section: list = [16, 24, 24],
    max_dist: int = 4096,
    const: float = CALIBRATION_CONST,
    device: torch.device = None,
):
    """
    Additive attention bias log(s_n) + log(c + 1 - D_mrope(m, n)).

    Args:
        position_ids: (3, B, seq_len) — M-RoPE 3D position IDs
        token_size:   (B, seq_len) — size of every token (text / retained = 1, merged = #sources)
    Returns:
        calibration_mask: (B, 1, seq_len, seq_len)
    """
    B = position_ids.shape[1]
    seq_len = position_ids.shape[2]

    # 1. Token size bias: log(s_n), broadcast over the query dimension
    log_s = torch.log(token_size).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, seq_len)

    # 2. Distance bias: D_mrope(m, n) = sum_k w_k * D_k(|pos_k[m] - pos_k[n]|)
    decay_tables, weights = compute_mrope_decay_table(
        head_dim=head_dim, rope_theta=rope_theta,
        mrope_section=mrope_section, max_dist=max_dist, device=device,
    )
    dist_bias = torch.zeros(B, 1, seq_len, seq_len, device=device, dtype=torch.float32)
    for k in range(3):  # temporal, height, width
        pos_k = position_ids[k]  # (B, seq_len)
        d_matrix = (pos_k.unsqueeze(-1) - pos_k.unsqueeze(-2)).abs()  # (B, seq_len, seq_len)
        d_matrix = d_matrix.clamp(max=max_dist - 1).long()
        dist_bias[:, 0, :, :] += weights[k] * decay_tables[k][d_matrix]

    dist_bias = torch.log(const + 1.0 - dist_bias)

    return log_s + dist_bias


def calibrated_attention_mask(config, position_ids, token_size_vis, n_sys):
    """
    Causal prefill mask carrying the RESTORE calibration; it replaces the model's attention mask.

    Args:
        position_ids:   (3, 1, seq_len) after reduction
        token_size_vis: (1, n_vis) sizes of the visual tokens, which start at index n_sys
    Returns:
        (1, 1, seq_len, seq_len) float mask, -inf above the diagonal
    """
    device = token_size_vis.device
    seq_len = position_ids.shape[2]
    n_txt = seq_len - n_sys - token_size_vis.shape[1]
    token_size = torch.cat([
        torch.ones(1, n_sys, device=device),
        token_size_vis,
        torch.ones(1, n_txt, device=device),
    ], dim=1)  # (1, seq_len)

    calibration_mask = compute_mrope_calibration(
        position_ids=position_ids,
        token_size=token_size,
        head_dim=config.hidden_size // config.num_attention_heads,
        rope_theta=config.rope_theta,
        mrope_section=config.rope_scaling.get('mrope_section', [16, 24, 24]),
        device=device,
    )
    causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    return calibration_mask.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
