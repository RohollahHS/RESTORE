"""Applies a pruning baseline (and optionally RESTORE) to the visual tokens of a Qwen2.5-VL prefill."""
import torch

from .restore import GAMMA, calibrated_attention_mask, compute_merged_positions, merge_pruned_tokens


class VisualTokenReduction:
    """Called from Qwen2_5_VLForConditionalGeneration.forward right after the image features are scattered in."""

    def __init__(self, select, retain_ratio, is_restore):
        self.select = select  # (image_tokens, image_attn, n_retain) -> (retained_tokens, retained_position)
        self.retain_ratio = retain_ratio
        self.is_restore = is_restore

    def __call__(self, config, inputs_embeds, input_ids, position_ids, attention_mask,
                 image_attn, image_token_start, image_token_len):
        """
        Args:
            inputs_embeds:     (1, L, D) prefill embeddings with the image features scattered in
            input_ids:         (1, L)
            position_ids:      (3, 1, L) M-RoPE positions
            attention_mask:    replaced by an explicit (1, 1, L, L) prefill mask, calibrated with RESTORE and plain causal without
            image_attn:        (image_token_len,) ViT last-layer attention score per image token
            image_token_start: index of the first image token (right after <|vision_start|>)
            image_token_len:   number of image tokens
        Returns:
            inputs_embeds, input_ids, position_ids, attention_mask of the reduced sequence
        """
        image_token_end = image_token_start + image_token_len
        pre_prompt = inputs_embeds[:, :image_token_start, :]
        image_tokens = inputs_embeds[:, image_token_start:image_token_end, :]
        others = inputs_embeds[:, image_token_end:, :]

        keep_num = max(1, int(image_token_len * self.retain_ratio))
        if self.is_restore:
            n_retain = max(1, int(keep_num * GAMMA))
            n_merged = keep_num - n_retain
        else:
            n_retain, n_merged = keep_num, 0

        # 1. Baseline selection
        retained_tokens, retained_position = self.select(image_tokens, image_attn, n_retain)

        if n_merged > 0:
            # 2. RESTORE: merge the pruned tokens and place every token at the mean position of its sources
            new_image_tokens, merging_pattern, token_size_vis = merge_pruned_tokens(
                image_tokens, retained_tokens, retained_position, n_merged)
            merged_position_ids = compute_merged_positions(
                position_ids[:, :, image_token_start:image_token_end], merging_pattern, token_size_vis)

            # Order the visual tokens by height position for spatial coherence
            _, order = torch.sort(merged_position_ids[1, 0])
            merged_position_ids = merged_position_ids[:, :, order]
            new_image_tokens[0] = new_image_tokens[0][order]
            token_size_vis = token_size_vis[:, order, :].squeeze(-1)  # (1, keep_num)

            inputs_embeds = torch.cat([pre_prompt, new_image_tokens, others], dim=1).to(pre_prompt.dtype)
            vis_ids = input_ids[:, image_token_start:image_token_end][:, :1].expand(-1, keep_num)
            input_ids = torch.cat([input_ids[:, :image_token_start], vis_ids, input_ids[:, image_token_end:]], dim=1)
            position_ids = torch.cat([
                position_ids[:, :, :image_token_start].float(),
                merged_position_ids.float(),
                position_ids[:, :, image_token_end:].float(),
            ], dim=2)
        else:
            # Prune only
            inputs_embeds = torch.cat([pre_prompt, retained_tokens, others], dim=1).to(pre_prompt.dtype)
            input_ids = select_visual(input_ids, image_token_start, image_token_end, retained_position)
            position_ids = select_visual(position_ids, image_token_start, image_token_end, retained_position)
            token_size_vis = torch.ones(1, n_retain, device=inputs_embeds.device)

        # 3. Prefill attention mask, always passed to SDPA explicitly (as in RESTORE's LLaVA code) so that
        #    w/ and w/o RESTORE run the same (memory-efficient) SDPA kernel
        if self.is_restore:
            # RESTORE: calibrate the prefill attention by token size and M-RoPE distance
            attention_mask = calibrated_attention_mask(
                config, position_ids, token_size_vis, n_sys=image_token_start).to(inputs_embeds.dtype)
        else:
            attention_mask = causal_attention_mask(inputs_embeds.shape[1], inputs_embeds.device, inputs_embeds.dtype)

        return inputs_embeds, input_ids, position_ids, attention_mask


def causal_attention_mask(seq_len, device, dtype):
    """Plain causal prefill mask (1, 1, seq_len, seq_len): 0 on and below the diagonal, -inf above."""
    causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    return torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=dtype).masked_fill(causal, float('-inf'))


def select_visual(ids, start, end, positions):
    """Keeps only the retained visual entries of input_ids (1, L) or position_ids (3, 1, L) along the last dim."""
    kept = ids[..., start:end][..., positions]
    return torch.cat([ids[..., :start], kept, ids[..., end:]], dim=-1)
