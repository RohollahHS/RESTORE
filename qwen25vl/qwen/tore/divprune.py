"""DivPrune (https://github.com/vbdi/divprune): greedy max-min diversity selection on cosine distance."""
import torch


def pairwise_cosine_similarity(matrix):
    norm_matrix = matrix / matrix.norm(dim=1, keepdim=True)
    cosine_similarity = torch.mm(norm_matrix, norm_matrix.t())
    return cosine_similarity


def DivPrune(visual_feature_vectors, image_feature_length, cosine_matrix=None, threshold_n=64):
    """
    Iteratively selects the token that is most distant from all previously selected tokens.

    Args:
        visual_feature_vectors: (N, D) — visual token features
        image_feature_length: int — total number of visual tokens (N)
        cosine_matrix: optional pre-computed cosine distance matrix (1 - cosine_sim)
        threshold_n: int — number of tokens to select

    Returns:
        s: (threshold_n,) — selected token indices (in greedy selection order, NOT sorted)
    """
    threshold_terms = threshold_n

    if cosine_matrix is None:
        cosine_matrix = 1.0 - pairwise_cosine_similarity(visual_feature_vectors)

    s = torch.empty(threshold_terms, dtype=torch.long, device=visual_feature_vectors.device)
    for i in range(threshold_terms):
        if i == 0:
            m2 = cosine_matrix
        else:
            m2 = torch.index_select(cosine_matrix, 0, torch.index_select(s, 0, torch.arange(0, i, device=cosine_matrix.device)))

        if i == 0:
            scores = torch.topk(m2, 2, dim=0, largest=False).values[1, :]
        else:
            scores = torch.min(m2, dim=0).values

        phrase_to_add_idx = torch.argmax(scores)
        s[i] = phrase_to_add_idx

    return s


def divprune_select(image_tokens, image_attn, n_retain):
    """
    Args:
        image_tokens: (1, N, D)
        image_attn:   unused (DivPrune only looks at the features)
        n_retain:     number of tokens to keep
    Returns:
        retained_tokens: (1, n_retain, D), retained_position: (n_retain,) sorted indices into N
    """
    selected_indices = DivPrune(
        visual_feature_vectors=image_tokens[0],
        image_feature_length=image_tokens.shape[1],
        threshold_n=n_retain,
    )

    selected_indices_sorted, _ = torch.sort(selected_indices)
    retained_tokens = image_tokens[:, selected_indices_sorted, :]

    return retained_tokens, selected_indices_sorted
