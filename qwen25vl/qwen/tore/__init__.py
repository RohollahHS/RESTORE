"""Visual token reduction for Qwen2.5-VL, organized like RESTORE's `tokenreduction/tore` for LLaVA-1.5.

    model = load_reduction_base(model, base="HoloV", retain_ratio=0.111, is_restore=True)

- base:         pruning baseline, one of BASES (None keeps every visual token)
- retain_ratio: fraction of visual tokens kept (Table 3 of the paper: 0.333 / 0.222 / 0.111)
- is_restore:   apply RESTORE on top of the baseline. Half of the budget is selected by the baseline,
                the other half is merged from the pruned tokens, and the LLM attention is calibrated
                by token size and M-RoPE distance. Without it the baseline prunes to the full budget.

A new baseline only needs a `select(image_tokens, image_attn, n_retain)` function registered in BASES.
"""
from .divprune import divprune_select
from .holov import holov_select
from .reduction import VisualTokenReduction
from .vispruner import vispruner_select

BASES = {
    "HoloV": holov_select,
    "DivPrune": divprune_select,
    "VisPruner": vispruner_select,
}


def load_reduction_base(model, base, retain_ratio, is_restore):
    if base is None:
        if is_restore:
            raise ValueError("--restore needs a base reduction method (--base)")
        return model
    if base not in BASES:
        raise ValueError(f"Undefined base reduction method: {base}")
    if not 0.0 < retain_ratio < 1.0:
        raise ValueError(f"--retain-ratio must be in (0, 1) when a base is given, got {retain_ratio}")

    print(f"========== Base reduction method: {base} | retain_ratio: {retain_ratio} | restore: {is_restore} ==========")
    model.reduction = VisualTokenReduction(BASES[base], retain_ratio, is_restore)
    return model
