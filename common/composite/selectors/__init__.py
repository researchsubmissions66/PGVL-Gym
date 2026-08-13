"""Registry of patch-selector implementations."""
from .identity import IdentitySelector
from .focus_three_stage import FocusThreeStage
from .cod_chain_mask import CoDChainMask
from .mscpt_topk import MSCPTTopK

REGISTRY = {
    "identity":          IdentitySelector,
    "focus_three_stage": FocusThreeStage,
    "cod_chain_mask":    CoDChainMask,
    "mscpt_topk":        MSCPTTopK,
}


def build_selector(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"Unknown selector '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
