"""Registry of prompt modules + fusion."""
from .coop_flat import CoOpFlatPrompts
from .top_two_level import TOPTwoLevelPrompts
from .maple_graph import MAPLEGraphPrompts
from .cod_chain import CoDChainPrompts
from .slip_tissue import SLIPTissuePrompts
from .fusion import PromptFusion

REGISTRY = {
    "coop_flat":        CoOpFlatPrompts,
    "top_two_level":    TOPTwoLevelPrompts,
    "maple_graph":      MAPLEGraphPrompts,
    "cod_chain":        CoDChainPrompts,
    "slip_tissue":      SLIPTissuePrompts,
}


def build_prompt_module(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"Unknown prompt module '{name}'. "
                       f"Available: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
