"""Registry of aggregators + fusion modes."""
from .attn_pool import AttnPoolAggregator
from .vila_prototypes import ViLaPrototypeAggregator
from .slip_routing import SLIPRoutingAggregator
from .pathpt_conv1d import PathPTConv1DAggregator
from .focus_mha import FocusMHAAggregator
from .transmil import TransMILAggregator
from .fusion import LogitEnsemble, VectorFusion

REGISTRY = {
    "attn_pool":         AttnPoolAggregator,
    "vila_prototypes":   ViLaPrototypeAggregator,
    "slip_routing":      SLIPRoutingAggregator,
    "pathpt_conv1d":     PathPTConv1DAggregator,
    "focus_mha":         FocusMHAAggregator,
    "transmil":          TransMILAggregator,
}


def build_aggregator(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"Unknown aggregator '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
