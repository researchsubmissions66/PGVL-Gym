"""Method registry.

The unified `train.py` calls `get_method(name)(cfg)` to obtain a
`BaseMethod` instance. Adding a new entry below is enough to expose
it on the command line.
"""
from __future__ import annotations
from typing import Type

from .base import BaseMethod
from common.backbones import MethodBackboneContract


def get_method(name: str) -> Type[BaseMethod]:
    """Resolve a method name or supported alias to its adapter class.

    Imports are intentionally lazy so registry inspection does not initialize
    foundation models or require every method's optional dependencies.

    Args:
        name: Canonical method name or a documented command-line alias.

    Returns:
        The matching :class:`methods.base.BaseMethod` subclass.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    name = name.lower().replace("-", "_")

    if name == "composite":
        from .composite.adapter import CompositeMethod
        return CompositeMethod

    if name == "focus":
        from .focus.adapter import FOCUSMethod
        return FOCUSMethod

    if name in {"vila_mil", "vilamil", "vila"}:
        from .vila_mil.adapter import ViLaMILMethod
        return ViLaMILMethod

    if name in {"cod_mil", "codmil"}:
        from .cod_mil.adapter import CoDMILMethod
        return CoDMILMethod

    if name == "maple":
        from .maple.adapter import MAPLEMethod
        return MAPLEMethod

    if name == "mscpt":
        from .mscpt.adapter import MSCPTMethod
        return MSCPTMethod

    if name == "pathpt":
        from .pathpt.adapter import PathPTMethod
        return PathPTMethod

    if name == "top":
        from .top.adapter import TOPMethod
        return TOPMethod

    if name == "slip":
        from .slip.adapter import SLIPMethod
        return SLIPMethod

    if name in {"wsi_five", "wsifive", "five"}:
        from .wsi_five.adapter import WSIFiVEMethod
        return WSIFiVEMethod

    if name == "muse":
        from .muse.adapter import MUSEMethod
        return MUSEMethod

    if name == "convlm":
        from .convlm.adapter import ConVLMMethod
        return ConVLMMethod

    if name == "sldpc":
        from .sldpc.adapter import SLDPCMethod
        return SLDPCMethod

    raise KeyError(
        f"Unknown method '{name}'. "
        f"Available: focus, vila_mil, cod_mil, maple, mscpt, "
        f"pathpt, top, slip, wsi_five, muse, convlm, sldpc.")


def list_methods() -> list[str]:
    """Return canonical method names in stable registry order."""
    return ["composite", "focus", "vila_mil", "cod_mil", "maple",
            "mscpt", "pathpt", "top", "slip", "wsi_five", "muse",
            "convlm", "sldpc"]


def get_backbone_contracts() -> dict[str, MethodBackboneContract]:
    """Return every adapter's declared encoder contract, keyed by method.

    Returns:
        A mapping from canonical registry name to its immutable
        :class:`common.backbones.MethodBackboneContract`.
    """
    return {name: get_method(name).get_backbone_contract()
            for name in list_methods()}
