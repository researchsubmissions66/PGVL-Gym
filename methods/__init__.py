"""Method registry.

The unified `train.py` calls `get_method(name)(cfg)` to obtain a
`BaseMethod` instance. Adding a new entry below is enough to expose
it on the command line.
"""
from __future__ import annotations
from typing import Type

from .base import BaseMethod
from common.backbones import MethodBackboneContract


_CANONICAL_METHODS = (
    "composite", "focus", "vila_mil", "cod_mil", "maple", "mscpt",
    "pathpt", "top", "slip", "wsi_five", "muse", "convlm", "sldpc",
)

_METHOD_ALIASES = {
    **{name: name for name in _CANONICAL_METHODS},
    "vilamil": "vila_mil",
    "vila": "vila_mil",
    "codmil": "cod_mil",
    "wsifive": "wsi_five",
    "five": "wsi_five",
}


def canonical_method_name(name: str) -> str:
    """Return the canonical registry name for a method or supported alias."""
    original = name
    if not isinstance(name, str) or not name.strip():
        raise KeyError(f"Invalid method name {original!r}")
    normalized = name.strip().lower().replace("-", "_")
    try:
        return _METHOD_ALIASES[normalized]
    except KeyError:
        raise KeyError(
            f"Unknown method '{original}'. Available: "
            f"{', '.join(_CANONICAL_METHODS)}."
        ) from None


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
    name = canonical_method_name(name)

    if name == "composite":
        from .composite.adapter import CompositeMethod
        return CompositeMethod

    if name == "focus":
        from .focus.adapter import FOCUSMethod
        return FOCUSMethod

    if name == "vila_mil":
        from .vila_mil.adapter import ViLaMILMethod
        return ViLaMILMethod

    if name == "cod_mil":
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

    if name == "wsi_five":
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

    raise AssertionError(f"Method registry is incomplete for {name!r}")


def list_methods() -> list[str]:
    """Return canonical method names in stable registry order."""
    return list(_CANONICAL_METHODS)


def get_backbone_contracts() -> dict[str, MethodBackboneContract]:
    """Return every adapter's declared encoder contract, keyed by method.

    Returns:
        A mapping from canonical registry name to its immutable
        :class:`common.backbones.MethodBackboneContract`.
    """
    return {name: get_method(name).get_backbone_contract()
            for name in list_methods()}
