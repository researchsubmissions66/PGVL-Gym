"""Identity selector. Returns patches unchanged; useful for ablations."""
from __future__ import annotations
import torch
from common.composite.interfaces import PatchSelector


class IdentitySelector(PatchSelector):
    name = "identity"

    def forward(self, patches, text_features, coords=None):
        return patches
