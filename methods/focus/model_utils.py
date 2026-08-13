"""Shim that re-exports the shared CLAM blocks under the original
`model_utils` namespace expected by `models/model_*.py`."""
from common.models._clam_blocks import *  # noqa: F401,F403
