"""3D generation, as a provider capability.

The agent asks for an asset; which service makes it is the registry's business.
Tripo is the MVP's provider and nothing above this layer knows it exists.
"""

from __future__ import annotations

from app.providers3d.base import ThreeDProvider
from app.providers3d.models import (
    AssetKind,
    AssetRequest,
    AssetResult,
    ProviderTask,
    TaskStatus,
)
from app.providers3d.registry import ThreeDRegistry
from app.providers3d.tripo import TripoProvider

__all__ = [
    "AssetKind",
    "AssetRequest",
    "AssetResult",
    "ProviderTask",
    "TaskStatus",
    "ThreeDProvider",
    "ThreeDRegistry",
    "TripoProvider",
]
