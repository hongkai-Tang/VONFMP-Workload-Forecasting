"""Controlled five-variant ablation layer for the Alibaba Ours experiment."""

from .model import (
    ABLATION_VARIANTS,
    AblationModel,
    AblationModelConfig,
    canonical_variant,
)

__all__ = [
    "ABLATION_VARIANTS",
    "AblationModel",
    "AblationModelConfig",
    "canonical_variant",
]
