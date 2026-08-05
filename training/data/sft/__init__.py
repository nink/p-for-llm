"""Deterministic masked-target SFT data preparation."""

from .pool import (
    SFTDataPool,
    SFTExample,
    build_event_sft_example,
    build_sft_example,
    load_or_prepare_sft_pool,
    load_sft_pool,
)
from .squad_v2 import convert_squad_v2

__all__ = [
    "SFTDataPool",
    "SFTExample",
    "build_event_sft_example",
    "build_sft_example",
    "convert_squad_v2",
    "load_or_prepare_sft_pool",
    "load_sft_pool",
]
