"""Streaming pretraining data sources and token packing."""

from .packing import (
    iter_profile_batches,
    iter_profile_documents,
    iter_profile_sequences,
    iter_split_documents,
    iter_token_sequences,
)
from .capability import CapabilityPlan, CapabilityWave, load_capability_plan
from .pool import PackedDataPool, load_or_prepare_packed_pool
from .sources import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_TOKENIZER_PATH,
    DataProfile,
    DataSource,
    discover_data_profile,
    iter_source_documents,
    load_data_profile,
)

__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_TOKENIZER_PATH",
    "DataProfile",
    "DataSource",
    "discover_data_profile",
    "CapabilityPlan",
    "CapabilityWave",
    "PackedDataPool",
    "iter_profile_batches",
    "iter_profile_documents",
    "iter_profile_sequences",
    "iter_split_documents",
    "iter_source_documents",
    "iter_token_sequences",
    "load_data_profile",
    "load_capability_plan",
    "load_or_prepare_packed_pool",
]
