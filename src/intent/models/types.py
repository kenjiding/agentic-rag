"""Intent recognition type definitions.

General-purpose types for intent recognition, not specific to any domain.
"""
from typing import Literal

# Retrieval strategy options
PipelineOption = Literal["semantic", "hybrid", "rerank", "bm25"]

# Query decomposition types
DecompositionType = Literal[
    "comparison",
    "multi_hop",
    "dimensional",
    "temporal",
    "causal_chain",
    "information_needs"
]

# Intent types
IntentType = Literal[
    "factual",
    "comparison",
    "analytical",
    "procedural",
    "causal",
    "temporal",
    "multi_hop",
    "other"
]

# Complexity levels
ComplexityLevel = Literal["simple", "moderate", "complex"]
