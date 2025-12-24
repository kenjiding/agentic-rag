"""Intent recognition module.

General-purpose intent recognition and query decomposition system.
Can be used independently or integrated with other systems like agentic_rag.

This module provides:
- IntentClassifier: LLM-based intent classification
- QueryIntent, SubQuery: Data models for query intent representation
- Type definitions for intent, complexity, decomposition, and retrieval strategies
- IntentConfig: Configuration for intent classification

Example usage:
    from src.intent import IntentClassifier

    classifier = IntentClassifier()
    intent = classifier.classify("What is the population of Beijing?")
    print(intent.intent_type)  # "factual"
    print(intent.complexity)  # "simple"
"""

# Main exports
from src.intent.classifier.base import BaseIntentClassifier
from src.intent.classifier.llm_classifier import IntentClassifier
from src.intent.models import (
    QueryIntent,
    SubQuery,
    PipelineOption,
    DecompositionType,
    IntentType,
    ComplexityLevel
)
from src.intent.config import IntentConfig

__all__ = [
    # Main classifier
    "IntentClassifier",
    "BaseIntentClassifier",

    # Models
    "QueryIntent",
    "SubQuery",

    # Types
    "PipelineOption",
    "DecompositionType",
    "IntentType",
    "ComplexityLevel",

    # Config
    "IntentConfig",
]
