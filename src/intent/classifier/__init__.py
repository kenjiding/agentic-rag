"""Intent classifier module.

Exports base classifier interface and LLM-based implementation.
"""
from src.intent.classifier.base import BaseIntentClassifier
from src.intent.classifier.llm_classifier import IntentClassifier

__all__ = [
    "BaseIntentClassifier",
    "IntentClassifier",
]
