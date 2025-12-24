"""Intent recognition configuration.

General-purpose configuration for intent recognition, decoupled from any specific system.
"""
from dataclasses import dataclass


@dataclass
class IntentConfig:
    """Configuration for intent recognition.

    This is a general-purpose configuration, not specific to any particular system.
    It can be used independently or integrated with other configurations.

    Attributes:
        llm_temperature: Temperature for LLM-based intent classification (lower = more stable)
        llm_model: Model name to use for intent classification
        enable_intent_classification: Whether to enable intent classification
        min_confidence: Minimum confidence threshold for classification results
    """
    # LLM settings
    llm_temperature: float = 0.0
    llm_model: str = "gpt-4o-mini"

    # Feature flags
    enable_intent_classification: bool = True

    # Thresholds
    min_confidence: float = 0.7

    @classmethod
    def default(cls) -> "IntentConfig":
        """Create default configuration."""
        return cls()

    @classmethod
    def from_dict(cls, config_dict: dict) -> "IntentConfig":
        """Create configuration from dictionary."""
        return cls(
            llm_temperature=config_dict.get("llm_temperature", 0.0),
            llm_model=config_dict.get("llm_model", "gpt-4o-mini"),
            enable_intent_classification=config_dict.get("enable_intent_classification", True),
            min_confidence=config_dict.get("min_confidence", 0.7),
        )
