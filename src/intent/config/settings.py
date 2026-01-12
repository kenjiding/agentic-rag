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
    llm_model: str = "openai:gpt-4o-mini"  # 格式: provider:model_name，支持 openai, anthropic, gemini, ollama, tongyi, moonshot

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
        # 兼容旧格式：如果模型名称不包含 provider，默认为 openai
        llm_model = config_dict.get("llm_model", "openai:gpt-4o-mini")
        if ":" not in llm_model:
            llm_model = f"openai:{llm_model}"
        
        return cls(
            llm_temperature=config_dict.get("llm_temperature", 0.0),
            llm_model=llm_model,
            enable_intent_classification=config_dict.get("enable_intent_classification", True),
            min_confidence=config_dict.get("min_confidence", 0.7),
        )
