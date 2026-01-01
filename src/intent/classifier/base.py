"""Base classifier interface.

Defines the abstract interface for intent classifiers.
"""
from abc import ABC, abstractmethod
from src.intent.models.query_intent import QueryIntent


class BaseIntentClassifier(ABC):
    """Abstract base class for intent classifiers.

    This class defines the interface that all intent classifiers must implement.
    Concrete implementations can use different approaches (LLM, rule-based, etc.).
    """

    @abstractmethod
    def classify(self, query: str) -> QueryIntent:
        """
        Classify the intent of a query.

        Args:
            query: User query string

        Returns:
            QueryIntent object containing the classification result including:
            - intent_type: The main type of intent
            - complexity: Query complexity level
            - needs_decomposition: Whether the query should be split
            - sub_queries: List of sub-queries if decomposition is needed
            - entities: Dict[str, Any] - 统一实体字典（包含通用实体和业务实体）
            - recommended_retrieval_strategy: Suggested retrieval approach
            - confidence: Confidence score (0-1)
            - reasoning: Explanation of the classification
        """
        pass
    
    @abstractmethod
    async def aclassify(self, query: str) -> QueryIntent:
        """
        Asynchronously classify the intent of a query.
        
        This method must be implemented by subclasses for async performance.

        Args:
            query: User query string

        Returns:
            QueryIntent object containing the classification result
        """
        pass