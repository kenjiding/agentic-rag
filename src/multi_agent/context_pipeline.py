"""Unified context pipeline for multi-agent system.

Centralizes context construction, compression, and shaping into a single
structure to avoid duplicated logic across nodes and agents.
"""
import json
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from src.multi_agent.context_manager import ContextManager, ContextSummary
from src.multi_agent.state import MultiAgentState
from src.multi_agent.constants import MetadataKeys


class UnifiedContext(BaseModel):
    """Structured context bundle with clear boundaries."""
    long_term_memory: Dict[str, Any] = Field(default_factory=dict)
    short_term_context: Dict[str, Any] = Field(default_factory=dict)
    task_input: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_prompt_context(self) -> Dict[str, Any]:
        """Return structured data for prompt rendering."""
        return {
            "long_term_memory": self.long_term_memory,
            "short_term_context": self.short_term_context,
            "task_input": self.task_input,
            "metadata": self.metadata,
        }


class ContextPipeline:
    """Builds unified context from state + current query.

    Keeps ContextManager as the single source for conversation compression,
    and wraps the output into a structured context bundle.
    """

    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager

    async def build(
        self, state: MultiAgentState, current_query: str
    ) -> UnifiedContext:
        """Build context summary + unified context bundle.

        Uses a lightweight cache keyed by message count to avoid recomputation
        when no new messages are added.
        """
        # Reuse last summary only if state fingerprint unchanged
        current_fingerprint = self._compute_fingerprint(state, current_query)
        last_fingerprint = (
            (state.context_bundle or {})
            .get("metadata", {})
            .get("fingerprint")
        )
        if last_fingerprint == current_fingerprint and state.context_bundle:
            return UnifiedContext(**state.context_bundle)

        summary: ContextSummary = await self.context_manager.build_context_summary(
            state=state,
            current_query=current_query,
        )

        unified = UnifiedContext(
            long_term_memory={},  # reserved for future persistent memory
            short_term_context=summary.model_dump(),
            task_input={
                "query": current_query,
                "entities": state.entities,
                "intent": state.query_intent,
                "conversation_phase": state.conversation_phase,
            },
            metadata={
                "pipeline_version": "1.0",
                "summary_strategy": summary.metadata.get("extraction_strategy", "rule_based"),
                "fingerprint": current_fingerprint,
                "message_count": len(state.messages),
            },
        )

        return unified

    @staticmethod
    def _compute_fingerprint(state: MultiAgentState, current_query: str) -> str:
        """Compute a stable fingerprint of state inputs affecting context."""
        payload = {
            "message_count": len(state.messages),
            "current_query": current_query,
            "entities": state.entities or {},
            "query_intent": state.query_intent or {},
            "conversation_phase": state.conversation_phase,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
