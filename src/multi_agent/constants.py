"""Shared constants and enums for multi-agent system."""
from enum import Enum


class ActionName(str, Enum):
    RAG_SEARCH = "rag_search"
    CHAT = "chat"
    PRODUCT_SEARCH = "product_search"
    ORDER_MANAGEMENT = "order_management"
    CONSULTATION = "consultation"
    BROWSER_SEARCH = "browser_search"
    FINISH = "finish"


class AgentName(str, Enum):
    RAG_AGENT = "rag_agent"
    CHAT_AGENT = "chat_agent"
    PRODUCT_AGENT = "product_agent"
    ORDER_AGENT = "order_agent"
    CONSULTATION_AGENT = "consultation_agent"
    BROWSER_AGENT = "browser_agent"


class MetadataKeys(str, Enum):
    CONTEXT_CACHE = "context_cache"
    CONTEXT_VERSION = "context_version"
    CONTEXT_OWNER = "context_owner"


class SystemNodeName(str, Enum):
    """Reserved system node names used in the LangGraph."""

    CONTEXT_MANAGER = "context_manager"
    INTENT_ROUTER = "intent_router"
    SUPERVISOR = "supervisor"
    POLICY_GATE = "policy_gate"
    PLANNER = "planner"
    PLAN_EXECUTOR = "plan_executor"
    POST_ACTION_VERIFIER = "post_action_verifier"
