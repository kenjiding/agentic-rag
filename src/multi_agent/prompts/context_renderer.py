"""Context rendering utilities for prompts."""
import json
from typing import Any, Dict, Optional


def render_context_bundle(context_bundle: Optional[Dict[str, Any]]) -> str:
    """Render structured context bundle as JSON string."""
    if not context_bundle:
        return "上下文: {}"

    payload = {
        "short_term_context": context_bundle.get("short_term_context", {}),
        "task_input": context_bundle.get("task_input", {}),
        "metadata": context_bundle.get("metadata", {}),
    }
    return "上下文(JSON): " + json.dumps(payload, ensure_ascii=False)
