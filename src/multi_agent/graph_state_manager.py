"""Graph状态管理器 - 封装状态初始化和恢复逻辑（一步一步智能模式）

将状态管理逻辑从主图类中分离，提高代码可维护性。

2025-2026 最佳实践：
- 每次请求都重新进行意图识别和路由决策
- 通过 entities 字段存储上下文信息
- 不依赖预先定义的任务链
"""
import logging
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.multi_agent.state import MultiAgentState
from src.multi_agent.utils import get_state_value, state_to_dict

logger = logging.getLogger(__name__)


class GraphStateManager:
    """图状态管理器 - 封装状态初始化和恢复逻辑（一步一步智能模式）"""

    def __init__(self, graph_instance):
        """
        初始化状态管理器

        Args:
            graph_instance: MultiAgentGraph实例，用于访问graph和配置
        """
        self.graph = graph_instance

    def create_initial_state(self, question: str) -> Dict[str, Any]:
        """创建初始状态（一步一步智能模式）"""
        return {
            "messages": [HumanMessage(content=question)],
            "current_agent": None,
            "agent_results": {},
            "agent_history": [],
            "tools_used": [],
            "metadata": {},
            "error_message": None,
            "iteration_count": 0,
            "max_iterations": self.graph.max_iterations,
            "next_action": None,
            "routing_reason": None,
            "query_intent": None,
            "original_question": question,
            "confirmation_pending": None,
            "entities": {},
            "last_product_search_context": None,
            "conversation_phase": "idle"
        }

    def prepare_config(
        self, config: Optional[Dict[str, Any]], session_id: str
    ) -> Dict[str, Any]:
        """准备执行配置"""
        if config is None:
            config = {}

        config.setdefault("recursion_limit", self.graph.max_iterations * 2)
        config.setdefault("configurable", {})
        config["configurable"].setdefault("session_id", session_id)
        config["configurable"].setdefault("thread_id", session_id)

        return config

    def restore_state_from_checkpointer(
        self, config: Dict[str, Any], question: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """从checkpointer恢复状态（一步一步智能模式）"""
        try:
            existing_snapshot = self.graph.graph.get_state(config)
            if existing_snapshot and existing_snapshot.values:
                logger.info(f"从 checkpointer 恢复状态")

                existing_dict = state_to_dict(existing_snapshot.values)
                if "messages" not in existing_dict or not existing_dict["messages"]:
                    existing_dict["messages"] = []

                if question is not None:
                    existing_dict["messages"].append(HumanMessage(content=question))

                return existing_dict
        except Exception as e:
            logger.warning(f"从 checkpointer 获取状态失败: {e}")

        return None

    def get_initial_state_for_stream(
        self, question: Optional[str], config: Dict[str, Any], session_id: str
    ) -> Dict[str, Any]:
        """获取流式执行的初始状态（一步一步智能模式）"""
        # 尝试从checkpointer恢复
        restored_state = self.restore_state_from_checkpointer(config, question)
        if restored_state:
            return restored_state

        # 如果没有question且无法恢复状态，报错
        if question is None:
            raise ValueError("question 不能为 None，且 checkpointer 中无现有状态")

        # 创建新状态
        logger.info(f"未找到现有状态，创建新状态: session_id={session_id}")
        return self.create_initial_state(question)

    def log_resume_state(self, config: Dict[str, Any], command: Command):
        """记录恢复执行前的状态（调试用）"""
        try:
            logger.info(f"[恢复执行] 使用 Command 恢复执行: resume={command.resume}")
            existing_snapshot = self.graph.graph.get_state(config)
            logger.info(f"[恢复执行] 检查 checkpointer 状态: snapshot={existing_snapshot is not None}")

            if existing_snapshot and existing_snapshot.values:
                state_dict = state_to_dict(existing_snapshot.values)
                logger.info(f"[恢复执行] checkpointer 状态键: {list(state_dict.keys()) if state_dict else 'None'}")
                logger.info(f"[恢复执行] entities: {state_dict.get('entities', {})}")
                logger.info(f"[恢复执行] last_product_search_context: {state_dict.get('last_product_search_context')}")
        except Exception as e:
            logger.error(f"[恢复执行] 检查状态失败: {e}", exc_info=True)
