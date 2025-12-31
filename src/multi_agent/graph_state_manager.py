"""Graph状态管理器 - 封装状态初始化和恢复逻辑

将状态管理逻辑从主图类中分离，提高代码可维护性。
"""
import logging
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.multi_agent.state import MultiAgentState
from src.multi_agent.utils import get_state_value, state_to_dict

logger = logging.getLogger(__name__)


class GraphStateManager:
    """图状态管理器 - 封装状态初始化和恢复逻辑"""
    
    def __init__(self, graph_instance):
        """初始化状态管理器
        
        Args:
            graph_instance: MultiAgentGraph实例，用于访问graph和配置
        """
        self.graph = graph_instance
    
    def create_initial_state(self, question: str) -> Dict[str, Any]:
        """创建初始状态"""
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
            "task_chain": None,
            "pending_selection": None
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
        """从checkpointer恢复状态"""
        try:
            existing_snapshot = self.graph.graph.get_state(config)
            if existing_snapshot and existing_snapshot.values:
                task_chain = get_state_value(existing_snapshot.values, "task_chain")
                logger.info(f"从 checkpointer 恢复状态: task_chain={task_chain is not None}")

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
        """获取流式执行的初始状态"""
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
                task_chain = get_state_value(existing_snapshot.values, "task_chain")
                logger.info(f"[恢复执行] checkpointer 状态键: {list(state_dict.keys()) if state_dict else 'None'}")
                logger.info(f"[恢复执行] checkpointer 中的 task_chain: {task_chain is not None}")
                if task_chain:
                    logger.info(f"[恢复执行] task_chain 详情: chain_id={task_chain.chain_id}, current_step_index={task_chain.current_step_index}, steps_count={len(task_chain.steps)}")
        except Exception as e:
            logger.error(f"[恢复执行] 检查状态失败: {e}", exc_info=True)

