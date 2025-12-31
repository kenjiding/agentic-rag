"""MultiAgentGraph 实例管理"""
import asyncio
import logging
from typing import Optional
from src.multi_agent.graph import MultiAgentGraph
from src.multi_agent.config import MultiAgentConfig

logger = logging.getLogger(__name__)

# 全局 MultiAgentGraph 实例
_graph: Optional[MultiAgentGraph] = None
_graph_initializing = False
_graph_lock = asyncio.Lock()


async def get_graph() -> MultiAgentGraph:
    """获取或创建 MultiAgentGraph 实例（异步，支持并发安全）"""
    global _graph, _graph_initializing

    logger.info(f"[get_graph] 调用: _graph is {_graph}, _graph_initializing={_graph_initializing}")

    if _graph is not None:
        logger.info(f"[get_graph] 返回现有 graph 实例: id={id(_graph)}, checkpointer id={id(_graph.checkpointer)}")
        return _graph

    logger.info(f"[get_graph] 创建新的 graph 实例")
    async with _graph_lock:
        # 双重检查，避免重复初始化
        if _graph is not None:
            return _graph
        
        if _graph_initializing:
            # 如果正在初始化，等待完成
            while _graph_initializing:
                await asyncio.sleep(0.1)
            return _graph
        
        _graph_initializing = True
        try:
            config = MultiAgentConfig()
            loop = asyncio.get_event_loop()
            
            def init_graph():
                return MultiAgentGraph(
                    llm=None,
                    max_iterations=config.max_iterations
                )
            
            _graph = await loop.run_in_executor(None, init_graph)
            return _graph
        except Exception as e:
            logger.error(f"MultiAgentGraph 初始化失败: {e}", exc_info=True)
            raise
        finally:
            _graph_initializing = False


def get_graph_status() -> str:
    """获取 Graph 状态"""
    global _graph, _graph_initializing
    if _graph is not None:
        return "initialized"
    elif _graph_initializing:
        return "initializing"
    else:
        return "not_started"

