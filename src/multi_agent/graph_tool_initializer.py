"""Graph工具初始化器 - 封装工具初始化逻辑

将工具初始化逻辑从主图类中分离，提高代码可维护性。
"""
import logging
from src.multi_agent.tools.tool_registry import ToolCategory, ToolPermission
from src.tools.web_search import create_web_search_tool

logger = logging.getLogger(__name__)


class GraphToolInitializer:
    """图工具初始化器 - 封装工具初始化逻辑"""
    
    def __init__(self, tool_registry):
        """初始化工具初始化器
        
        Args:
            tool_registry: 工具注册表
        """
        self.tool_registry = tool_registry
        self._web_search_initialized = False
    
    def init_web_search_tools(self) -> bool:
        """初始化web search tools（同步）"""
        try:
            web_search_tools = create_web_search_tool()
            
            if web_search_tools:
                for tool in web_search_tools:
                    self.tool_registry.register_tool(
                        name=tool.name,
                        tool=tool,
                        category=ToolCategory.SEARCH,
                        permission=ToolPermission.PUBLIC,
                        allowed_agents=["chat_agent", "rag_agent"]
                    )
                logger.info(f"成功注册 {len(web_search_tools)} 个web search tools（基于 DDGS）")
                self._web_search_initialized = True
                return True
            else:
                logger.warning("Web search tools返回为空")
                self._web_search_initialized = False
                return False
        except Exception as e:
            logger.warning(f"Web search tools初始化失败: {e}", exc_info=True)
            self._web_search_initialized = False
            return False
    
    async def async_init_web_search_tools(self, supervisor_agents=None) -> bool:
        """异步初始化web search tools"""
        if self._web_search_initialized:
            logger.info("Web search tools已经初始化")
            return True
        
        try:
            web_search_tools = create_web_search_tool()
            if web_search_tools:
                for tool in web_search_tools:
                    self.tool_registry.register_tool(
                        name=tool.name,
                        tool=tool,
                        category=ToolCategory.SEARCH,
                        permission=ToolPermission.PUBLIC,
                        allowed_agents=["chat_agent", "rag_agent"]
                    )
                logger.info(f"成功注册 {len(web_search_tools)} 个web search tools（基于 DDGS）")
                self._web_search_initialized = True
                
                # 刷新所有已注册的agent的工具列表
                if supervisor_agents:
                    for agent in supervisor_agents:
                        if hasattr(agent, 'refresh_tools'):
                            agent.refresh_tools()
                return True
            else:
                logger.warning("Web search tools返回为空")
                self._web_search_initialized = False
                return False
        except Exception as e:
            logger.warning(f"异步初始化web search tools失败: {e}", exc_info=True)
            self._web_search_initialized = False
            return False
    
    @property
    def web_search_initialized(self) -> bool:
        """检查web search tools是否已初始化"""
        return self._web_search_initialized

