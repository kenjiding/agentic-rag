"""Product Agent - 商品搜索 Agent

处理与商品相关的查询：
- 商品搜索（支持多条件筛选）
- 商品详情查询
- 品牌/分类查询
"""

import json
import logging
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from src.tools.product_tools import get_product_tools
from src.multi_agent.state import MultiAgentState
from src.multi_agent.utils import clean_messages_for_llm

logger = logging.getLogger(__name__)


# System Prompt
PRODUCT_AGENT_SYSTEM_PROMPT = """你是一个专业的电商客服助手 - 商品查询专家。

你的职责是帮助用户查找商品信息，包括：
1. 根据用户需求搜索商品（支持价格、品牌、分类等多条件筛选）
2. 提供商品详细信息
3. 推荐符合条件的商品

重要业务规则：
- 优先展示评分高、有库存的商品
- 如果用户提供的搜索条件过于严格导致无结果，建议放宽条件
- 所有工具参数都是可选的，根据用户输入动态构建查询

回复风格：
- 使用友好的语气，用 emoji 让回复更生动
- 如果找到多个结果，用列表展示
- 如果没有找到，给出建议（如放宽筛选条件）
- 主动询问用户是否需要更详细的信息
"""


class ProductAgent:
    """商品搜索 Agent

    使用 LangGraph 模式，集成商品搜索工具。
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        tools: list | None = None,
    ):
        """初始化 Product Agent

        Args:
            llm: LangChain LLM 实例
            tools: 商品工具列表，默认使用内置工具
        """
        self.llm = llm or ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
        )
        self.tools = tools or get_product_tools()
        self.name = "product_agent"

        # 绑定工具到 LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def get_name(self) -> str:
        """获取 Agent 名称"""
        return self.name

    def get_description(self) -> str:
        """获取 Agent 描述"""
        return "商品搜索专家 - 处理商品查询、搜索、比价等请求"

    def _build_system_prompt_hints(self, state: MultiAgentState) -> str:
        """构建系统提示的上下文信息
        
        企业级最佳实践：通过 system prompt 提示 LLM 上下文信息，
        让 LLM 自己判断如何使用工具，而不是硬编码工具调用。
        
        Args:
            state: 当前多Agent状态

        Returns:
            上下文提示字符串
        """
        hints = []

        # 检查任务链上下文
        task_chain = state.task_chain
        entities = state.entities

        # 如果有任务链或实体信息，提示 LLM
        if task_chain and entities:
            hints.append("\n\n=== 任务链上下文 ===")
            if entities.get("search_keyword"):
                search_keyword = entities["search_keyword"]
                hints.append(f"当前处于任务链的商品搜索步骤。")
                hints.append(f"需要搜索关键词：{search_keyword}")
                hints.append("请使用 search_products_tool 工具执行搜索，根据工具描述选择合适的参数。")

            # 显示其他��下文信息
            other_context = {k: v for k, v in entities.items() if k != "search_keyword" and v is not None}
            if other_context:
                hints.append("\n其他上下文信息：")
                for key, value in other_context.items():
                    hints.append(f"- {key}: {value}")

        return "\n".join(hints) if hints else ""

    async def execute(self, state: MultiAgentState) -> Dict[str, Any]:
        """执行商品查询（异步接口，符合LangGraph 1.x规范）

        企业级最佳实践：让 LLM 自己决定使用哪些工具，而不是硬编码工具调用。
        LLM 会根据工具描述和上下文，自动判断需要调用什么工具。

        Args:
            state: 当前多 Agent 状态

        Returns:
            更新后的状态片段
        """
        # 获取最新消息
        messages = state.messages
        if not messages:
            return {
                "messages": [
                    AIMessage(content="您好！我是商品查询助手，请问有什么可以帮您？")
                ],
                "current_agent": self.name,
            }

        # 构建系统提示（包含任务链上下文）
        hints = self._build_system_prompt_hints(state)
        system_prompt = PRODUCT_AGENT_SYSTEM_PROMPT + hints
        
        # 构建 Agent 消息
        # 使用最新的用户消息和最近的几轮对话
        # 清理消息历史，确保消息序列完整性（过滤无效的 ToolMessage）
        cleaned_messages = clean_messages_for_llm(messages, keep_recent_n=5)
        
        agent_messages = [SystemMessage(content=system_prompt)]
        agent_messages.extend(cleaned_messages)

        # 调用 LLM（异步执行）
        response = await self.llm_with_tools.ainvoke(agent_messages)

        # 处理工具调用
        if hasattr(response, "tool_calls") and response.tool_calls:
            # 执行工具调用并构建 ToolMessage
            tool_messages = []
            tool_used_info = []
            structured_result = None  # 存储结构化数据结果

            for tool_call in response.tool_calls:
                tool = next((t for t in self.tools if t.name == tool_call["name"]), None)
                if tool:
                    try:
                        result = await tool.ainvoke(tool_call["args"])

                        # 尝试解析工具返回的结构化数据
                        try:
                            result_json = json.loads(result)
                            if isinstance(result_json, dict):
                                # 检查是否包含结构化数据（products/product/orders/brands/categories）
                                if any(key in result_json for key in ["products", "product", "orders", "brands", "categories"]):
                                    structured_result = result_json
                        except (json.JSONDecodeError, TypeError):
                            pass

                        # 构建 ToolMessage
                        tool_messages.append(
                            ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call["id"],
                            )
                        )
                        tool_used_info.append({
                            "agent": self.name,
                            "tool": tool_call["name"],
                            "args": tool_call["args"],
                        })
                    except Exception as e:
                        # 工具调用失败，也需要返回 ToolMessage
                        tool_messages.append(
                            ToolMessage(
                                content=f"错误: {str(e)}",
                                tool_call_id=tool_call["id"],
                            )
                        )

            # 如果有结构化数据，直接使用工具的 text，不再调用 LLM
            if structured_result and "text" in structured_result:
                # 使用工具返回的简短文本，避免 LLM 重新生成长文本
                final_response = AIMessage(content=structured_result["text"])
            else:
                # 没有结构化数据，需要调用 LLM 生成回复
                followup_messages = agent_messages + [response] + tool_messages
                final_response = await self.llm.ainvoke(followup_messages)

            # 返回结果（只添加新的 AIMessage 和 ToolMessage）
            result = {
                "messages": messages + [response] + tool_messages + [final_response],
                "current_agent": self.name,
                "tools_used": state.tools_used + tool_used_info,
            }

            return result

        # 无工具调用，直接返回响应
        result = {
            "messages": messages + [response],
            "current_agent": self.name,
        }

        return result
