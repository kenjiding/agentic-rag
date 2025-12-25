"""Product Agent - 商品搜索 Agent

处理与商品相关的查询：
- 商品搜索（支持多条件筛选）
- 商品详情查询
- 品牌/分类查询
"""

import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage

from src.tools.product_tools import get_product_tools
from src.multi_agent.state import MultiAgentState


# System Prompt
PRODUCT_AGENT_SYSTEM_PROMPT = """你是一个专业的电商客服助手 - 商品查询专家。

你的职责是帮助用户查找商品信息，包括：
1. 根据用户需求搜索商品（支持价格、品牌、分类等多条件筛选）
2. 提供商品详细信息
3. 推荐符合条件的商品

工具使用指南：
- search_products_tool: 主要工具，支持多条件搜索
  * name: 商品名称关键词
  * category/sub_category: 分类筛选
  * brand: 品牌筛选
  * price_min/price_max: 价格范围
  * min_rating: 最低评分
  * in_stock_only: 是否仅显示有货

- get_product_detail: 查询指定商品的详细信息
- get_brands: 获取所有可用品牌
- get_categories: 获取所有可用分类

回复风格：
- 使用友好的语气，用 emoji 让回复更生动
- 如果找到多个结果，用列表展示
- 如果没有找到，给出建议（如放宽筛选条件）
- 主动询问用户是否需要更详细的信息

注意事项：
- 所有参数都是可选的，根据用户输入动态构建查询
- 如果用户提供的搜索条件过于严格导致无结果，建议放宽条件
- 优先展示评分高、有库存的商品
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

    def invoke(self, state: MultiAgentState) -> Dict[str, Any]:
        """执行商品查询

        Args:
            state: 当前多 Agent 状态

        Returns:
            更新后的状态片段
        """
        # 获取最新消息
        messages = state.get("messages", [])
        if not messages:
            return {
                "messages": [
                    AIMessage(content="您好！我是商品查询助手，请问有什么可以帮您？")
                ],
                "current_agent": self.name,
            }

        # 构建 Agent 消息
        agent_messages = [SystemMessage(content=PRODUCT_AGENT_SYSTEM_PROMPT)]
        agent_messages.extend(messages)

        # 调用 LLM
        response = self.llm_with_tools.invoke(agent_messages)

        # 处理工具调用
        if hasattr(response, "tool_calls") and response.tool_calls:
            # 执行工具调用并构建 ToolMessage
            tool_messages = []
            tool_used_info = []

            for tool_call in response.tool_calls:
                tool = next((t for t in self.tools if t.name == tool_call["name"]), None)
                if tool:
                    try:
                        result = tool.invoke(tool_call["args"])
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

            # 构建后续消息列表（包含 tool_calls 的 assistant 消息 + ToolMessage）
            followup_messages = agent_messages + [response] + tool_messages

            # 再次调用 LLM 生成最终回复
            final_response = self.llm.invoke(followup_messages)

            # 返回结果（只添加新的 AIMessage，不重复添加 response）
            return {
                "messages": messages + [final_response],
                "current_agent": self.name,
                "tools_used": state.get("tools_used", []) + tool_used_info,
            }

        # 无工具调用，直接返回响应
        return {
            "messages": messages + [response],
            "current_agent": self.name,
        }


# 兼容 LangGraph 节点函数
def product_agent_node(state: MultiAgentState, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """LangGraph 节点函数 - 商品 Agent

    Args:
        state: 当前状态
        config: 配置（可包含 llm 实例）

    Returns:
        状态更新
    """
    llm = config.get("llm") if config else None
    agent = ProductAgent(llm=llm)
    return agent.invoke(state)
