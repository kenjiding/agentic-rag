"""Product Agent - 商品搜索 Agent

处理与商品相关的查询：
- 商品搜索（支持多条件筛选）
- 商品详情查询
- 品牌/分类查询
"""

import json
import logging
from typing import Any, Dict

from langchain_core.language_models import BaseChatModel
from src.utils.llm_factory import create_llm_for_agent
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from src.tools.product_tools import get_product_tools
from src.multi_agent.state import MultiAgentState
from src.multi_agent.utils import clean_messages_for_llm
from src.multi_agent.response_models import ProductListResponse, TextResponse
from src.multi_agent.prompts import prompt_registry, render_context_bundle
from src.multi_agent.constants import AgentName

logger = logging.getLogger(__name__)


PRODUCT_AGENT_SYSTEM_PROMPT = """你是一个专业的电商客服助手 - 商品查询专家。"""


class ProductAgent:
    """商品搜索 Agent

    使用 LangGraph 模式，集成商品搜索工具。
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        tools: list | None = None,
    ):
        """初始化 Product Agent

        Args:
            llm: LangChain LLM 实例，如果为None则使用工厂函数创建默认模型
            tools: 商品工具列表，默认使用内置工具
        """
        self.llm = llm or create_llm_for_agent(temperature=0.7)
        self.tools = tools or get_product_tools()
        self.name = AgentName.PRODUCT_AGENT

        # 绑定工具到 LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def get_name(self) -> str:
        """获取 Agent 名称"""
        return self.name

    def get_description(self) -> str:
        """获取 Agent 描述"""
        return "商品搜索专家 - 处理商品查询、搜索等请求"

    async def execute(self, state: MultiAgentState, session_id: str = "default") -> Dict[str, Any]:
        """执行商品查询（异步接口，符合LangGraph 1.x规范）

        企业级最佳实践：让 LLM 自己决定使用哪些工具，而不是硬编码工具调用。
        LLM 会根据工具描述和上下文，自动判断需要调用什么工具。

        Args:
            state: 当前多 Agent 状态
            session_id: 会话ID（用于会话管理）

        Returns:
            更新后的状态片段（遵循统一的返回格式规范）
        """
        # 获取最新消息
        messages = state.messages
        if not messages:
            return {
                "result": {},  # Agent执行结果（必需）
                "messages": [
                    AIMessage(content="您好！我是商品查询助手，请问有什么可以帮您？")
                ],
                "current_agent": AgentName.PRODUCT_AGENT,
            }

        # 构建系统提示（模板组合，结构化上下文注入）
        # 企业级最佳实践：明确区分指令和上下文，确保 LLM 理解必须调用工具
        system_prompt = "\n\n".join([
            prompt_registry.render("base_tone"),
            prompt_registry.render("product_capabilities"),
            PRODUCT_AGENT_SYSTEM_PROMPT
        ]).strip()
        context_block = render_context_bundle(state.context_bundle)

        # 合并系统提示和上下文到一个 SystemMessage
        # 使用 XML 标签确保指令和上下文清晰分离（符合业界最佳实践）
        unified_system_content = "\n\n".join([
            system_prompt,
            "<context>",
            context_block,
            "</context>"
        ]).strip()

        # 构建 Agent 消息
        # 使用最新的用户消息和最近的几轮对话
        # 清理消息历史，确保消息序列完整性（过滤无效的 ToolMessage）
        cleaned_messages = clean_messages_for_llm(messages, keep_recent_n=5)

        agent_messages = [
            SystemMessage(content=unified_system_content),
        ]
        agent_messages.extend(cleaned_messages)

        # 调用 LLM（异步执行）
        response = await self.llm_with_tools.ainvoke(agent_messages)

        # 处理工具调用
        if hasattr(response, "tool_calls") and response.tool_calls:
            # 执行工具调用并构建 ToolMessage
            tool_messages = []
            tool_used_info = []
            structured_result = None  # 存储结构化数据结果（用于单个搜索场景）
            all_search_results = []  # 存储所有搜索结果（用于对比场景）

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
                                    # 如果是产品搜索结果，收集到all_search_results
                                    if "products" in result_json:
                                        all_search_results.append(result_json)
                                    # 保持最后一个结果作为structured_result
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

            # 返回结果（使用ResponseModel构建完整的前端数据）
            # 企业级规范（2025-2026终极重构）：ResponseModel包含所有前端显示字段
            if structured_result and "products" in structured_result:
                # 【关键修复】使用工具返回的原始 text（structured_result["text"]）作为 content
                # 而不是 LLM 重新生成的 final_response.content
                # 这样前端可以看到工具返回的格式化商品列表
                content_text = structured_result.get("text", final_response.content)
                response_model = ProductListResponse(
                    products=structured_result.get("products", []),
                    total=structured_result.get("total", 0),
                    query_summary=structured_result.get("query_summary", ""),
                    content=content_text  # 使用工具返回的原始 text
                )

                result = {
                    "result": structured_result,  # 必需：权威数据源
                    "messages": [response] + tool_messages + [final_response],  # 只返回新增消息
                    "current_agent": AgentName.PRODUCT_AGENT,
                    "tools_used": state.tools_used + tool_used_info,
                    **response_model.to_full_response()
                }
            else:
                # 使用TextResponse构建文本响应
                response_model = TextResponse(content=final_response.content)
                result = {
                    "result": {"response": final_response.content},
                    "messages": [response] + tool_messages + [final_response],  # 只返回新增消息
                    "current_agent": AgentName.PRODUCT_AGENT,
                    "tools_used": state.tools_used + tool_used_info,
                    **response_model.to_full_response()
                }

            return result

        # 无工具调用，直接返回响应
        response_model = TextResponse(content=response.content)
        result = {
            "result": {"response": response.content},  # 必需字段：Agent执行结果
            "messages": [response],  # 只返回新增消息
            "current_agent": AgentName.PRODUCT_AGENT,
            **response_model.to_full_response()
        }

        return result
