"""Browser Agent - Agentic Web 浏览器自动化 Agent

处理需要真实浏览器交互的任务：
- 真实网站商品搜索（支持 JavaScript 渲染）
- 跨平台价格比较
- 商品详情提取
- 动态内容抓取

企业级最佳实践：
- 集成 browser-use 实现真实浏览器自动化
- 使用 LLM 决策工具调用（不硬编码工具链）
- 返回结构化 JSON 数据
- 完善的错误处理和降级策略
"""

import json
import logging
from typing import Any, Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from src.utils.llm_factory import create_llm_for_agent

from src.tools.browser_tools import get_browser_tools, is_browser_available
from src.multi_agent.agents.base_agent import BaseAgent
from src.multi_agent.state import MultiAgentState
from src.multi_agent.utils import clean_messages_for_llm
from src.multi_agent.response_models import ProductListResponse, TextResponse
from src.multi_agent.prompts import prompt_registry, render_context_bundle
from src.multi_agent.constants import AgentName

logger = logging.getLogger(__name__)


BROWSER_AGENT_SYSTEM_PROMPT = """你是一个专业的 Agentic Web Agent - 网络搜索专家。

你的核心能力：
1. 电商商品搜索 - 通过 Google 搜索聚合京东、淘宝、天猫、闲鱼等平台的商品信息
2. 跨平台比价 - 一次搜索同时获取多个电商平台的价格对比
3. 商品详情提取 - 从商品详情页提取完整的结构化信息

搜索策略（重要）：
- 使用 Google 搜索 + site: 操作符来查找电商平台商品
- 优势：不需要登录、避免验证码、结果稳定可靠
- 支持同时搜索多个平台（京东/淘宝/天猫/闲鱼）

使用场景：
- 用户要求在淘宝/京东/天猫/闲鱼等平台搜索商品
- 用户需要跨平台比价
- 用户需要查找外部网站的商品信息

工具使用规则：
- browser_search_product: 搜索商品（site="all" 搜索所有平台，或指定具体平台）
- browser_compare_prices: 跨平台比价（一次 Google 搜索聚合多平台结果）
- browser_extract_product_info: 提取商品详情页的完整信息

注意事项：
1. 搜索结果来自 Google 索引，可能不是最新实时数据
2. 价格信息可能需要点击进入商品页面才能确认
3. 如果浏览器工具不可用，告知用户并建议使用数据库搜索
"""


class BrowserAgent(BaseAgent):
    """浏览器自动化 Agent

    使用 browser-use 在真实浏览器中执行商品搜索、比价、信息提取等任务。
    支持处理 JavaScript 渲染的动态内容，获取实时数据。
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        tools: list | None = None,
    ):
        """初始化 Browser Agent

        Args:
            llm: LangChain LLM 实例
            tools: 浏览器工具列表（默认使用内置工具）
        """
        super().__init__(
            name=AgentName.BROWSER_AGENT,
            llm=llm or create_llm_for_agent(temperature=0.3),
            description="浏览器自动化专家 - 在真实网站搜索商品、比价、提取详情"
        )

        # 检查浏览器工具是否可用
        logger.info(f"[BrowserAgent.__init__] 开始初始化")
        browser_available = is_browser_available()
        logger.info(f"[BrowserAgent.__init__] is_browser_available() = {browser_available}")
        
        if not browser_available:
            logger.error("❌ browser-use 不可用，BrowserAgent 将无法正常工作")
            self.tools = []
            self.llm_with_tools = self.llm
        else:
            logger.info(f"[BrowserAgent.__init__] 准备获取工具")
            fetched_tools = get_browser_tools()
            logger.info(f"[BrowserAgent.__init__] 获取到 {len(fetched_tools)} 个工具")
            
            self.tools = tools or fetched_tools
            logger.info(f"[BrowserAgent.__init__] 最终 tools 数量: {len(self.tools)}")
            
            if self.tools:
                self.llm_with_tools = self.llm.bind_tools(self.tools)
                logger.info(f"✅ BrowserAgent 已初始化，可用工具: {[t.name for t in self.tools]}")
            else:
                logger.error("❌ 工具列表为空！")
                self.llm_with_tools = self.llm

    async def execute(self, state: MultiAgentState, session_id: str = "default") -> Dict[str, Any]:
        """执行浏览器自动化任务（异步）

        让 LLM 根据用户需求自主决定使用哪些工具，实现真正的 Agentic 行为。

        Args:
            state: 当前多 Agent 状态
            session_id: 会话ID

        Returns:
            更新后的状态片段（遵循统一的返回格式规范）
        """
        # 检查工具可用性
        if not self.tools:
            return {
                "result": {"error": "browser-use 不可用"},
                "messages": [
                    AIMessage(
                        content="抱歉，浏览器自动化功能当前不可用（browser-use 未安装）。"
                        "建议使用数据库搜索功能查询商品。"
                    )
                ],
                "current_agent": self.get_name(),
                "response_type": "error"
            }

        # 获取最新消息
        messages = state.messages
        if not messages:
            return {
                "result": {},
                "messages": [
                    AIMessage(content="您好！我是浏览器自动化助手，可以帮您在真实网站搜索商品和比价。")
                ],
                "current_agent": self.get_name(),
                "response_type": "text"
            }

        # 构建系统提示（模板组合，结构化上下文注入）
        system_prompt = "\n\n".join([
            prompt_registry.render("base_tone"),
            BROWSER_AGENT_SYSTEM_PROMPT
        ]).strip()
        context_block = render_context_bundle(state.context_bundle)

        # 合并系统提示和上下文
        unified_system_content = "\n\n".join([
            system_prompt,
            "<context>",
            context_block,
            "</context>"
        ]).strip()

        # 清理消息历史
        cleaned_messages = clean_messages_for_llm(messages, keep_recent_n=5)

        agent_messages = [
            SystemMessage(content=unified_system_content),
        ]
        agent_messages.extend(cleaned_messages)

        # 调用 LLM（让 LLM 决定工具使用）
        response = await self.llm_with_tools.ainvoke(agent_messages)

        # 处理工具调用
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_messages = []
            tool_used_info = []
            structured_result = None
            all_products = []

            for tool_call in response.tool_calls:
                tool = next((t for t in self.tools if t.name == tool_call["name"]), None)
                if tool:
                    try:
                        logger.info(f"执行浏览器工具: {tool_call['name']}, 参数: {tool_call['args']}")
                        result = await tool.ainvoke(tool_call["args"])

                        # 解析工具返回的结构化数据
                        try:
                            result_json = json.loads(result)
                            if isinstance(result_json, dict):
                                # 检查是否包含商品数据
                                if "products" in result_json:
                                    products = result_json.get("products", [])
                                    all_products.extend(products)
                                    structured_result = result_json
                                elif "comparison" in result_json:
                                    # 比价结果
                                    comparison = result_json.get("comparison", {})
                                    all_products.extend(comparison.get("all_products", []))
                                    structured_result = result_json
                                elif "product" in result_json:
                                    # 单个商品详情
                                    structured_result = result_json
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(f"工具返回结果不是 JSON: {result}")

                        # 构建 ToolMessage
                        tool_messages.append(
                            ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call["id"],
                            )
                        )
                        tool_used_info.append({
                            "agent": self.get_name(),
                            "tool": tool_call["name"],
                            "args": tool_call["args"],
                        })

                    except Exception as e:
                        logger.error(f"工具调用失败: {e}", exc_info=True)
                        tool_messages.append(
                            ToolMessage(
                                content=f"错误: {str(e)}",
                                tool_call_id=tool_call["id"],
                            )
                        )

            # 使用工具返回的文本（如果有），避免 LLM 重新生成
            if structured_result and "text" in structured_result:
                final_response = AIMessage(content=structured_result["text"])
            else:
                # 让 LLM 根据工具结果生成回复
                followup_messages = agent_messages + [response] + tool_messages
                final_response = await self.llm.ainvoke(followup_messages)

            # 构建返回结果（使用 ResponseModel）
            if all_products:
                # 有商品数据，使用 ProductListResponse
                content_text = structured_result.get("text", final_response.content)
                response_model = ProductListResponse(
                    products=all_products,
                    total=len(all_products),
                    query_summary=structured_result.get("query_summary", ""),
                    content=content_text
                )

                return {
                    "result": structured_result,
                    "messages": [response] + tool_messages + [final_response],
                    "current_agent": self.get_name(),
                    "tools_used": state.tools_used + tool_used_info,
                    **response_model.to_full_response()
                }
            else:
                # 无商品数据，使用 TextResponse
                response_model = TextResponse(content=final_response.content)
                return {
                    "result": structured_result or {"response": final_response.content},
                    "messages": [response] + tool_messages + [final_response],
                    "current_agent": self.get_name(),
                    "tools_used": state.tools_used + tool_used_info,
                    **response_model.to_full_response()
                }

        # 无工具调用，直接返回响应
        response_model = TextResponse(content=response.content)
        return {
            "result": {"response": response.content},
            "messages": [response],
            "current_agent": self.get_name(),
            **response_model.to_full_response()
        }
