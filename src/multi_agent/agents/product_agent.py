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
        return "商品搜索专家 - 处理商品查询、搜索、比价等请求"

    def _is_comparison_scenario(self, state: MultiAgentState) -> bool:
        """检测是否为产品对比场景

        企业级最佳实践：直接使用意图识别阶段（LLM已判断）的结果，
        而不是硬编码关键词，这样可以更好地适配自然语言的多样性。

        Args:
            state: 当前多Agent状态

        Returns:
            是否为对比场景
        """
        # 直接使用意图识别结果（意图识别阶段已通过LLM判断）
        query_intent = state.query_intent
        if query_intent:
            intent_type = query_intent.get("intent_type")
            if intent_type == "comparison":
                return True

        return False

    def _extract_product_ids_from_search_results(self, structured_result: Dict[str, Any]) -> list[int]:
        """从搜索结果中提取产品ID列表

        Args:
            structured_result: 搜索结果的结构化数据

        Returns:
            产品ID列表
        """
        product_ids = []
        if structured_result and "products" in structured_result:
            products = structured_result["products"]
            for product in products:
                product_id = product.get("id") or product.get("product_id")
                if product_id and isinstance(product_id, int):
                    product_ids.append(product_id)
        return product_ids

    def _merge_search_results_and_extract_ids(
        self, 
        all_search_results: list[Dict[str, Any]]
    ) -> tuple[Dict[str, Any], list[int]]:
        """合并多个搜索结果并提取产品ID（企业级优化：一次循环完成两个任务）

        企业级最佳实践：
        - 在单次循环中同时完成合并和提取，避免重复遍历
        - 使用 dict.fromkeys() 进行去重并保持顺序（Python 3.7+）

        Args:
            all_search_results: 所有搜索结果列表

        Returns:
            (merged_result, product_ids): 合并后的结果和提取的产品ID列表
        """
        merged_products = []
        merged_text_parts = []
        all_product_ids = []
        
        for idx, search_result in enumerate(all_search_results, 1):
            # 合并产品列表
            products = search_result.get("products", [])
            merged_products.extend(products)
            
            # 合并文本内容
            if search_result.get("text"):
                merged_text_parts.append(f"搜索结果{idx}：\n{search_result['text']}")
            
            # 同时提取产品ID（避免后续再次循环）
            product_ids = self._extract_product_ids_from_search_results(search_result)
            all_product_ids.extend(product_ids)
        
        # 去重并保持顺序（Python 3.7+ dict.fromkeys 保持插入顺序）
        unique_product_ids = list(dict.fromkeys(all_product_ids))
        
        # 构建合并后的结果
        structured_result = {
            "products": merged_products,
            "total": len(merged_products),
            "text": "\n\n".join(merged_text_parts) if merged_text_parts else f"找到{len(merged_products)}个产品",
            "query_summary": f"对比场景：搜索了{len(all_search_results)}个产品"
        }
        
        logger.info(f"合并{len(all_search_results)}个搜索结果，共{len(merged_products)}个产品，提取{len(unique_product_ids)}个唯一产品ID")
        
        return structured_result, unique_product_ids

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
        # 使用清晰的分隔符确保指令部分突出（LLM 需要明确理解必须调用工具）
        unified_system_content = "\n\n".join([
            system_prompt,
            "---",
            context_block
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

            # 合并多个搜索结果并提取产品ID（对比场景）
            is_comparison = self._is_comparison_scenario(state)
            extracted_product_ids = None  # 存储提取的产品ID（用于后续entities更新）
            
            if is_comparison and all_search_results and len(all_search_results) > 1:
                # 使用优化方法：一次循环完成合并和提取
                structured_result, extracted_product_ids = self._merge_search_results_and_extract_ids(all_search_results)

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
                
                # 更新entities（使用已提取的产品ID，避免重复循环）
                entities_update = {}
                
                if is_comparison:
                    # 对比场景：使用已提取的产品ID
                    if extracted_product_ids:
                        # 多个搜索结果的情况，已在合并时提取
                        entities_update["product_ids"] = extracted_product_ids
                        logger.info(f"检测到对比场景，使用已提取的产品ID: {extracted_product_ids}")
                    elif all_search_results and len(all_search_results) == 1:
                        # 对比场景但只有一个搜索结果，需要单独提取
                        product_ids = self._extract_product_ids_from_search_results(structured_result)
                        if product_ids:
                            entities_update["product_ids"] = product_ids
                            logger.info(f"检测到对比场景（单结果），提取产品ID: {product_ids}")
                
                result = {
                    "result": structured_result,  # 必需：权威数据源
                    "messages": [response] + tool_messages + [final_response],  # 只返回新增消息
                    "current_agent": AgentName.PRODUCT_AGENT,
                    "tools_used": state.tools_used + tool_used_info,
                    "conversation_phase": "product_selecting",  # 设置对话阶段
                    "entities": entities_update,  # 更新entities
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
