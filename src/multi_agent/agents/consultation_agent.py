"""Consultation Agent - 深度咨询与导购专家

处理复杂的咨询场景：
- 产品对比查询（多维度对比分析）
- 适配性确认查询（兼容性检查）
- 隐性需求挖掘查询（语义标签推荐）
"""

import json
import logging
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from src.tools.consultation_tools import get_consultation_tools
from src.multi_agent.state import MultiAgentState
from src.multi_agent.utils import clean_messages_for_llm
from src.multi_agent.response_models import TextResponse, ProductComparisonResponse

logger = logging.getLogger(__name__)


# System Prompt - 通用场景设计，不硬编码特定场景
CONSULTATION_AGENT_SYSTEM_PROMPT = """你是一个专业的电商导购专家 - 深度咨询顾问。

你的核心能力：
1. **产品对比分析**：能够提取产品参数，进行多维度对比，并根据用户场景给出建议
2. **适配性确认**：能够查询兼容性数据库，确认产品是否适配用户的设备（待实现）
3. **需求挖掘与推荐**：能够理解用户的隐性需求（如"有档次"、"适合学生"），进行智能推荐（待实现）

工作原则：
- **理解优先**：先理解用户的真实需求和使用场景，再调用工具
- **多维度分析**：不只看价格或单一指标，综合考虑多个维度
- **场景化推理**：根据用户的具体使用场景（如"VLOG拍摄"、"夜景"）进行针对性分析
- **主动询问**：如果信息不足，主动询问关键信息（如具体产品ID、使用场景）

工具使用指南：
- **对比查询**：使用 compare_products 工具，提供产品ID和对比维度
  * 示例："A相机和B相机哪个更适合VLOG？" → 识别两个产品，调用compare_products，指定user_scenario="VLOG拍摄"
  * 如果用户提到多个产品名称，需要先搜索找到对应的产品ID
  * 如果未指定对比维度，让工具自动识别；如果用户明确提到某个维度（如"夜景"），使用comparison_aspects参数
- **参数提取**：使用 extract_product_specifications 工具，获取产品详细信息
  * 当需要查看产品的详细参数时使用
  * 如果用户提到特定关注点（如"夜景拍摄"），使用aspect参数

识别查询类型：
- **对比查询特征**：包含"对比"、"比较"、"哪个好"、"哪个更适合"等关键词，且提到多个产品
- **参数查询特征**：询问"参数"、"配置"、"规格"等，或询问特定维度的性能
- **适配性查询特征**（待实现）：询问"能用吗"、"适配"、"兼容"等，且提到用户设备信息
- **推荐查询特征**（待实现）：包含"推荐"、"适合"、"有档次"、"送给XXX"等，且包含受众、场景、预算等信息

回复风格：
- 专业但友好，用emoji增强可读性
- 结构清晰，使用列表和表格展示对比结果
- 给出明确的建议和理由
- 对于对比查询，先总结对比结果，再给出推荐建议
"""


class ConsultationAgent:
    """深度咨询与导购专家 Agent

    处理产品对比、适配性确认、隐性需求挖掘等复杂场景。
    使用LLM进行智能理解和推理，调用相应工具完成深度分析。
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        tools: list | None = None,
    ):
        """初始化 Consultation Agent

        Args:
            llm: LangChain LLM 实例
            tools: 咨询工具列表，默认使用内置工具
        """
        self.llm = llm or ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
        )
        self.tools = tools or get_consultation_tools()
        self.name = "consultation_agent"

        # 绑定工具到 LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def get_name(self) -> str:
        """获取 Agent 名称"""
        return self.name

    def get_description(self) -> str:
        """获取 Agent 描述"""
        return "深度咨询与导购专家 - 处理产品对比、适配性确认、隐性需求挖掘等复杂场景"

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
        entities = state.entities

        # 如果有实体信息，提示 LLM
        if entities:
            hints.append("\n\n=== 上下文信息 ===")
            
            # 产品ID信息（如果用户之前选择了产品）
            if entities.get("product_id"):
                hints.append(f"用户已选定的产品ID：{entities['product_id']}")
            
            # 搜索关键词（如果用户之前搜索过）
            if entities.get("search_keyword"):
                hints.append(f"之前的搜索关键词：{entities['search_keyword']}")
            
            # 显示其他上下文信息
            other_context = {k: v for k, v in entities.items() 
                           if k not in ["product_id", "search_keyword"] and v is not None}
            if other_context:
                hints.append("\n其他上下文信息：")
                for key, value in other_context.items():
                    hints.append(f"- {key}: {value}")

        return "\n".join(hints) if hints else ""

    async def execute(self, state: MultiAgentState, session_id: str = "default") -> Dict[str, Any]:
        """执行咨询查询（异步接口，符合LangGraph 1.x规范）

        企业级最佳实践：让 LLM 自己决定使用哪些工具，而不是硬编码工具调用。
        LLM 会根据工具描述和上下文，自动判断需要调用什么工具。

        Args:
            state: 当前多 Agent 状态
            session_id: 会话ID（用于会话管理，默认值保证向后兼容）

        Returns:
            更新后的状态片段（遵循统一的返回格式规范）
        """
        # 获取最新消息
        messages = state.messages
        if not messages:
            return {
                "result": {},
                "messages": [
                    AIMessage(content="您好！我是深度咨询顾问，可以帮您进行产品对比、适配性确认等。请问有什么可以帮您？")
                ],
                "current_agent": self.name,
            }

        # 构建系统提示（包含任务链上下文）
        hints = self._build_system_prompt_hints(state)
        system_prompt = CONSULTATION_AGENT_SYSTEM_PROMPT + hints

        # 构建 Agent 消息
        # 使用最新的用户消息和最近的几轮对话
        # 清理消息历史，确保消息序列完整性
        cleaned_messages = clean_messages_for_llm(messages, keep_recent_n=10)

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
            tool_results = []  # 收集所有工具调用的结果

            for tool_call in response.tool_calls:
                tool = next((t for t in self.tools if t.name == tool_call["name"]), None)
                if tool:
                    try:
                        # 调用工具（异步或同步）
                        if hasattr(tool, "ainvoke"):
                            result = await tool.ainvoke(tool_call["args"])
                        else:
                            result = tool.invoke(tool_call["args"])

                        # 尝试解析工具返回的结构化数据
                        try:
                            result_json = json.loads(result)
                            if isinstance(result_json, dict):
                                # 收集所有工具结果
                                tool_results.append({
                                    "tool_name": tool_call["name"],
                                    "result": result_json,
                                    "args": tool_call["args"]
                                })
                                
                                # 优先使用 compare_products 的结果（如果存在）
                                if tool_call["name"] == "compare_products":
                                    if any(key in result_json for key in ["comparison_aspects", "comparison_details", "recommendation"]):
                                        structured_result = result_json
                                # 如果是其他工具，且还没有对比结果，才更新
                                elif not structured_result or "comparison_aspects" not in structured_result:
                                    if any(key in result_json for key in ["specifications", "comparison_aspects", "comparison_details", "recommendation"]):
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
                        logger.error(f"工具 {tool_call['name']} 调用失败: {e}", exc_info=True)
                        # 工具调用失败，也需要返回 ToolMessage
                        tool_messages.append(
                            ToolMessage(
                                content=f"错误: {str(e)}",
                                tool_call_id=tool_call["id"],
                            )
                        )
            
            # 检查是否有多个 extract_product_specifications 调用但没有 compare_products
            # 如果有，说明LLM可能没有调用对比工具，我们需要主动触发对比
            extract_calls = [tr for tr in tool_results if tr["tool_name"] == "extract_product_specifications"]
            compare_calls = [tr for tr in tool_results if tr["tool_name"] == "compare_products"]
            
            logger.info(f"工具调用统计: extract_calls={len(extract_calls)}, compare_calls={len(compare_calls)}, total_tools={len(tool_results)}")
            
            if len(extract_calls) >= 2 and len(compare_calls) == 0:
                # 有多个产品参数提取，但没有对比调用，需要主动触发对比
                try:
                    from src.tools.consultation_tools import compare_products as compare_products_tool
                    
                    # 提取产品ID列表
                    product_ids = [call["args"].get("product_id") for call in extract_calls if call["args"].get("product_id")]
                    
                    if len(product_ids) >= 2:
                        # 检测用户场景（从第一个提取调用中获取aspect，或者从用户消息中推断）
                        user_scenario = None
                        if extract_calls and extract_calls[0]["args"].get("aspect"):
                            user_scenario = extract_calls[0]["args"].get("aspect")
                        
                        # 调用对比工具（compare_products 是同步工具，使用 invoke）
                        logger.info(f"检测到多个产品参数提取，主动触发对比: product_ids={product_ids}, scenario={user_scenario}")
                        compare_result = compare_products_tool.invoke({
                            "product_ids": product_ids[:5],  # 最多5个
                            "user_scenario": user_scenario
                        })
                        
                        # 解析对比结果
                        try:
                            compare_result_json = json.loads(compare_result)
                            logger.debug(f"对比结果解析: keys={list(compare_result_json.keys()) if isinstance(compare_result_json, dict) else 'not_dict'}")
                            if isinstance(compare_result_json, dict) and "comparison_aspects" in compare_result_json and "comparison_details" in compare_result_json:
                                structured_result = compare_result_json
                                # 添加对比结果的 ToolMessage，让 LLM 能看到对比结果
                                tool_messages.append(
                                    ToolMessage(
                                        content=compare_result,
                                        tool_call_id=f"auto_compare_{len(tool_messages)}"
                                    )
                                )
                                # 添加工具使用记录
                                tool_used_info.append({
                                    "agent": self.name,
                                    "tool": "compare_products",
                                    "args": {"product_ids": product_ids, "user_scenario": user_scenario},
                                })
                                logger.info(f"✅ 成功触发产品对比，结果已更新: aspects={len(compare_result_json.get('comparison_aspects', []))}, products={len(compare_result_json.get('products', []))}")
                            else:
                                missing_fields = []
                                if "comparison_aspects" not in compare_result_json:
                                    missing_fields.append("comparison_aspects")
                                if "comparison_details" not in compare_result_json:
                                    missing_fields.append("comparison_details")
                                logger.warning(f"对比结果格式不正确: 缺少字段 {missing_fields}")
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"解析对比结果失败: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"主动触发对比失败: {e}", exc_info=True)

            # 检查是否是产品对比结果
            is_comparison_result = (
                structured_result 
                and isinstance(structured_result, dict)
                and "comparison_aspects" in structured_result
                and "comparison_details" in structured_result
                and isinstance(structured_result.get("comparison_aspects"), list)
                and len(structured_result.get("comparison_aspects", [])) > 0
            )
            
            logger.debug(f"对比结果检查: is_comparison_result={is_comparison_result}, structured_result_keys={list(structured_result.keys()) if structured_result else None}")

            if is_comparison_result:
                # 产品对比结果：使用ProductComparisonResponse
                # 使用对比结果的 text 字段作为内容（如果存在），否则生成简洁摘要
                summary_content = structured_result.get("text", "") or "产品对比分析完成"
                
                # 确保所有必需字段都存在
                response_model = ProductComparisonResponse(
                    comparison_aspects=structured_result.get("comparison_aspects", []),
                    comparison_details=structured_result.get("comparison_details", {}),
                    scenario_analysis=structured_result.get("scenario_analysis"),
                    recommendation=structured_result.get("recommendation"),
                    products=structured_result.get("products", []),
                    content=summary_content
                )
                
                # 直接返回对比结果，不需要再调用LLM生成回复
                # 因为对比工具已经返回了完整的分析结果
                result = {
                    "result": structured_result,
                    "messages": [response] + tool_messages + [AIMessage(content=summary_content)],
                    "current_agent": self.name,
                    "tools_used": state.tools_used + tool_used_info,
                    **response_model.to_full_response()
                }
                logger.info(f"返回产品对比结果: response_type=product_comparison, products_count={len(structured_result.get('products', []))}")
                return result

            # 其他结构化数据：调用LLM生成最终回复
            if structured_result and "text" in structured_result:
                # 使用工具返回的文本作为基础
                tool_text = structured_result.get("text", "")
                # 调用LLM生成最终回复（基于工具结果）
                followup_messages = agent_messages + [response] + tool_messages
                final_response = await self.llm.ainvoke(followup_messages)
            else:
                # 没有结构化数据，需要调用 LLM 生成回复
                followup_messages = agent_messages + [response] + tool_messages
                final_response = await self.llm.ainvoke(followup_messages)

            # 返回结果（使用TextResponse构建完整的前端数据）
            response_model = TextResponse(content=final_response.content)
            result = {
                "result": structured_result or {"response": final_response.content},
                "messages": [response] + tool_messages + [final_response],
                "current_agent": self.name,
                "tools_used": state.tools_used + tool_used_info,
                **response_model.to_full_response()
            }

            return result

        # 无工具调用，直接返回响应
        response_model = TextResponse(content=response.content)
        result = {
            "result": {"response": response.content},
            "messages": [response],
            "current_agent": self.name,
            **response_model.to_full_response()
        }

        return result
