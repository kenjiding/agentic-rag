"""任务编排引擎

TaskChainOrchestrator 负责：
1. 检测是否为多步骤任务
2. 创建任务链
3. 执行当前步骤
4. 管理任务链生命周期

2025-2026 最佳实践：
- 使用 LLM-based 意图识别替代硬编码规则
- 结合规则和 LLM：规则作为快速路径，LLM 作为精确判断
- 利用已有的意图分类器基础设施
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import logging
import re

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.multi_agent.state import MultiAgentState, TaskChain, TaskStep
from src.confirmation.selection_manager import get_selection_manager
from src.multi_agent.task_chain_storage import get_task_chain_storage

logger = logging.getLogger(__name__)


class MultiStepTaskDetection(BaseModel):
    """多步骤任务检测结果
    
    使用结构化输出确保 LLM 返回格式正确。
    """
    is_multi_step: bool = Field(
        description="是否为多步骤任务"
    )
    task_type: Optional[str] = Field(
        default=None,
        description="任务类型，如 'order_with_search'（搜索商品后下单），如果不是多步骤任务则为 None"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="检测置信度，0.0-1.0"
    )
    reasoning: str = Field(
        description="检测推理过程，解释为什么判断为多步骤任务或不是"
    )
    has_product_id: bool = Field(
        default=False,
        description="用户消息中是否包含具体的 product_id（如果有，通常不需要搜索步骤）"
    )


class TaskChainOrchestrator:
    """任务链编排引擎

    负责检测和编排多步骤任务，如"搜索商品 → 用户选择 → 创建订单"。
    
    改进点：
    1. 使用 LLM-based 检测替代硬编码规则
    2. 保留规则作为快速路径（可选）
    3. 更好的语义理解和上下文感知
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None, use_fast_path: bool = True):
        """初始化任务编排器
        
        Args:
            llm: 语言模型实例，用于 LLM-based 检测（如果为 None，则创建默认实例）
            use_fast_path: 是否使用规则作为快速路径（默认 True，提高性能）
        """
        self.selection_manager = get_selection_manager()
        self.use_fast_path = use_fast_path
        
        # 初始化 LLM（用于精确检测）
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.structured_llm = self.llm.with_structured_output(MultiStepTaskDetection)
        
        logger.info(f"TaskChainOrchestrator 初始化完成 (use_fast_path={use_fast_path})")

    def detect_multi_step_task(self, state: MultiAgentState) -> Optional[str]:
        """检测是否为多步骤任务
        
        改进后的实现：
        1. 快速路径：使用规则快速过滤明显不是多步骤任务的情况
        2. 精确路径：使用 LLM 进行语义理解和上下文感知
        
        Args:
            state: 当前状态

        Returns:
            任务类型（如 "order_with_search"），如果不是多步骤任务则返回 None
        """
        # 提取最后一条用户消息
        user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

        if not user_message:
            return None

        # === 快速路径：规则检测（可选，用于提高性能）===
        if self.use_fast_path:
            fast_result = self._fast_path_detection(user_message)
            if fast_result is not None:
                return fast_result

        # === 精确路径：LLM-based 检测 ===
        llm_result = self._llm_based_detection(user_message, state)
        if llm_result:
            return llm_result
        
        # LLM 检测失败或返回 None，使用降级规则
        return self._fallback_rule_detection(user_message)

    def _fast_path_detection(self, user_message: str) -> Optional[str]:
        """快速路径：基于规则的快速检测
        
        用于快速排除明显不是多步骤任务的情况。
        如果检测到明确的多步骤任务信号，直接返回结果。
        """
        user_message_lower = user_message.lower()
        
        # 快速排除：已经有明确的 product_id
        if re.search(r"product[_\s]*id[:\s]*\d+|商品[_\s]*id[:\s]*\d+|商品编号[:\s]*\d+", user_message_lower):
            return None
        
        # 快速排除：完全没有购买意图
        order_keywords = ["下单", "购买", "买", "订购", "我要", "帮我"]
        if not any(kw in user_message_lower for kw in order_keywords):
            return None
        
        # 检测到购买意图，检查是否有商品关键词
        product_keywords = [
            "商品", "产品", "手机", "电脑", "笔记本", "家电", "电视", "冰箱",
            "华为", "苹果", "小米", "西门子", "海尔", "格力", "联想", "戴尔"
        ]
        if any(kw in user_message_lower for kw in product_keywords):
            # 明确的多步骤任务信号，直接返回
            logger.info("快速路径：检测到明确的多步骤任务信号")
            return "order_with_search"
        
        return None

    def _llm_based_detection(self, user_message: str, state: MultiAgentState) -> Optional[str]:
        """LLM-based 精确检测
        
        使用 LLM 进行语义理解和上下文感知，更准确地判断是否为多步骤任务。
        
        Args:
            user_message: 用户消息
            state: 当前状态（可用于上下文信息）
            
        Returns:
            任务类型，如果不是多步骤任务则返回 None
        """
        # 构建检测 prompt
        template = """你是一个专业的任务分析专家。请分析用户消息，判断是否为多步骤任务。

# 多步骤任务定义

多步骤任务是指需要多个步骤才能完成的任务，例如：
- "搜索商品 → 用户选择 → 创建订单"（order_with_search）

# 当前场景：电商订单

在这个场景中，多步骤任务主要指：
- **order_with_search**: 用户想要购买商品，但还没有选择具体的商品
  - 特征：包含购买意图（下单/购买/买）+ 商品描述/关键词/品牌名 + 没有具体的 product_id
  - 示例：
    * "我要下单，购买西门子商品 2 件" ✅ 是多步骤任务（需要先搜索西门子商品）
    * "我想买一部华为手机" ✅ 是多步骤任务
    * "帮我下单购买苹果笔记本" ✅ 是多步骤任务
  - 不是多步骤任务的情况：
    * 已经有明确的 product_id（如"购买 product_id: 123"）
    * 只是查询商品信息，没有购买意图（如"华为手机有哪些"）
    * 只是查询订单，没有创建意图（如"我的订单"）

# 关键判断标准

1. **必须同时满足**：
   - 有购买/下单意图（下单、购买、买、订购等）
   - 有商品描述/关键词/品牌名（但**没有**具体的 product_id）
   
2. **如果用户说"我要下单，购买XX商品"**，这**一定是**多步骤任务，因为需要先搜索商品

# 用户消息

{user_message}

# 输出要求

请分析用户消息，判断是否为多步骤任务（order_with_search）。
如果是，返回 task_type="order_with_search" 和较高的置信度（>=0.8）。
如果不是，返回 is_multi_step=false 和相应的推理过程。"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.structured_llm

        try:
            result = chain.invoke({"user_message": user_message})
            
            if isinstance(result, MultiStepTaskDetection):
                detection = result
            elif isinstance(result, dict):
                detection = MultiStepTaskDetection(**result)
            else:
                detection = MultiStepTaskDetection.model_validate(result)
            
            if detection.is_multi_step and detection.task_type:
                logger.info(
                    f"LLM检测到多步骤任务: {detection.task_type}, "
                    f"置信度: {detection.confidence:.2f}, "
                    f"推理: {detection.reasoning[:100]}..."
                )
                return detection.task_type
            else:
                logger.debug(
                    f"LLM判断不是多步骤任务: {detection.reasoning[:100]}..."
                )
                return None
                
        except Exception as e:
            logger.error(f"LLM-based 检测失败: {e}", exc_info=True)
            # 降级到规则检测
            logger.warning("降级到规则检测")
            return self._fallback_rule_detection(user_message)

    def _fallback_rule_detection(self, user_message: str) -> Optional[str]:
        """降级方案：基于规则的检测"""
        user_message_lower = user_message.lower()
        
        # 检测购买意图
        has_order_intent = any(kw in user_message_lower for kw in [
            "下单", "购买", "买", "订购", "我要", "帮我"
        ])
        
        # 检测商品关键词（包括品牌名）
        has_product_keyword = any(kw in user_message_lower for kw in [
            "商品", "产品", "手机", "电脑", "笔记本", "家电", "电视", "冰箱",
            "华为", "苹果", "小米", "西门子", "海尔", "格力", "联想", "戴尔"
        ])
        
        # 检测是否已经有明确的 product_id
        has_product_id = bool(re.search(
            r"product[_\s]*id[:\s]*\d+|商品[_\s]*id[:\s]*\d+|商品编号[:\s]*\d+",
            user_message_lower
        ))
        
        if has_order_intent and has_product_keyword and not has_product_id:
            logger.info(f"降级规则检测到多步骤任务: order_with_search")
            return "order_with_search"
        
        return None

    def create_task_chain(
        self,
        task_type: str,
        initial_state: MultiAgentState
    ) -> TaskChain:
        """根据任务类型创建任务链

        Args:
            task_type: 任务类型
            initial_state: 初始状态

        Returns:
            TaskChain 对象
        """
        if task_type == "order_with_search":
            return self._create_order_with_search_chain(initial_state)
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")

    def _create_order_with_search_chain(
        self,
        state: MultiAgentState
    ) -> TaskChain:
        """创建"搜索商品 → 选择 → 下单"任务链

        Args:
            state: 当前状态

        Returns:
            TaskChain 对象
        """
        # 从 state["entities"] 读取实体信息（2025最佳实践：统一实体管理）
        entities = state.get("entities", {})
        context_data = {
            "user_phone": entities.get("user_phone"),
            "quantity": entities.get("quantity", 1),
            "search_keyword": entities.get("search_keyword"),
        }

        logger.info(f"从 entities 读取任务链上下文: {context_data}")

        task_chain: TaskChain = {
            "chain_id": str(uuid.uuid4()),
            "chain_type": "order_with_search",
            "steps": [
                {
                    "step_id": "search-1",
                    "step_type": "product_search",
                    "status": "pending",
                    "agent_name": "product_agent",
                    "result_data": None,
                    "metadata": None
                },
                {
                    "step_id": "select-1",
                    "step_type": "user_selection",
                    "status": "pending",
                    "agent_name": None,
                    "result_data": None,
                    "metadata": None
                },
                {
                    "step_id": "order-1",
                    "step_type": "order_creation",
                    "status": "pending",
                    "agent_name": "order_agent",
                    "result_data": None,
                    "metadata": None
                }
            ],
            "current_step_index": 0,
            "context_data": context_data,
            "created_at": datetime.utcnow().isoformat()
        }

        logger.info(
            f"创建任务链: id={task_chain['chain_id']}, "
            f"type={task_chain['chain_type']}, "
            f"steps={len(task_chain['steps'])}"
        )

        return task_chain

    async def execute_current_step(
        self,
        state: MultiAgentState,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """执行任务链的当前步骤

        Args:
            state: 当前状态
            session_id: 会话ID

        Returns:
            要更新到状态中的数据
        """
        task_chain = state.get("task_chain")
        if not task_chain:
            logger.error("task_chain 不存在")
            return {"next_action": "finish"}

        current_index = task_chain["current_step_index"]
        steps = task_chain["steps"]

        if current_index >= len(steps):
            logger.info("任务链已完成")
            return {
                "next_action": "finish",
                "task_chain": None  # 清除任务链
            }

        current_step: TaskStep = steps[current_index]
        step_type = current_step["step_type"]

        logger.info(
            f"执行步骤: index={current_index}, "
            f"type={step_type}, "
            f"agent={current_step.get('agent_name')}"
        )

        if step_type == "product_search":
            return self._execute_product_search(state, task_chain, current_step)
        elif step_type == "user_selection":
            return await self._execute_user_selection(state, task_chain, current_step, session_id)
        elif step_type == "order_creation":
            return self._execute_order_creation(state, task_chain, current_step)
        else:
            logger.error(f"不支持的步骤类型: {step_type}")
            return {"next_action": "finish"}

    def _execute_product_search(
        self,
        state: MultiAgentState,
        task_chain: TaskChain,
        current_step: TaskStep
    ) -> Dict[str, Any]:
        """执行产品搜索步骤

        Args:
            state: 当前状态
            task_chain: 任务链
            current_step: 当前步骤

        Returns:
            路由到 product_agent
        """
        # 标记当前步骤为 in_progress
        current_step["status"] = "in_progress"

        # 路由到 product_agent，并传递context_data
        return {
            "next_action": "product_search",
            "selected_agent": "product_agent",
            "task_chain": task_chain,
            "context_data": task_chain.get("context_data", {})
        }

    async def _execute_user_selection(
        self,
        state: MultiAgentState,
        task_chain: TaskChain,
        current_step: TaskStep,
        session_id: str
    ) -> Dict[str, Any]:
        """执行用户选择步骤"""
        # 从任务链的上一步骤获取产品列表
        products = None
        current_index = task_chain["current_step_index"]
        if current_index > 0:
            prev_step = task_chain["steps"][current_index - 1]
            if prev_step.get("step_type") == "product_search":
                products = prev_step.get("result_data", {}).get("products")
        
        # 降级：从 agent_results 获取
        if not products:
            product_result = state.get("agent_results", {}).get("product_agent", {})
            products = product_result.get("products") if isinstance(product_result, dict) else None

        if not products:
            logger.error("未找到产品列表，无法创建选择")
            return {
                "next_action": "finish",
                "task_chain": None,
                "messages": state.get("messages", []) + [{
                    "role": "assistant",
                    "content": "抱歉，未找到相关商品，请更换关键词重试。"
                }]
            }

        # 创建选择请求
        search_keyword = task_chain["context_data"].get("search_keyword", "商品")
        selection = await self.selection_manager.request_selection(
            session_id=session_id,
            selection_type="product",
            options=products,
            display_message=f"请选择要购买的{search_keyword}:",
            metadata={"task_chain_id": task_chain["chain_id"]}
        )

        # 标记当前步骤为 in_progress
        current_step["status"] = "in_progress"

        # 返回待选择状态，graph 将暂停
        return {
            "pending_selection": {
                "selection_id": selection.selection_id,
                "selection_type": selection.selection_type,
                "options": selection.options,
                "display_message": selection.display_message,
                "metadata": selection.metadata
            },
            "next_action": "wait_for_selection",
            "task_chain": task_chain
        }

    def _execute_order_creation(
        self,
        state: MultiAgentState,
        task_chain: TaskChain,
        current_step: TaskStep
    ) -> Dict[str, Any]:
        """执行订单创建步骤"""
        current_step["status"] = "in_progress"
        context_data = task_chain.get("context_data", {})
        
        return {
            "next_action": "order_management",
            "selected_agent": "order_agent",
            "task_chain": task_chain,
            "context_data": context_data
        }

    def move_to_next_step(self, task_chain: TaskChain) -> TaskChain:
        """移动到下一步

        Args:
            task_chain: 任务链

        Returns:
            更新后的任务链
        """
        current_index = task_chain["current_step_index"]
        steps = task_chain["steps"]

        if current_index < len(steps):
            # 标记当前步骤为完成
            steps[current_index]["status"] = "completed"

            # 移动到下一步
            task_chain["current_step_index"] = current_index + 1

            logger.info(
                f"移动到下一步: chain={task_chain['chain_id']}, "
                f"new_index={task_chain['current_step_index']}"
            )

        return task_chain


# 单例实例
_task_orchestrator: Optional[TaskChainOrchestrator] = None


def get_task_orchestrator() -> TaskChainOrchestrator:
    """获取 TaskChainOrchestrator 单例

    Returns:
        TaskChainOrchestrator 实例
    """
    global _task_orchestrator

    if _task_orchestrator is None:
        _task_orchestrator = TaskChainOrchestrator()

    return _task_orchestrator


def reset_task_orchestrator() -> None:
    """重置 TaskChainOrchestrator 单例（用于测试）"""
    global _task_orchestrator
    _task_orchestrator = None
