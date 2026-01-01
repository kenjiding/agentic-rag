"""任务编排引擎

TaskChainOrchestrator 负责：
1. 检测是否为多步骤任务
2. 创建任务链（支持LLM动态生成）
3. 执行当前步骤
4. 管理任务链生命周期

2025-2026 最佳实践：
- 使用 LLM-based 意图识别替代硬编码规则
- 结合规则和 LLM：规则作为快速路径，LLM 作为精确判断
- LLM动态生成任务链，无需枚举所有场景
- 利用已有的意图分类器基础设施
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import logging
import re

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.multi_agent.state import MultiAgentState, TaskChain, TaskStep
from src.multi_agent.config import get_keywords_config
from src.confirmation.selection_manager import get_selection_manager

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
    
    2025企业级最佳实践：
    1. 使用 LLM-based 检测替代硬编码规则
    2. 保留规则作为快速路径（可选）
    3. 更好的语义理解和上下文感知
    4. LLM动态生成任务链，无需枚举所有场景
    """
    
    # 定义可用的步骤类型和能力（作为LLM的上下文）
    AVAILABLE_STEP_TYPES = {
        "product_search": {
            "description": "搜索商品，返回商品列表",
            "agent_name": "product_agent",
            "output": "products: List[Product]",
            "requires": ["search_keyword"]
        },
        "user_selection": {
            "description": "用户从选项中选择（需要用户交互）",
            "agent_name": None,  # 需要用户交互
            "output": "selected_item: Any",
            "requires": ["options"]
        },
        "order_creation": {
            "description": "准备订单信息（调用prepare_create_order准备订单详情，但不创建确认。订单信息会保存到result_data中，供后续confirmation步骤使用）",
            "agent_name": "order_agent",
            "output": "order_info: Dict（包含订单详情，如items、total_amount等）",
            "requires": ["product_id", "quantity", "user_phone"],
            "note": "此步骤只准备订单信息，不创建确认。必须在order_creation之后添加confirmation步骤来确认订单"
        },
        "confirmation": {
            "description": "确认操作（需要用户确认。如果上一步是order_creation，则确认订单创建；否则用于其他确认场景）",
            "agent_name": None,
            "output": "confirmed: bool（确认后执行相应操作）",
            "requires": ["confirmation_data（从上一步骤的result_data中获取）"],
            "note": "用于订单确认或其他需要用户确认的操作"
        },
        "web_search": {
            "description": "网络搜索，获取最新信息",
            "agent_name": "rag_agent",
            "output": "search_results: List[str]",
            "requires": ["query"]
        },
        "rag_search": {
            "description": "RAG检索，从知识库中搜索相关信息",
            "agent_name": "rag_agent",
            "output": "retrieved_docs: List[Document]",
            "requires": ["query"]
        }
    }

    def __init__(self, llm: Optional[ChatOpenAI] = None, use_fast_path: bool = True):
        """初始化任务编排器
        
        Args:
            llm: 语言模型实例，用于 LLM-based 检测（如果为 None，则创建默认实例）
            use_fast_path: 是否使用规则作为快速路径（默认 True，提高性能）
        """
        self.selection_manager = get_selection_manager()
        self.use_fast_path = use_fast_path
        
        # 初始化 LLM（用于精确检测和动态生成）
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.structured_llm = self.llm.with_structured_output(MultiStepTaskDetection)
        
        logger.info(f"TaskChainOrchestrator 初始化完成 (use_fast_path={use_fast_path})")

    async def detect_multi_step_task(self, state: MultiAgentState) -> Optional[str]:
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
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

        if not user_message:
            logger.debug("[多步骤检测] 无用户消息，返回 None")
            return None

        logger.info(f"[多步骤检测] 用户消息: {user_message[:50]}...")

        # === 快速路径：规则检测（可选，用于提高性能）===
        if self.use_fast_path:
            logger.debug("[多步骤检测] 尝试快速路径检测")
            fast_result = self._fast_path_detection(user_message)
            if fast_result is not None:
                logger.info(f"[多步骤检测] ✓ 快速路径检测到: {fast_result}")
                return fast_result
            logger.debug("[多步骤检测] 快速路径未检测到，继续")

        # === 精确路径：LLM-based 检测 ===
        llm_result = await self._llm_based_detection(user_message, state)
        if llm_result:
            logger.info(f"[多步骤检测] ✓ LLM检测到: {llm_result}")
            return llm_result

        # LLM 检测失败或返回 None，使用降级规则
        fallback_result = self._fallback_rule_detection(user_message)
        if fallback_result:
            logger.info(f"[多步骤检测] ✓ 降级规则检测到: {fallback_result}")
        else:
            logger.info(f"[多步骤检测] ✗ 所有检测方法均未检测到多步骤任务")
        return fallback_result

    def _fast_path_detection(self, user_message: str) -> Optional[str]:
        """快速路径：基于规则的快速检测

        用于快速排除明显不是多步骤任务的情况。
        如果检测到明确的多步骤任务信号，直接返回结果。

        使用配置化的关键词列表，支持扩展和多语言。
        """
        user_message_lower = user_message.lower()
        keywords_config = get_keywords_config()

        # 快速排除：已经有明确的 product_id
        has_product_id = re.search(r"product[_\s]*id[:\s]*\d+|商品[_\s]*id[:\s]*\d+|商品编号[:\s]*\d+", user_message_lower)
        if has_product_id:
            logger.info(f"[快速路径] 检测到product_id，跳过多步骤任务检测")
            return None

        # 快速排除：完全没有购买意图（使用配置化关键词）
        matched_order_keywords = [kw for kw in keywords_config.order_intent_keywords if kw in user_message_lower]
        has_order_intent = len(matched_order_keywords) > 0
        logger.info(f"[快速路径] 购买意图关键词匹配: {matched_order_keywords}, has_intent={has_order_intent}")

        if not has_order_intent:
            return None

        # 检测到购买意图，检查是否有商品关键词（使用配置化关键词）
        all_product_keywords = keywords_config.get_all_product_keywords()
        matched_product_keywords = [kw for kw in all_product_keywords if kw.lower() in user_message_lower]
        has_product_keyword = len(matched_product_keywords) > 0
        logger.info(f"[快速路径] 商品关键词匹配: {matched_product_keywords}, has_product={has_product_keyword}")

        if has_product_keyword:
            # 明确的多步骤任务信号，直接返回
            logger.info(f"[快速路径] ✓ 检测到明确的多步骤任务信号 (order_with_search)")
            return "order_with_search"

        logger.info(f"[快速路径] 未检测到商品关键词，返回 None")
        return None

    async def _llm_based_detection(self, user_message: str, state: MultiAgentState) -> Optional[str]:
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
            # 使用异步LLM调用提高性能
            result = await chain.ainvoke({"user_message": user_message})
            
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
        """降级方案：基于规则的检测

        使用配置化的关键词列表，支持扩展和多语言。
        """
        user_message_lower = user_message.lower()
        keywords_config = get_keywords_config()

        # 检测购买意图（使用配置化关键词）
        has_order_intent = any(kw in user_message_lower for kw in keywords_config.order_intent_keywords)

        # 检测商品关键词（使用配置化关键词）
        has_product_keyword = any(
            kw.lower() in user_message_lower
            for kw in keywords_config.get_all_product_keywords()
        )

        # 检测是否已经有明确的 product_id
        has_product_id = bool(re.search(
            r"product[_\s]*id[:\s]*\d+|商品[_\s]*id[:\s]*\d+|商品编号[:\s]*\d+",
            user_message_lower
        ))

        if has_order_intent and has_product_keyword and not has_product_id:
            logger.info("降级规则检测到多步骤任务: order_with_search")
            return "order_with_search"

        return None

    async def create_task_chain(
        self,
        task_type: str,
        initial_state: MultiAgentState
    ) -> TaskChain:
        """根据任务类型创建任务链
        
        2025企业级最佳实践：支持LLM动态生成任务链
        - 已知任务类型使用快速路径（如 order_with_search）
        - 未知任务类型使用LLM动态生成

        Args:
            task_type: 任务类型
            initial_state: 初始状态

        Returns:
            TaskChain 对象
        """
        # 快速路径：已知的任务类型
        # if task_type == "order_with_search":
        #     return self._create_order_with_search_chain(initial_state)
        
        # LLM动态生成：未知的任务类型
        logger.info(f"使用LLM动态生成任务链: task_type={task_type}")
        return await self._llm_generate_task_chain(task_type, initial_state)

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
        entities = state.entities
        context_data = {
            "user_phone": entities.get("user_phone"),
            "quantity": entities.get("quantity", 1),
            "search_keyword": entities.get("search_keyword"),
        }

        logger.info(f"从 entities 读取任务链上下文: {context_data}")

        task_chain: TaskChain = TaskChain(
            chain_id=str(uuid.uuid4()),
            chain_type="order_with_search",
            steps=[
                TaskStep(
                    step_id="search-1",
                    step_type="product_search",
                    status="pending",
                    agent_name="product_agent",
                    result_data=None,
                    metadata=None
                ),
                TaskStep(
                    step_id="select-1",
                    step_type="user_selection",
                    status="pending",
                    agent_name=None,
                    result_data=None,
                    metadata=None
                ),
                TaskStep(
                    step_id="order-1",
                    step_type="order_creation",
                    status="pending",
                    agent_name="order_agent",
                    result_data=None,
                    metadata=None
                )
            ],
            current_step_index=0,
            context_data=context_data,
            created_at=datetime.utcnow().isoformat()
        )

        logger.info(
            f"创建任务链: id={task_chain.chain_id}, "
            f"type={task_chain.chain_type}, "
            f"steps={len(task_chain.steps)}"
        )

        return task_chain

    async def _llm_generate_task_chain(
        self,
        task_type: str,
        initial_state: MultiAgentState
    ) -> TaskChain:
        """使用LLM动态生成任务链
        
        2025企业级最佳实践：让LLM根据任务需求自动组合步骤。
        无需枚举所有场景，LLM会根据可用步骤类型和用户需求动态生成。

        Args:
            task_type: 任务类型
            initial_state: 初始状态
            
        Returns:
            TaskChain 对象
        """
        from pydantic import BaseModel as PydanticBaseModel
        
        class TaskStepPlan(PydanticBaseModel):
            """任务步骤计划"""
            step_id: str = Field(description="步骤唯一标识，如 'search-1', 'select-1'")
            step_type: str = Field(description="步骤类型，必须是可用的步骤类型之一")
            agent_name: Optional[str] = Field(
                default=None,
                description="执行该步骤的Agent名称，如果为None则表示需要用户交互"
            )
            description: str = Field(description="步骤描述，说明这个步骤要做什么")
            required_context: List[str] = Field(
                default_factory=list,
                description="需要的上下文数据字段列表，如 ['search_keyword', 'quantity']"
            )
            output_context: List[str] = Field(
                default_factory=list,
                description="输出的上下文数据字段列表，如 ['products', 'selected_product_id']"
            )
        
        class TaskChainPlan(PydanticBaseModel):
            """任务链计划"""
            chain_type: str = Field(description="任务链类型，描述性名称")
            steps: List[TaskStepPlan] = Field(description="步骤列表，按执行顺序排列")
            context_data: Dict[str, Any] = Field(
                default_factory=dict,
                description="初始上下文数据，包含任务所需的所有初始信息（键值对形式）"
            )
            reasoning: str = Field(description="为什么这样设计任务链，解释设计思路")
        
        # 构建可用步骤的描述
        available_steps_desc = "\n".join([
            f"- **{step_id}**: {info['description']}\n"
            f"  - 需要输入: {', '.join(info['requires'])}\n"
            f"  - 输出: {info['output']}\n"
            f"  - Agent: {info['agent_name'] or '需要用户交互'}"
            + (f"\n  - ⚠️ 注意: {info.get('note', '')}" if info.get('note') else "")
            for step_id, info in self.AVAILABLE_STEP_TYPES.items()
        ])
        
        # 从状态中提取信息
        entities = initial_state.entities
        user_message = None
        for msg in reversed(initial_state.messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        # LLM生成任务链计划
        template = """你是一个任务编排专家。请根据用户需求，设计一个多步骤任务链。

# 任务类型
{task_type}

# 用户消息
{user_message}

# 可用步骤类型
{available_steps}

# 当前上下文（已提取的实体信息）
{context}

# 重要规则（单一职责原则）
1. **order_creation 步骤的职责**：只负责准备订单信息（调用 prepare_create_order），将订单详情保存到 result_data 中，**不创建确认请求**。
2. **confirmation 步骤的职责**：负责确认操作。如果上一步是 order_creation，则从 result_data 中读取订单信息并创建确认请求；确认后执行订单创建。
3. **订单创建的标准流程**：order_creation → confirmation（必须按此顺序，遵循单一职责原则）

# 要求
1. 分析用户需求，确定需要哪些步骤来完成这个任务
2. 按照逻辑顺序排列步骤，确保前一步的输出能满足下一步的输入需求
3. 从可用步骤类型中选择，不要创建新的步骤类型
4. 如果某个步骤需要用户交互（如选择），使用 user_selection
5. **订单创建必须包含两个步骤**：order_creation（准备订单信息）→ confirmation（确认订单）
6. 确保步骤之间的数据流是连贯的（前一步的输出要包含下一步需要的输入）
7. 初始上下文数据应该包含从用户消息中提取的所有必要信息

# 输出要求
请设计任务链，包括：
- chain_type: 任务链类型（描述性名称）
- steps: 步骤列表（每个步骤包含 step_id, step_type, agent_name, description, required_context, output_context）
- context_data: 初始上下文数据（从用户消息和entities中提取）
- reasoning: 设计理由（解释为什么这样设计）

请确保步骤类型必须是可用步骤类型之一，不要创建新的步骤类型。"""

        prompt = ChatPromptTemplate.from_template(template)
        # 使用 function_calling 方法，因为 OpenAI structured output 不支持开放的 Dict[str, Any]
        chain_plan_llm = self.llm.with_structured_output(
            TaskChainPlan,
            method="function_calling"
        )
        chain = prompt | chain_plan_llm
        
        try:
            # 使用异步LLM调用提高性能
            plan = await chain.ainvoke({
                "task_type": task_type,
                "user_message": user_message or "",
                "available_steps": available_steps_desc,
                "context": str(entities) if entities else "无"
            })
            
            # 将计划转换为TaskChain
            return self._plan_to_task_chain(plan, task_type, initial_state)
            
        except Exception as e:
            logger.error(f"LLM生成任务链失败: {e}", exc_info=True)
            # 降级：返回通用任务链
            logger.warning(f"降级到通用任务链: {task_type}")
            return self._create_fallback_chain(task_type, initial_state)
    
    def _plan_to_task_chain(
        self,
        plan: Any,  # TaskChainPlan (定义在方法内部，使用Any避免前向引用)
        task_type: str,
        initial_state: MultiAgentState
    ) -> TaskChain:
        """将LLM生成的计划转换为TaskChain对象
        
        Args:
            plan: LLM生成的任务链计划
            initial_state: 初始状态
            
        Returns:
            TaskChain 对象
        """
        steps = []
        for i, step_plan in enumerate(plan.steps):
            # 验证步骤类型是否可用
            if step_plan.step_type not in self.AVAILABLE_STEP_TYPES:
                logger.warning(
                    f"LLM生成的步骤类型 {step_plan.step_type} 不可用，跳过。"
                    f"可用类型: {list(self.AVAILABLE_STEP_TYPES.keys())}"
                )
                continue
            
            step_info = self.AVAILABLE_STEP_TYPES[step_plan.step_type]
            # 使用LLM指定的agent_name，如果没有则使用默认值
            agent_name = step_plan.agent_name or step_info.get("agent_name")

            step = TaskStep(
                step_id=step_plan.step_id or f"{step_plan.step_type}-{i+1}",
                step_type=step_plan.step_type,  # type: ignore
                status="pending",
                agent_name=agent_name,
                result_data=None,
                metadata={
                    "description": step_plan.description,
                    "required_context": step_plan.required_context,
                    "output_context": step_plan.output_context,
                    "reasoning": plan.reasoning,
                    "generated_by": "llm"
                }
            )
            steps.append(step)

        # 合并初始上下��数据
        entities = initial_state.entities
        context_data = {**entities, **plan.context_data}

        task_chain = TaskChain(
            chain_id=str(uuid.uuid4()),
            chain_type=plan.chain_type or f"dynamic_{task_type}",
            steps=steps,
            current_step_index=0,
            context_data=context_data,
            created_at=datetime.utcnow().isoformat()
        )
        
        logger.info(
            f"LLM动态生成任务链: type={task_chain.chain_type}, "
            f"steps={len(steps)}, reasoning={plan.reasoning[:100]}..."
        )
        
        return task_chain
    
    def _create_fallback_chain(
        self,
        task_type: str,
        initial_state: MultiAgentState
    ) -> TaskChain:
        """创建降级任务链（当LLM生成失败时使用）
        
        Args:
            task_type: 任务类型
            initial_state: 初始状态
            
        Returns:
            通用的降级任务链
        """
        entities = initial_state.entities
        context_data = entities.copy()

        # 创建一个通用的任务链，包含基本的步骤
        task_chain = TaskChain(
            chain_id=str(uuid.uuid4()),
            chain_type=f"fallback_{task_type}",
            steps=[
                TaskStep(
                    step_id="rag-search-1",
                    step_type="rag_search",  # type: ignore
                    status="pending",
                    agent_name="rag_agent",
                    result_data=None,
                    metadata={
                        "description": "使用RAG搜索相关信息",
                        "fallback": True
                    }
                )
            ],
            current_step_index=0,
            context_data=context_data,
            created_at=datetime.utcnow().isoformat()
        )
        
        logger.warning(
            f"创建降级任务链: type={task_chain.chain_type}, "
            f"steps={len(task_chain.steps)}"
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
        task_chain = state.task_chain
        if not task_chain:
            logger.error("task_chain 不存在")
            return {"next_action": "finish"}

        current_index = task_chain.current_step_index
        steps = task_chain.steps

        if current_index >= len(steps):
            logger.info("任务链已完成")
            return {
                "next_action": "finish",
                "task_chain": None  # 清除任务链
            }

        current_step: TaskStep = steps[current_index]
        step_type = current_step.step_type

        logger.info(
            f"[execute_current_step] 执行步骤: index={current_index}, "
            f"type={step_type}, "
            f"agent={current_step.agent_name}, "
            f"chain_id={task_chain.chain_id}"
        )

        if step_type == "product_search":
            logger.info(f"[execute_current_step] 路由到 product_search")
            return self._execute_product_search(state, task_chain, current_step)
        elif step_type == "user_selection":
            logger.info(f"[execute_current_step] 路由到 user_selection，准备执行用户选择步骤")
            return await self._execute_user_selection(state, task_chain, current_step, session_id)
        elif step_type == "order_creation":
            return self._execute_order_creation(state, task_chain, current_step)
        elif step_type == "rag_search":
            return self._execute_rag_search(state, task_chain, current_step)
        elif step_type == "web_search":
            return self._execute_web_search(state, task_chain, current_step)
        elif step_type == "confirmation":
            return await self._execute_confirmation(state, task_chain, current_step, session_id)
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

        注意：步骤状态在 product_agent 执行完成后更新，不需要在这里更新
        """
        # 路由到 product_agent，并传递context_data
        context_data = task_chain.context_data
        
        return {
            "next_action": "product_search",
            "selected_agent": "product_agent",
            "task_chain": task_chain,
            "context_data": context_data
        }

    async def _execute_user_selection(
        self,
        state: MultiAgentState,
        task_chain: TaskChain,
        current_step: TaskStep,
        session_id: str
    ) -> Dict[str, Any]:
        """执行用户选择步骤（LangGraph 1.x 最佳实践：使用 interrupt()）

        工作流程：
        1. 获取产品列表
        2. 创建选择请求（selection_manager）
        3. 调用 interrupt() 暂停图执行
        4. 用户选择后，通过 /api/selection/resolve 提交
        5. 使用 Command(resume=...) 恢复执行

        LangGraph 1.x interrupt() 机制：
        - interrupt() 会保存当前状态到 checkpointer
        - 图执行暂停，等待 resume 值
        - 恢复时 interrupt() 返回 resume 值
        """
        # 从任务链的上一步骤获取产品列表
        products = None
        current_index = task_chain.current_step_index
        if current_index > 0:
            prev_step = task_chain.steps[current_index - 1]
            if prev_step.step_type == "product_search":
                result_data = prev_step.result_data or {}
                products = result_data.get("products")

        # 降级：从 agent_results 获取
        if not products:
            agent_results = state.agent_results or {}
            product_result = agent_results.get("product_agent", {})
            products = product_result.get("products") if isinstance(product_result, dict) else None

        # 降级：从 state.messages 中提取
        if not products:
            import json
            from langchain_core.messages import ToolMessage

            for msg in reversed(state.messages):
                if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
                    try:
                        data = json.loads(msg.content)
                        if isinstance(data, dict) and "products" in data:
                            extracted_products = data["products"]
                            if extracted_products:
                                products = extracted_products
                                logger.info(f"从 messages 中的 ToolMessage 提取到 {len(products)} 个产品")
                                break
                    except (json.JSONDecodeError, TypeError):
                        continue

        if not products or not isinstance(products, list) or len(products) == 0:
            logger.error("未找到产品列表，无法创建选择")
            return {
                "next_action": "finish",
                "task_chain": None,
                "messages": state.messages + [
                    AIMessage(content="抱歉，未找到相关商品，请更换关键词重试。")
                ]
            }

        # 创建选择请求
        search_keyword = task_chain.context_data.get("search_keyword", "商品")

        # 构建选择信息供 interrupt() 使用
        selection_info = {
            "selection_id": str(uuid.uuid4()),
            "selection_type": "product",
            "options": products,
            "display_message": f"请选择要购买的{search_keyword}:",
            "metadata": {"task_chain_id": task_chain.chain_id, "session_id": session_id}
        }

        # 保存到 selection_manager（供前端查询）
        selection = await self.selection_manager.request_selection(
            session_id=session_id,
            selection_type="product",
            options=products,
            display_message=selection_info["display_message"],
            metadata={"task_chain_id": task_chain.chain_id}
        )

        # 更新 selection_info 中的 selection_id 为实际创建的 ID
        selection_info["selection_id"] = selection.selection_id

        logger.info(f"[用户选择步骤] 创建选择请求: selection_id={selection_info['selection_id']}, options_count={len(products)}")
        logger.info(f"[用户选择步骤] 准备调用 interrupt()，当前 task_chain: chain_id={task_chain.chain_id}, current_step_index={task_chain.current_step_index}, session_id={session_id}")

        # 【LangGraph 1.x】使用 interrupt() 暂停图执行
        # interrupt() 的返回值在恢复时就是用户的选择结果
        # 注意：interrupt() 会抛出 GraphInterrupt 异常，状态会在异常处理时保存到 checkpointer
        from langgraph.types import interrupt

        # 【关键调试】在调用 interrupt() 之前记录完整的状态
        logger.info(f"[用户选择步骤] ========================================")
        logger.info(f"[用户选择步骤] 准备调用 interrupt()")
        logger.info(f"[用户选择步骤] 当前 task_chain.chain_id={task_chain.chain_id}")
        logger.info(f"[用户选择步骤] 当前 task_chain.current_step_index={task_chain.current_step_index}")
        logger.info(f"[用户选择步骤] 当前 task_chain.steps_count={len(task_chain.steps)}")
        logger.info(f"[用户选择步骤] session_id={session_id}")
        logger.info(f"[用户选择步骤] selection_id={selection_info['selection_id']}")
        logger.info(f"[用户选择步骤] state 完整内容: {state.model_dump()}")

        logger.info(f"[用户选择步骤] 调用 interrupt()，selection_info={selection_info}, session_id={session_id}")
        logger.info(f"[用户选择步骤] interrupt() 将抛出 GraphInterrupt 异常，状态将保存到 checkpointer，thread_id 应该与 session_id 一致: {session_id}")
        user_selection = interrupt(selection_info)
        logger.warning(f"[用户选择步骤] interrupt() 返回（不应该执行到这里），这表明 interrupt() 没有抛出异常")

        # 恢复执行后，user_selection 包含用户的选择
        # 格式：{"selected_option_id": "1"}
        logger.info(f"[用户选择步骤] interrupt() 恢复执行，收到用户选择: {user_selection}")

        # 验证用户选择
        selected_option_id = user_selection.get("selected_option_id") if isinstance(user_selection, dict) else None
        if not selected_option_id:
            logger.warning(f"[用户选择步骤] 用户选择无效: {user_selection}")
            return {
                "next_action": "finish",
                "task_chain": None,
                "messages": state.messages + [
                    AIMessage(content="选择无效，请重新开始。")
                ]
            }

        logger.info(f"[用户选择步骤] 验证用户选择成功: selected_option_id={selected_option_id}")

        # 查找选择的产品
        selected_product = None
        for product in products:
            product_id = product.get("id") or product.get("product_id")
            if str(product_id) == str(selected_option_id):
                selected_product = product
                break

        if not selected_product:
            logger.warning(f"[用户选择步骤] 选择的产品不存在: selected_option_id={selected_option_id}, products_count={len(products)}")
            return {
                "next_action": "finish",
                "task_chain": None,
                "messages": state.messages + [
                    AIMessage(content="选择的产品不存在，请重新开始。")
                ]
            }

        logger.info(f"[用户选择步骤] 找到选择的产品: product_id={selected_product.get('id')}, product_name={selected_product.get('name')}")

        # 更新步骤为完成状态（使用 model_copy 创建新实例，因为 Pydantic 模型不可变）
        updated_step = current_step.model_copy(update={
            "status": "completed",
            "result_data": {
                "selected_product": selected_product,
                "selected_option_id": selected_option_id
            }
        })

        # 更新任务链：先更新当前步骤，然后移动到下一步
        current_index = task_chain.current_step_index
        logger.info(f"[用户选择步骤] 更新任务链步骤: current_index={current_index}, steps_count={len(task_chain.steps)}")
        updated_steps = list(task_chain.steps)
        updated_steps[current_index] = updated_step
        updated_task_chain = task_chain.model_copy(update={"steps": updated_steps})
        
        # 移动到下一步
        updated_task_chain = self.move_to_next_step(updated_task_chain)
        next_index = updated_task_chain.current_step_index
        logger.info(f"[用户选择步骤] 移动到下一步: new_index={next_index}, steps_count={len(updated_task_chain.steps)}")

        # 检查下一步是否需要路由到其他 agent
        steps = updated_task_chain.steps

        if next_index < len(steps):
            next_step = steps[next_index]
            next_step_type = next_step.step_type
            logger.info(f"[用户选择步骤] 下一步类型: step_type={next_step_type}")

            # 根据下一步类型设置 next_action
            if next_step_type == "order_creation":
                logger.info(f"[用户选择步骤] 路由到 order_agent 执行订单创建")
                return {
                    "task_chain": updated_task_chain,
                    "next_action": "order_management",
                    "selected_agent": "order_agent",
                    "selected_product": selected_product
                }
            elif next_step_type == "confirmation":
                logger.info(f"[用户选择步骤] 路由到 order_agent 执行确认")
                return {
                    "task_chain": updated_task_chain,
                    "next_action": "order_management",
                    "selected_agent": "order_agent",
                    "selected_product": selected_product
                }
        else:
            logger.warning(f"[用户选择步骤] 没有更多步骤，任务链已完成")

        # 默认完成
        logger.info(f"[用户选择步骤] 返回完成状态")
        return {
            "task_chain": updated_task_chain,
            "next_action": "finish"
        }

    def _execute_order_creation(
        self,
        state: MultiAgentState,
        task_chain: TaskChain,
        current_step: TaskStep
    ) -> Dict[str, Any]:
        """执行订单创建步骤"""
        current_step.status = "in_progress"
        context_data = task_chain.context_data

        return {
            "next_action": "order_management",
            "selected_agent": "order_agent",
            "task_chain": task_chain,
            "context_data": context_data
        }

    def _execute_rag_search(
        self,
        state: MultiAgentState,
        task_chain: TaskChain,
        current_step: TaskStep
    ) -> Dict[str, Any]:
        """执行RAG搜索步骤"""
        # 注意：步骤状态在执行完成时更新，不需要在这里更新
        context_data = task_chain.context_data

        return {
            "next_action": "rag_search",
            "selected_agent": "rag_agent",
            "task_chain": task_chain,
            "context_data": context_data
        }

    def _execute_web_search(
        self,
        state: MultiAgentState,
        task_chain: TaskChain,
        current_step: TaskStep
    ) -> Dict[str, Any]:
        """执行网络搜索步骤"""
        # 注意：步骤状态在执行完成时更新，不需要在这里更新
        context_data = task_chain.context_data

        # web_search 通过 rag_agent 的工具调用实现
        return {
            "next_action": "rag_search",  # 使用rag_agent，它会调用web_search工具
            "selected_agent": "rag_agent",
            "task_chain": task_chain,
            "context_data": context_data
        }

    async def _execute_confirmation(
        self,
        state: MultiAgentState,
        task_chain: TaskChain,
        current_step: TaskStep,
        session_id: str
    ) -> Dict[str, Any]:
        """执行确认步骤"""
        from langgraph.types import interrupt
        
        current_index = task_chain.current_step_index

        # 检查上一步是否是 order_creation（订单确认场景）
        if current_index > 0:
            prev_step = task_chain.steps[current_index - 1]
            if prev_step.step_type == "order_creation":
                # 订单确认场景：从 order_creation 的 result_data 中读取订单信息
                prev_result_data = prev_step.result_data or {}
                order_info = prev_result_data.get("order_info", {})
                
                if not order_info:
                    logger.error("未找到订单信息，无法创建确认请求")
                    return {
                        "next_action": "finish",
                        "task_chain": None,
                        "messages": state.messages + [
                            AIMessage(content="❌ 订单信息缺失，无法确认")
                        ]
                    }
                
                # 构建订单确认消息
                order_text = order_info.get("text", "订单信息")
                display_message = f"请确认订单信息：\n{order_text}"

                from src.confirmation.manager import get_confirmation_manager
                confirmation_manager = get_confirmation_manager()
                existing_confirmation = await confirmation_manager.get_pending_confirmation(session_id)
                
                if existing_confirmation and existing_confirmation.action_type == "create_order":
                    logger.info(f"[确认步骤] 第一次执行，使用已存在的确认请求: confirmation_id={existing_confirmation.confirmation_id}")
                    confirmation = existing_confirmation
                    confirmation_info = {
                        "confirmation_id": confirmation.confirmation_id,
                        "action_type": confirmation.action_type,
                        "display_message": confirmation.display_message,
                        "display_data": confirmation.display_data,
                        "metadata": {
                            "task_chain_id": task_chain.chain_id,
                            "step_id": current_step.step_id,
                            "session_id": session_id
                        }
                    }
                else:
                    logger.info("[确认步骤] 恢复执行场景，interrupt() 将直接返回 resume 值")
                    confirmation_info = {
                        "confirmation_id": None,
                        "action_type": "create_order",
                        "display_message": display_message,
                        "metadata": {
                            "task_chain_id": task_chain.chain_id,
                            "step_id": current_step.step_id,
                            "session_id": session_id
                        }
                    }
                
                logger.info(f"[确认步骤] 调用 interrupt()")
                user_confirmation = interrupt(confirmation_info)
                logger.info(f"[确认步骤] interrupt() 恢复执行，收到用户确认: {user_confirmation}")
                
                return self._handle_confirmation_result(
                    state, task_chain, current_step, current_index, user_confirmation
                )
        
        context_data = task_chain.context_data
        step_metadata = current_step.metadata or {}
        confirmation_data = step_metadata.get("confirmation_data") or context_data

        confirmation_message = (
            step_metadata.get("confirmation_message") or
            step_metadata.get("description") or
            "请确认是否继续执行此操作？"
        )

        agent_name = current_step.agent_name or "task_orchestrator"

        from src.confirmation.manager import get_confirmation_manager
        confirmation_manager = get_confirmation_manager()

        confirmation = await confirmation_manager.request_confirmation(
            session_id=session_id,
            action_type="task_chain_action",
            action_data=confirmation_data,
            agent_name=agent_name,
            display_message=confirmation_message,
            display_data={
                "task_chain_id": task_chain.chain_id,
                "step_id": current_step.step_id,
                "chain_type": task_chain.chain_type
            }
        )
        
        confirmation_info = {
            "confirmation_id": confirmation.confirmation_id,
            "action_type": confirmation.action_type,
            "display_message": confirmation.display_message,
            "metadata": {
                "task_chain_id": task_chain.chain_id,
                "step_id": current_step.step_id,
                "session_id": session_id
            }
        }
        
        logger.info(f"[确认步骤] 调用 interrupt()，confirmation_id={confirmation.confirmation_id}")
        user_confirmation = interrupt(confirmation_info)
        logger.info(f"[确认步骤] interrupt() 恢复执行，收到用户确认: {user_confirmation}")
        
        return self._handle_confirmation_result(
            state, task_chain, current_step, current_index, user_confirmation
        )

    def _handle_confirmation_result(
        self,
        state: MultiAgentState,
        task_chain: TaskChain,
        current_step: TaskStep,
        current_index: int,
        user_confirmation: Any
    ) -> Dict[str, Any]:
        """统一处理确认结果
        
        根据 LangGraph 最佳实践，interrupt() 返回 resume 值后，统一处理确认结果
        不区分第一次执行和恢复执行，因为逻辑完全相同
        
        Args:
            state: 当前状态
            task_chain: 任务链
            current_step: 当前步骤
            current_index: 当前步骤索引
            user_confirmation: interrupt() 返回的确认结果，格式：{"confirmed": True/False}
            
        Returns:
            更新后的状态
        """
        confirmed = user_confirmation.get("confirmed", False) if isinstance(user_confirmation, dict) else False
        
        if not confirmed:
            # 用户取消确认，结束任务链
            logger.info("[确认步骤] 用户取消确认")
            updated_step = current_step.model_copy(update={"status": "completed"})
            updated_steps = list(task_chain.steps)
            updated_steps[current_index] = updated_step
            updated_task_chain = task_chain.model_copy(update={"steps": updated_steps})
            
            return {
                "next_action": "finish",
                "task_chain": None,
                "messages": state.messages + [
                    AIMessage(content="已取消确认")
                ]
            }
        
        # 用户确认，标记步骤完成并移动到下一步
        logger.info("[确认步骤] 用户确认，标记步骤完成并移动到下一步")
        updated_step = current_step.model_copy(update={"status": "completed"})
        updated_steps = list(task_chain.steps)
        updated_steps[current_index] = updated_step
        updated_task_chain = task_chain.model_copy(update={"steps": updated_steps})
        updated_task_chain = self.move_to_next_step(updated_task_chain)
        
        # 检查任务链是否已完成
        if updated_task_chain.current_step_index >= len(updated_task_chain.steps):
            logger.info("[确认步骤] 任务链已完成")
            # 订单创建消息已在流式响应中发送，这里不再返回重复消息
            return {
                "next_action": "finish",
                "task_chain": None
            }
        
        # 继续执行任务链的下一步
        return {
            "task_chain": updated_task_chain,
            "next_action": "execute_task_chain"
        }

    def move_to_next_step(self, task_chain: TaskChain) -> TaskChain:
        """移动到下一步

        Args:
            task_chain: 任务链

        Returns:
            更新后的任务链
        """
        current_index = task_chain.current_step_index
        steps = task_chain.steps

        if current_index < len(steps):
            # 标记当前步骤为完成（使用 model_copy 创建新实例，因为 Pydantic 模型不可变）
            current_step = steps[current_index]
            updated_step = current_step.model_copy(update={"status": "completed"})
            updated_steps = list(steps)
            updated_steps[current_index] = updated_step
            
            # 移动到下一步（创建新的 task_chain 实例）
            updated_task_chain = task_chain.model_copy(update={
                "steps": updated_steps,
                "current_step_index": current_index + 1
            })

            logger.info(
                f"移动到下一步: chain={updated_task_chain.chain_id}, "
                f"new_index={updated_task_chain.current_step_index}"
            )
            
            return updated_task_chain

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
