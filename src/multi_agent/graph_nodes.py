"""Graph节点处理器 - 封装所有节点执行逻辑（一步一步智能模式）

将节点执行逻辑从主图类中分离，提高代码可维护性和可测试性。

2025-2026 最佳实践：
- 动态节点调用：通过Agent名称动态获取Agent实例
- 注册表驱动：从AgentRegistry获取Agent描述符
- 统一执行逻辑：所有Agent使用统一的执行流程
- 特殊逻辑下放：Agent特定的状态更新逻辑由Agent.execute返回值处理
- interrupt()支持：捕获并转换GraphInterrupt为状态更新
"""
import logging
from typing import Dict, Any, Optional, Callable
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphInterrupt

from src.multi_agent.state import MultiAgentState, StepDisplay
from src.multi_agent.constants import ActionName, MetadataKeys, AgentName
from src.multi_agent.constants import SystemNodeName
from src.multi_agent.planning.models import (
    Plan,
    PlanStep,
    PlanStatus,
    PlanStepStatus,
    PlanStepType,
    RiskLevel,
    PolicyMethod,
    PlanningOutput,
    StepCondition,
    StepConditionType,
)
from src.multi_agent.planning.planner import Planner
from src.multi_agent.response_models import TextResponse, ErrorResponse
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt
from src.multi_agent.interrupt_framework import create_input_interrupt

logger = logging.getLogger(__name__)


class GraphNodeHandler:
    """图节点处理器 - 封装所有节点执行逻辑（一步一步智能模式）

    使用注册表模式，支持动态Agent调用。新增Agent时无需修改此文件。
    """

    def __init__(self, graph_instance):
        """
        初始化节点处理器

        Args:
            graph_instance: MultiAgentGraph实例，用于访问agents和注册表
        """
        self.graph = graph_instance
        self._planner: Planner | None = None

    def _get_planner(self) -> Planner:
        if self._planner is None:
            self._planner = Planner(llm=self.graph.llm)
        return self._planner

    # =========================
    # 展示信息辅助方法
    # =========================
    # Agent 展示名称映射（用于 plan_executor 展示当前执行的 Agent）
    _AGENT_DISPLAY_NAMES = {
        AgentName.RAG_AGENT: ("📚 知识检索", "知识库检索助手"),
        AgentName.CHAT_AGENT: ("💬 对话处理", "智能对话助手"),
        AgentName.PRODUCT_AGENT: ("🛍️ 商品搜索", "商品搜索助手"),
        AgentName.ORDER_AGENT: ("📦 订单管理", "订单管理助手"),
        AgentName.CONSULTATION_AGENT: ("💡 咨询服务", "咨询服务助手"),
        AgentName.BROWSER_AGENT: ("🌐 网页搜索", "网页搜索助手"),
    }

    def _build_agent_step_display(
        self,
        agent: AgentName,
        instruction: str = ""
    ) -> StepDisplay:
        """构建 Agent 执行步骤的展示信息"""
        short_name, full_name = self._AGENT_DISPLAY_NAMES.get(
            agent,
            (f"🤖 {agent.value}", agent.value)
        )
        detail = f"正在执行 {full_name}"
        if instruction:
            # 截取指令前60字符
            instr_display = instruction[:60] + "..." if len(instruction) > 60 else instruction
            detail = f"正在执行 {full_name}: {instr_display}"
        return StepDisplay.create(name=f"⚡ 执行: {short_name}", detail=detail)

    # =========================
    # 声明式条件评估器（替代硬编码的 fallback 逻辑）
    # =========================
    def _evaluate_step_condition(
        self,
        step: PlanStep,
        state: MultiAgentState,
    ) -> tuple[bool, str]:
        """评估步骤的执行条件

        设计原则（企业级最佳实践）：
        - 条件评估与业务逻辑解耦：executor 不知道具体的 agent 实现细节
        - 声明式：条件定义在 Plan 中，不在代码中硬编码
        - 可扩展：新增条件类型只需扩展此方法，不需修改其他代码
        - 可审计：返回 skip_reason 便于追踪和调试

        Args:
            step: 当前要执行的步骤
            state: 当前状态

        Returns:
            (should_execute, skip_reason):
            - should_execute=True: 应该执行此步骤
            - should_execute=False, skip_reason: 应该跳过，返回跳过原因
        """
        condition = step.execution_condition

        # 无条件或 ALWAYS → 执行
        if condition is None or condition.type == StepConditionType.ALWAYS:
            return True, ""

        # IF_PREVIOUS_EMPTY：仅当引用 agent 返回空结果时执行
        if condition.type == StepConditionType.IF_PREVIOUS_EMPTY:
            reference_agent = condition.reference_agent
            result_key = condition.result_key

            if reference_agent is None or result_key is None:
                # 配置错误，默认执行（防御性编程）
                logger.warning(
                    f"[条件评估] IF_PREVIOUS_EMPTY 缺少 reference_agent 或 result_key，默认执行"
                )
                return True, ""

            # 获取引用 agent 的结果
            agent_result = state.agent_results.get(reference_agent.value)
            logger.info(
                f"[条件评估] step={step.step_id}, condition=IF_PREVIOUS_EMPTY, "
                f"reference_agent={reference_agent.value}, result_key={result_key}"
            )

            if agent_result is None:
                # 引用 agent 没有结果（可能未执行），执行当前步骤
                logger.info(f"[条件评估] {reference_agent.value} 无结果，执行 {step.step_id}")
                return True, ""

            # 检查 result_key 对应的值是否为空
            result_value = (
                agent_result.get(result_key, [])
                if isinstance(agent_result, dict)
                else []
            )
            is_empty = not result_value or (
                isinstance(result_value, list) and len(result_value) == 0
            )

            if is_empty:
                # 结果为空 → 执行当前步骤（fallback 场景）
                logger.info(
                    f"[条件评估] {reference_agent.value}.{result_key} 为空，执行 {step.step_id}"
                )
                return True, ""
            else:
                # 结果非空 → 跳过当前步骤
                skip_reason = (
                    f"{reference_agent.value}_has_{result_key}"
                    f"(count={len(result_value) if isinstance(result_value, list) else 1})"
                )
                logger.info(
                    f"[条件评估] {reference_agent.value}.{result_key} 非空，跳过 {step.step_id}"
                )
                return False, skip_reason

        # 未知条件类型 → 默认执行
        logger.warning(f"[条件评估] 未知条件类型 {condition.type}，默认执行")
        return True, ""

    def _get_agent(self, agent_name: str):
        """根据名称获取Agent实例

        Args:
            agent_name: Agent名称

        Returns:
            Agent实例，如果不存在返回None
        """
        return getattr(self.graph, agent_name, None)

    async def context_manager_node(self, state: MultiAgentState) -> MultiAgentState:
        """上下文管理节点 - 智能提取和压缩上下文

        职责：
        1. 提取当前查询（最后一条HumanMessage）
        2. 使用ContextManager构建上下文摘要
        3. 更新state.context_summary

        Args:
            state: 当前状态

        Returns:
            更新后的状态（包含context_summary）
        """
        try:
            # 1. 提取当前查询（从messages中获取最后一条HumanMessage）
            current_query = None
            for msg in reversed(state.messages):
                if hasattr(msg, 'content') and msg.content:
                    from langchain_core.messages import HumanMessage
                    if isinstance(msg, HumanMessage):
                        current_query = msg.content
                        break

            if not current_query:
                current_query = state.original_question or ""

            # 2. 使用ContextPipeline构建统一上下文与摘要
            context_bundle = await self.graph.context_pipeline.build(
                state=state,
                current_query=current_query
            )

            logger.info(
                f"📊【上下文管理】构建摘要完成: "
                f"{len(context_bundle.short_term_context.get('conversation_history', []))}轮对话, "
                f"{len(context_bundle.short_term_context.get('recent_tool_calls', []))}个工具调用"
            )

            # 3. 更新state
            # 更新上下文缓存元数据（轻量缓存）
            context_cache = {
                "message_count": len(state.messages),
                "history_rounds": len(context_bundle.short_term_context.get("conversation_history", [])),
                "tool_calls_count": len(context_bundle.short_term_context.get("recent_tool_calls", [])),
            }
            context_version = (state.metadata or {}).get("context_version", 0) + 1

            return {
                "context_bundle": context_bundle.model_dump(),
                "original_question": current_query,
                "metadata": {
                    **state.metadata,
                    MetadataKeys.CONTEXT_CACHE.value: context_cache,
                    MetadataKeys.CONTEXT_VERSION.value: context_version,
                    MetadataKeys.CONTEXT_OWNER.value: "context_manager",
                },
                # 上下文管理是内部节点，不展示给前端
                "step_display": StepDisplay.hidden(),
            }

        except Exception as e:
            logger.error(f"上下文管理节点执行错误: {str(e)}", exc_info=True)
            # 返回None，避免阻塞流程
            return {
                "context_bundle": None,
                "original_question": state.original_question,
                "step_display": StepDisplay.hidden(),
            }

    # intent_recognition_node removed:
    # Intent/entities extraction is merged into the planner for single-shot consistency.

    async def policy_gate_node(self, state: MultiAgentState) -> MultiAgentState:
        """Policy gate node - risk assessment & governance decision.

        This node must be deterministic and side-effect-free. It only writes
        risk metadata into LangGraph state, so downstream nodes can enforce
        confirmation/approval paths.
        """
        try:
            # Root-cause fix:
            # Never infer risk from raw keywords (negation/contrast breaks it).
            # Risk is derived from the structured plan produced by the planner node.
            plan = state.plan
            if plan is None or not plan.steps:
                risk = RiskLevel.MEDIUM
                method = PolicyMethod.NO_PLAN_FALLBACK
            else:
                # Deterministic aggregation: overall risk = max(step risk)
                order = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
                max_risk = RiskLevel.LOW
                for s in plan.steps:
                    if order.get(s.risk_level, 0) > order[max_risk]:
                        max_risk = s.risk_level
                risk = max_risk
                method = PolicyMethod.MAX_STEP_RISK

            audit_event = {
                "node": SystemNodeName.POLICY_GATE.value,
                "event": "risk_assessed",
                "risk_level": risk,
                "method": method,
            }

            # 总是展示安全检查步骤
            risk_display_map = {
                RiskLevel.LOW: "低风险",
                RiskLevel.MEDIUM: "中风险",
                RiskLevel.HIGH: "高风险",
            }
            step_display = StepDisplay.create(
                name="🔒 安全检查",
                detail=f"风险评估: {risk_display_map.get(risk, str(risk))}"
            )

            return {
                "risk_level": risk,
                "action_audit": state.action_audit + [audit_event],
                "step_display": step_display,
            }
        except Exception as e:
            logger.error(f"policy_gate_node error: {e}", exc_info=True)
            return {
                "risk_level": RiskLevel.HIGH,
                "action_audit": state.action_audit + [{
                    "node": SystemNodeName.POLICY_GATE.value,
                    "event": "risk_assess_failed",
                    "error": str(e),
                }],
                "step_display": StepDisplay.create(
                    name="🔒 安全检查",
                    detail="风险评估失败，默认高风险处理"
                ),
            }

    async def planner_node(self, state: MultiAgentState) -> MultiAgentState:
        """Planner node - produce PlanningOutput (query_intent + plan) stored in state."""
        try:
            # Extract latest user query.
            #
            # Root-cause fix:
            # In interrupt/resume flows, `original_question` can be stale because the resume
            # does not pass through GraphStateManager.restore_state_from_checkpointer(question=...).
            # Always prefer the latest HumanMessage as the "current turn" user query.
            user_query = ""
            for msg in reversed(state.messages):
                if hasattr(msg, "content") and msg.content and isinstance(msg, HumanMessage):
                    user_query = msg.content
                    break
            if not user_query:
                user_query = state.original_question or ""

            if not user_query:
                response_model = TextResponse(content="未找到用户输入，无法进行规划。")
                return {
                    "plan": None,
                    **response_model.to_full_response(),
                }

            planner = self._get_planner()
            output: PlanningOutput = await planner.plan(
                user_query=user_query,
                context_bundle=state.context_bundle,
            )

            # Structural repair (no text heuristics):
            # The planner must not output a plan that immediately FINISHes without any
            # actionable step (agent_call / ask_user). If it happens, repair it
            # deterministically using the structured business intent (routing signal).
            plan = output.plan
            has_actionable_step = any(
                s.step_type in (PlanStepType.AGENT_CALL, PlanStepType.ASK_USER)
                for s in (plan.steps or [])
            )
            finish_first = bool(plan.steps) and plan.steps[0].step_type == PlanStepType.FINISH

            if not has_actionable_step or finish_first:
                # Do NOT rely on LLM-provided action strings.
                # Action binding is a deterministic system concern.
                bit = getattr(output.query_intent, "business_intent_type", None)
                action = {
                    "order_management": ActionName.ORDER_MANAGEMENT,
                    "product_search": ActionName.PRODUCT_SEARCH,
                    "product_comparison": ActionName.CONSULTATION,
                    "social_chat": ActionName.CHAT,
                    "general_chat": ActionName.CHAT,
                }.get(str(bit), ActionName.CHAT)

                # Reuse the single source of truth for execution binding:
                # ActionName -> agent node mapping lives in GraphRouter, not here.
                tmp_state = MultiAgentState(next_action=action)
                node_name = self.graph.router.route_after_supervisor(tmp_state)
                try:
                    agent = AgentName(node_name)
                except Exception:
                    agent = AgentName.CHAT_AGENT
                    action = ActionName.CHAT

                instruction = f"处理用户请求（repair: action={action.value}）"

                output.plan = Plan(
                    goal=plan.goal or "处理用户请求",
                    steps=[
                        PlanStep(
                            step_id="auto_step_1",
                            step_type=PlanStepType.AGENT_CALL,
                            risk_level=RiskLevel.LOW,
                            selected_agent=agent,
                            next_action=action,
                            instruction=instruction,
                            inputs={},
                        ),
                        PlanStep(
                            step_id="finish",
                            step_type=PlanStepType.FINISH,
                            risk_level=RiskLevel.LOW,
                            instruction="结束对话。",
                            inputs={},
                        ),
                    ],
                )

            # Persist intent & entities into state (single source of truth)
            intent_dict = output.query_intent.model_dump()
            entities_update = {}
            if output.query_intent.entities:
                entities_update = output.query_intent.entities.model_dump(exclude_none=True)
            merged_entities = {**(state.entities or {}), **(entities_update or {})}

            audit_event = {
                "node": SystemNodeName.PLANNER.value,
                "event": "plan_created",
                "goal": output.plan.goal,
                "steps": [s.step_id for s in output.plan.steps],
                "status": output.plan.status,
            }

            # 构建展示信息
            goal_display = output.plan.goal
            if len(goal_display) > 50:
                goal_display = goal_display[:50] + "..."

            return {
                "query_intent": intent_dict,
                "entities": merged_entities,
                "plan": output.plan,
                "action_audit": state.action_audit + [audit_event],
                # Keep state consistent for downstream nodes/observability
                "original_question": user_query,
                "step_display": StepDisplay.create(
                    name="📋 制定计划",
                    detail=f"已制定执行计划: {goal_display}"
                ),
            }

        except Exception as e:
            logger.error(f"planner_node error: {e}", exc_info=True)
            response_model = ErrorResponse(
                content="规划失败，系统将降级处理。",
                error_message=str(e),
                error_code="PLANNER_ERROR",
            )
            return {
                "plan": None,
                "action_audit": state.action_audit + [{
                    "node": SystemNodeName.PLANNER.value,
                    "event": "plan_failed",
                    "error": str(e),
                }],
                **response_model.to_full_response(),
                "step_display": StepDisplay.create(
                    name="📋 分析问题",
                    detail="规划失败，系统将降级处理"
                ),
            }

    async def plan_executor_node(self, state: MultiAgentState) -> MultiAgentState:
        """Plan executor node - advance plan progress and choose next_action/current_agent.

        Execution loop:
        planner -> plan_executor -> agent -> plan_executor -> ... -> finish
        """
        plan = state.plan
        if plan is None:
            # Fallback: no plan, route to supervisor (existing behavior)
            return {
                "next_action": None,
                "current_agent": None,
                "routing_reason": "无可用计划，交由supervisor进行路由。",
                "step_display": StepDisplay.create(
                    name="⚡ 执行计划",
                    detail="无可用计划，交由智能路由处理"
                ),
            }

        # End conditions
        if plan.current_step_index >= len(plan.steps):
            plan.status = PlanStatus.COMPLETED
            audit_event = {
                "node": SystemNodeName.PLAN_EXECUTOR.value,
                "event": "plan_completed",
                "goal": plan.goal,
            }
            return {
                "plan": plan,
                "next_action": ActionName.FINISH,
                "current_agent": None,
                "routing_reason": "计划已完成。",
                "action_audit": state.action_audit + [audit_event],
                "step_display": StepDisplay.create(
                    name="✅ 完成处理",
                    detail="所有步骤已完成"
                ),
            }

        step = plan.current_step()
        if step is None:
            plan.status = PlanStatus.FAILED
            plan.failure_reason = "Plan step index out of range"
            return {
                "plan": plan,
                "next_action": ActionName.FINISH,
                "current_agent": None,
                "routing_reason": "计划执行失败：step越界。",
                "step_display": StepDisplay.create(
                    name="⚠️ 执行失败",
                    detail="计划执行失败：步骤索引越界"
                ),
            }

        # Dispatch per step type
        if step.step_type == PlanStepType.ASK_USER:
            # Use LangGraph interrupt/resume (root-cause fix):
            # - avoids losing plan progress due to re-planning on the next user turn
            # - makes “ask user” an explicit, resumable state transition
            plan.status = PlanStatus.NEEDS_USER_INPUT
            step.status = PlanStepStatus.IN_PROGRESS
            plan.steps[plan.current_step_index] = step

            prompt_text = step.ask_user_message or "请补充必要信息。"
            interrupt_data = create_input_interrupt(
                action_type="plan_input",
                action_data={"plan_goal": plan.goal, "step_id": step.step_id},
                display_message=prompt_text,
                input_id=f"plan_input::{step.step_id}",
                label="请补充必要信息",
                placeholder="请输入补充信息...",
                multiline=True,
                required=True,
                submit_label="提交并继续",
                node=SystemNodeName.PLAN_EXECUTOR.value,
                expected=plan.missing_information,
            )

            audit_event = {
                "node": SystemNodeName.PLAN_EXECUTOR.value,
                "event": "interrupt_ask_user",
                "step_id": step.step_id,
            }

            resume_value = interrupt(interrupt_data)
            # If we got here, we are resuming execution and have resume_value.
            # We treat the resume as an additional user message and inject it into context.
            user_input = None
            if isinstance(resume_value, dict):
                user_input = resume_value.get("input_value") or resume_value.get("text")
            if isinstance(resume_value, str):
                user_input = resume_value

            if user_input:
                plan.status = PlanStatus.ACTIVE
                step.status = PlanStepStatus.COMPLETED
                plan.steps[plan.current_step_index] = step
                plan.current_step_index += 1

                context_bundle = dict(state.context_bundle or {})
                task_input = dict((context_bundle.get("task_input") or {}))
                task_input["clarification"] = {"input": user_input, "step_id": step.step_id}
                context_bundle["task_input"] = task_input

                # 【关键修复】检查是否还有下一个step需要执行
                # 如果plan只有一个ASK_USER step，current_step_index会超出范围
                # 此时应该重新规划，因为用户提供了新的输入信息
                if plan.current_step_index >= len(plan.steps):
                    # 没有下一个step了，应该重新规划以处理用户的输入
                    # 清除plan，让系统重新规划（基于新的用户输入）
                    logger.info(
                        f"Plan已完成所有steps，但用户提供了新的输入。"
                        f"将清除plan以触发重新规划（goal={plan.goal}）"
                    )
                    return {
                        "plan": None,  # 清除plan，触发重新规划
                        "context_bundle": context_bundle,
                        "messages": state.messages + [HumanMessage(content=user_input)],
                        "original_question": user_input,
                        "action_audit": state.action_audit
                        + [audit_event, {"node": SystemNodeName.PLAN_EXECUTOR.value, "event": "resumed_input"}, {
                            "node": SystemNodeName.PLAN_EXECUTOR.value,
                            "event": "plan_cleared_for_replanning",
                            "reason": "user_input_after_all_steps_completed"
                        }],
                        "next_action": None,  # 让路由系统回到planner
                        "current_agent": None,
                    }

                # 还有下一个step，继续执行计划
                next_step = plan.current_step()
                if next_step is None:
                    # 这种情况理论上不应该发生，因为我们已经检查了index范围
                    plan.status = PlanStatus.FAILED
                    plan.failure_reason = "Next step not found after ASK_USER resume"
                    logger.error(f"Plan执行错误：ASK_USER恢复后找不到下一个step (current_index={plan.current_step_index}, steps_len={len(plan.steps)})")
                    return {
                        "plan": plan,
                        "context_bundle": context_bundle,
                        "messages": state.messages + [HumanMessage(content=user_input)],
                        "next_action": ActionName.FINISH,
                        "current_agent": None,
                        "routing_reason": "计划执行失败：找不到下一个step",
                        "action_audit": state.action_audit
                        + [audit_event, {"node": SystemNodeName.PLAN_EXECUTOR.value, "event": "resumed_input"}, {
                            "node": SystemNodeName.PLAN_EXECUTOR.value,
                            "event": "error_after_resume",
                            "error": plan.failure_reason
                        }],
                    }
                
                if next_step.step_type == PlanStepType.AGENT_CALL:
                    # 下一个step是AGENT_CALL，设置next_action让它执行
                    return {
                        "plan": plan,
                        "context_bundle": context_bundle,
                        "messages": state.messages + [HumanMessage(content=user_input)],
                        "original_question": user_input,
                        "next_action": next_step.next_action,
                        "current_agent": next_step.selected_agent,
                        "routing_reason": f"继续执行计划步骤: {next_step.step_id} (用户输入后)",
                        "action_audit": state.action_audit
                        + [audit_event, {"node": SystemNodeName.PLAN_EXECUTOR.value, "event": "resumed_input"}],
                        "step_display": self._build_agent_step_display(
                            agent=next_step.selected_agent,
                            instruction=next_step.instruction
                        ),
                    }
                elif next_step.step_type == PlanStepType.FINISH:
                    # 下一个step是FINISH，标记plan完成
                    plan.status = PlanStatus.COMPLETED
                    return {
                        "plan": plan,
                        "context_bundle": context_bundle,
                        "messages": state.messages + [HumanMessage(content=user_input)],
                        "original_question": user_input,
                        "next_action": ActionName.FINISH,
                        "current_agent": None,
                        "routing_reason": "计划完成（用户输入后）",
                        "action_audit": state.action_audit
                        + [audit_event, {"node": SystemNodeName.PLAN_EXECUTOR.value, "event": "resumed_input"}],
                        "step_display": StepDisplay.create(
                            name="✅ 完成处理",
                            detail="用户补充信息后计划完成"
                        ),
                    }
                elif next_step.step_type == PlanStepType.ASK_USER:
                    logger.info(f"下一个step也是ASK_USER ({next_step.step_id})，需要继续在plan_executor中处理")
                    return {
                        "plan": plan,
                        "context_bundle": context_bundle,
                        "messages": state.messages + [HumanMessage(content=user_input)],
                        "original_question": user_input,
                        # 不设置next_action，让路由系统知道需要继续执行plan
                        "action_audit": state.action_audit
                        + [audit_event, {"node": SystemNodeName.PLAN_EXECUTOR.value, "event": "resumed_input"}],
                    }
                else:
                    # 未知的step类型
                    logger.warning(f"未知的step类型: {next_step.step_type}")
                    return {
                        "plan": plan,
                        "context_bundle": context_bundle,
                        "messages": state.messages + [HumanMessage(content=user_input)],
                        "original_question": user_input,
                        "action_audit": state.action_audit
                        + [audit_event, {"node": SystemNodeName.PLAN_EXECUTOR.value, "event": "resumed_input"}],
                    }

            # No resume input provided; keep plan waiting (should be rare)
            return {
                "plan": plan,
                "action_audit": state.action_audit + [audit_event],
            }

        if step.step_type == PlanStepType.FINISH:
            plan.status = PlanStatus.COMPLETED
            audit_event = {
                "node": SystemNodeName.PLAN_EXECUTOR.value,
                "event": "finish_step",
                "step_id": step.step_id,
            }
            return {
                "plan": plan,
                "next_action": ActionName.FINISH,
                "current_agent": None,
                "routing_reason": "计划结束。",
                "action_audit": state.action_audit + [audit_event],
                "step_display": StepDisplay.create(
                    name="✅ 完成处理",
                    detail="计划执行完成"
                ),
            }

        # AGENT_CALL: 使用声明式条件评估器判断是否执行
        # 设计原则：条件定义在 Plan 中（由 Planner 生成），executor 只做通用评估
        should_execute, skip_reason = self._evaluate_step_condition(step, state)

        if not should_execute:
            # 条件不满足，跳过此步骤
            step.status = PlanStepStatus.SKIPPED
            step.outputs = {"skipped_reason": skip_reason}
            plan.steps[plan.current_step_index] = step
            plan.current_step_index += 1

            audit_event = {
                "node": SystemNodeName.PLAN_EXECUTOR.value,
                "event": "skip_conditional_step",
                "step_id": step.step_id,
                "reason": skip_reason,
                "condition": step.execution_condition.model_dump() if step.execution_condition else None,
            }

            # 继续执行下一步（如果有）
            if plan.current_step_index >= len(plan.steps):
                # 已经是最后一步，完成计划
                plan.status = PlanStatus.COMPLETED
                return {
                    "plan": plan,
                    "next_action": ActionName.FINISH,
                    "current_agent": None,
                    "routing_reason": "计划完成（跳过条件步骤后）",
                    "action_audit": state.action_audit + [audit_event],
                    "step_display": StepDisplay.create(
                        name="✅ 完成处理",
                        detail="计划执行完成"
                    ),
                }
            else:
                # 还有下一步，继续执行（不展示跳过信息，减少噪音）
                return {
                    "plan": plan,
                    "action_audit": state.action_audit + [audit_event],
                    "step_display": StepDisplay.hidden(),
                }

        step.status = PlanStepStatus.IN_PROGRESS
        plan.steps[plan.current_step_index] = step

        # Inject current step into context_bundle so downstream agent prompts can use it.
        # This is the root-cause fix for multi-step execution: agents must know "what to do now"
        # beyond the original user message.
        context_bundle = dict(state.context_bundle or {})
        task_input = dict((context_bundle.get("task_input") or {}))
        task_input["plan_step"] = {
            "step_id": step.step_id,
            "step_type": step.step_type,
            "instruction": step.instruction,
            "inputs": step.inputs,
            "risk_level": step.risk_level,
        }
        context_bundle["task_input"] = task_input

        audit_event = {
            "node": SystemNodeName.PLAN_EXECUTOR.value,
            "event": "dispatch_agent",
            "step_id": step.step_id,
            "agent": step.selected_agent,
            "action": step.next_action,
            "risk_level": step.risk_level,
        }
        return {
            "plan": plan,
            "next_action": step.next_action,
            "current_agent": step.selected_agent,
            "routing_reason": f"执行计划步骤: {step.step_id} ({step.instruction})",
            "action_audit": state.action_audit + [audit_event],
            "context_bundle": context_bundle,
            # 展示当前正在执行的 Agent（用户明确要求必须展示）
            "step_display": self._build_agent_step_display(
                agent=step.selected_agent,
                instruction=step.instruction
            ),
        }

    async def post_action_verifier_node(self, state: MultiAgentState) -> MultiAgentState:
        """Post-action verifier node.

        Goal:
        - Make step completion explicit and based on observable outputs (no silent success).
        - Record verification outcome in action_audit.
        """
        plan = state.plan
        if plan is None or plan.is_done():
            return {"step_display": StepDisplay.hidden()}

        step = plan.current_step()
        if step is None:
            plan.status = PlanStatus.FAILED
            plan.failure_reason = "No current step to verify"
            return {
                "plan": plan,
                "action_audit": state.action_audit + [{
                    "node": SystemNodeName.POST_ACTION_VERIFIER.value,
                    "event": "verify_failed",
                    "reason": plan.failure_reason,
                }],
                "next_action": ActionName.FINISH,
                "routing_reason": "计划执行失败：无可验证步骤。",
                "step_display": StepDisplay.create(
                    name="⚠️ 验证失败",
                    detail="计划执行失败：无可验证步骤"
                ),
            }

        # Basic verification: treat any state.error_message as failure.
        # (We will expand to per-action verifiers later: order/refund/compensation, etc.)
        ok = state.error_message is None
        step.status = PlanStepStatus.COMPLETED if ok else PlanStepStatus.FAILED
        plan.steps[plan.current_step_index] = step

        audit_event = {
            "node": SystemNodeName.POST_ACTION_VERIFIER.value,
            "event": "verified",
            "step_id": step.step_id,
            "ok": ok,
            "error_message": state.error_message,
        }

        if not ok:
            plan.status = PlanStatus.FAILED
            plan.failure_reason = state.error_message or "unknown_error"
            # 截取错误信息前40字符
            error_display = plan.failure_reason[:40] + "..." if len(plan.failure_reason) > 40 else plan.failure_reason
            return {
                "plan": plan,
                "action_audit": state.action_audit + [audit_event],
                "next_action": ActionName.FINISH,
                "routing_reason": f"步骤验证失败：{plan.failure_reason}",
                "step_display": StepDisplay.create(
                    name="⚠️ 验证失败",
                    detail=f"步骤验证失败: {error_display}"
                ),
            }

        # Verified OK: advance index and continue plan
        # 验证成功时不展示（减少噪音）
        plan.current_step_index += 1
        return {
            "plan": plan,
            "action_audit": state.action_audit + [audit_event],
            "step_display": StepDisplay.hidden(),
        }

    async def supervisor_node(
        self, state: MultiAgentState, config: Optional[RunnableConfig] = None
    ) -> MultiAgentState:
        """Supervisor节点 - 路由决策"""
        try:
            iteration_count = state.iteration_count
            if iteration_count >= self.graph.max_iterations:
                logger.warning(f"达到最大迭代次数 {self.graph.max_iterations}，结束执行")
                return {
                    "next_action": ActionName.FINISH,
                    "routing_reason": f"达到最大迭代次数 {self.graph.max_iterations}"
                }

            routing_decision = await self.graph.supervisor.route(state)

            updated_state = {
                "next_action": routing_decision["next_action"],
                "current_agent": routing_decision.get("selected_agent"),
                "routing_reason": routing_decision.get("routing_reason", ""),
                "iteration_count": iteration_count + 1
            }

            logger.info(
                f"Supervisor决策: {routing_decision.get('next_action')} → {routing_decision.get('selected_agent')}"
            )
            return updated_state

        except Exception as e:
            logger.error(f"Supervisor节点执行错误: {str(e)}", exc_info=True)
            return {
                "next_action": ActionName.FINISH,
                "error_message": f"Supervisor错误: {str(e)}",
                "routing_reason": f"执行错误: {str(e)}"
            }

    async def _execute_agent_node(
        self,
        state: MultiAgentState,
        agent_name: str,
        config: Optional[RunnableConfig] = None
    ) -> MultiAgentState:
        """通用Agent节点执行逻辑

        统一处理所有Agent的执行，特殊逻辑由Agent.execute返回值处理。

        支持 LangGraph 1.x interrupt() 机制：
        - 捕获 GraphInterrupt 异常
        - 转换为 __interrupt__ 状态更新传递给客户端
        - 客户端使用 Command(resume=...) 恢复执行

        Args:
            state: 当前状态
            agent_name: Agent名称
            config: LangGraph配置（用于传递session_id等）

        Returns:
            更新后的状态片段
        """
        try:
            agent = self._get_agent(agent_name)
            if not agent:
                logger.error(f"{agent_name} 未找到")
                return {
                    "next_action": ActionName.FINISH,
                    "error_message": f"{agent_name} 未找到"
                }

            # 获取session_id（用于order_agent等需要会话的Agent）
            session_id = "default"
            if config and "configurable" in config:
                session_id = config["configurable"].get("session_id", "default")

            # 所有Agent统一接受session_id参数
            result = await agent.execute(state, session_id=session_id)

            # Agent 已经在 messages 中添加了所有需要的新消息
            # result["result"] 中的字段只是用于存储数据，不应该再次添加为消息
            additional_messages = result.get("messages", [])
            agent_result = result.get("result")

            # 合并基础状态更新
            updated_state = {
                "messages": state.messages + additional_messages,
                "agent_results": {
                    **state.agent_results,
                    agent_name: agent_result
                },
                "agent_history": state.agent_history + [{
                    "agent": agent_name,
                    "result": agent_result,
                    "metadata": result.get("metadata", {})
                }]
            }

            # 合并Agent返回的所有其他字段（支持Agent自定义状态更新）
            for key, value in result.items():
                if key not in ["messages", "result", "metadata"]:
                    # 特殊处理entities字段：合并而不是覆盖
                    if key == "entities" and isinstance(value, dict) and isinstance(state.entities, dict):
                        updated_state[key] = {**state.entities, **value}
                    else:
                        updated_state[key] = value

            logger.info(f"{agent_name} 执行完成")
            return updated_state

        except GraphInterrupt as e:
            # LangGraph 1.x interrupt() 机制
            # 捕获 interrupt() 调用，转换为状态更新传递给客户端
            # GraphInterrupt 的结构: (Interrupt(value={...}),)
            # 需要从 e.args[0].value 获取实际的值
            interrupt_value = None
            if e.args and len(e.args) > 0:
                interrupt_obj = e.args[0]
                if hasattr(interrupt_obj, 'value'):
                    interrupt_value = interrupt_obj.value
                else:
                    interrupt_value = interrupt_obj

            # 【关键修复】GraphInterrupt 的 value 可能是 (Interrupt(...),) 这样的 tuple
            # 需要继续解析获取实际的字典值
            if interrupt_value and isinstance(interrupt_value, tuple) and len(interrupt_value) > 0:
                first_element = interrupt_value[0]
                if hasattr(first_element, 'value'):
                    # Interrupt 对象，获取其 value 属性
                    interrupt_value = first_element.value
                elif isinstance(first_element, dict):
                    # 直接是字典
                    interrupt_value = first_element

            # 【关键修复】GraphInterrupt 会被 LangGraph 捕获，不会将返回值包含在 stream 输出中
            # 所以我们需要重新抛出异常，让 LangGraph 处理
            # LangGraph 会将 interrupt 信息保存到 checkpointer，客户端可以通过 get_state() 获取
            raise

        except Exception as e:
            logger.error(f"{agent_name} 节点执行错误: {str(e)}", exc_info=True)
            return {
                "next_action": ActionName.FINISH,
                "error_message": f"{agent_name} 错误: {str(e)}"
            }

    def create_agent_node(self, agent_name: str) -> Callable:
        """创建Agent节点函数（工厂方法）

        根据Agent名称创建对应的节点函数，用于LangGraph图构建。
        新增Agent时无需添加新的节点方法，只需调用此工厂方法即可。

        Args:
            agent_name: Agent名称

        Returns:
            节点函数

        Example:
            graph.add_node("rag_agent", node_handler.create_agent_node("rag_agent"))
            graph.add_node("chat_agent", node_handler.create_agent_node("chat_agent"))
        """
        async def agent_node(state: MultiAgentState, config: Optional[RunnableConfig] = None) -> MultiAgentState:
            return await self._execute_agent_node(state, agent_name, config)

        agent_node.__name__ = f"{agent_name}_node"
        return agent_node
