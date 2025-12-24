"""Agentic RAG 完整图实现 - 2025 企业级最佳实践版

包含:
1. 意图识别节点（支持动态重识别）
2. 检索节点（支持自适应多轮检索）
3. 生成节点
4. 决策节点（集成失败分析和策略调整）

2025 企业级最佳实践：
- 自适应多轮检索：每轮使用不同的查询和策略
- 失败分析驱动：根据上一轮失败原因调整策略
- 动态意图重识别：多轮失败后重新分析意图
- 渐进式策略升级：逐步放宽参数以获得更好结果
"""
from src.agentic_rag.advance_detector import AdvancedNeedsMoreInfoDetector
from langgraph.graph import StateGraph, END
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from typing import Optional, Dict, Any
from src.agentic_rag.state import AgenticRAGState
from src.agentic_rag.retriever import IntelligentRetriever
from src.agentic_rag.generator import IntelligentGenerator
from src.agentic_rag.nodes import (
    create_intent_classification_node,
    create_retrieve_node,
    create_generate_node,
    create_decision_node,
)
from src.intent import IntentClassifier
from src.agentic_rag.threshold_config import ThresholdConfig


def create_agentic_rag_graph(
    vectorstore: Chroma,
    llm: ChatOpenAI = None,
    max_iterations: int = 3,
    threshold_config: Optional[ThresholdConfig] = None,
    skip_intent_classification: bool = False
):
    """
    创建完整的 Agentic RAG 图 - 2025 企业级最佳实践版

    新增功能：
    - 自适应多轮检索
    - 失败分析驱动的策略调整
    - 动态意图重识别

    Args:
        vectorstore: 向量数据库
        llm: LLM 实例
        max_iterations: 最大迭代次数
        threshold_config: 阈值配置（如果为None，使用默认配置）
        skip_intent_classification: 是否跳过意图识别（当通过multi_agent进入时设为True，因为已经做过意图识别）

    Returns:
        编译后的图
    """
    # 使用默认配置如果未提供
    if threshold_config is None:
        threshold_config = ThresholdConfig.default()

    # 初始化组件（传递 threshold_config）
    if llm is None:
        # 使用配置的生成器温度
        generator_temp = threshold_config.generator.default_temperature
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=generator_temp)

    # 初始化检索器（支持 BM25 + Dense + Rerank）
    retriever = IntelligentRetriever(
        vectorstore=vectorstore,
        llm=llm,
        threshold_config=threshold_config,
        enable_bm25=True,  # 启用 BM25
        enable_reranker=True  # 启用 Cross-encoder 重排序
    )

    generator = IntelligentGenerator(
        llm=llm,
        threshold_config=threshold_config
    )

    detector = AdvancedNeedsMoreInfoDetector(
        llm=llm,
        vectorstore=vectorstore,
        use_llm=True,
        use_embedding=True,
        threshold_config=threshold_config
    )

    # 创建意图分类器（如果启用且未跳过）
    intent_classifier = None
    if not skip_intent_classification and threshold_config.intent_classification.enable_intent_classification:
        intent_classifier = IntentClassifier(
            llm=llm,
            config=threshold_config.intent_classification.to_intent_config()
        )

    # 创建节点（传递 threshold_config）
    intent_node = None
    if intent_classifier and not skip_intent_classification:
        intent_node = create_intent_classification_node(intent_classifier, threshold_config=threshold_config)

    retrieve_node = create_retrieve_node(retriever, threshold_config=threshold_config)
    generate_node = create_generate_node(generator, threshold_config=threshold_config)

    # 创建决策节点（统一 API，支持自适应检索）
    decision_node = create_decision_node(
        detector,
        threshold_config=threshold_config
    )

    # 创建状态图
    graph = StateGraph(AgenticRAGState)

    # 添加节点
    if intent_node:
        graph.add_node("intent", intent_node)
    graph.add_node("decision", decision_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    # 设置入口点
    if intent_node:
        graph.set_entry_point("intent")
        # 意图识别后进入决策节点
        graph.add_edge("intent", "decision")
    else:
        graph.set_entry_point("decision")

    # 构建条件边的路由映射
    def get_next_node(state: AgenticRAGState) -> str:
        """决定下一个节点"""
        next_action = state.get("next_action", "finish")

        # 动态意图重识别：如果需要重识别且有意图节点，转到意图节点
        if next_action == "reclassify_intent" and intent_node:
            return "intent"

        # 其他情况按照 next_action 路由
        return next_action

    # 添加条件边
    if intent_node:
        # 包含动态意图重识别
        graph.add_conditional_edges(
            "decision",
            get_next_node,
            {
                "retrieve": "retrieve",
                "generate": "generate",
                "reclassify_intent": "intent",  # 动态意图重识别
                "finish": END
            }
        )
    else:
        # 基础版
        graph.add_conditional_edges(
            "decision",
            lambda state: state.get("next_action", "finish"),
            {
                "retrieve": "retrieve",
                "generate": "generate",
                "finish": END
            }
        )

    # 检索后回到决策节点
    graph.add_edge("retrieve", "decision")

    # 生成后回到决策节点
    graph.add_edge("generate", "decision")

    # 编译图
    compiled_graph = graph.compile()

    return compiled_graph


def create_initial_state(
    question: str,
    max_iterations: int = 3,
    query_intent: Optional[Dict[str, Any]] = None
) -> AgenticRAGState:
    """
    创建初始状态 - 2025 企业级最佳实践版

    包含自适应检索所需的所有字段

    Args:
        question: 用户问题
        max_iterations: 最大迭代次数
        query_intent: 意图识别结果（可选，当从multi_agent传入时使用）

    Returns:
        初始状态
    """
    return {
        "question": question,

        # 意图识别相关
        "query_intent": query_intent,  # 意图识别结果（如果从multi_agent传入则使用）
        "intent_reclassification_count": 0,  # 意图重识别次数

        # 检索相关
        "retrieved_docs": [],
        "retrieval_history": [],
        "retrieval_quality": 0.0,
        "retrieval_strategy": "semantic",

        # 自适应检索相关 - 简化版
        "failure_analysis": None,  # 检索失败分析结果（核心功能）
        "current_round_config": None,  # 当前轮次的检索配置

        # 生成相关
        "answer": "",
        "generation_history": [],
        "answer_quality": 0.0,
        "evaluation_feedback": "",

        # 控制流
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "next_action": None,

        # 元数据
        "error_message": "",
        "tools_used": []
    }
