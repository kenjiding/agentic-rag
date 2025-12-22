"""Agentic RAG 完整图实现 - 2025 最佳实践版

包含:
1. 意图识别节点
2. 检索节点（支持 BM25 + Dense + Rerank）
3. 生成节点
4. 决策节点
5. Web Search 节点 (Corrective RAG)
"""
from agentic_rag.advance_detector import AdvancedNeedsMoreInfoDetector
from langgraph.graph import StateGraph, END
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from typing import Optional
from src.agentic_rag.state import AgenticRAGState
from src.agentic_rag.retriever import IntelligentRetriever
from src.agentic_rag.generator import IntelligentGenerator
from src.agentic_rag.nodes import (
    create_intent_classification_node,
    create_retrieve_node,
    create_generate_node,
    create_decision_node,
    create_web_search_node,
)
from src.agentic_rag.intent_analyse import IntentClassifier
from src.agentic_rag.threshold_config import ThresholdConfig
from src.agentic_rag.web_search import CorrectiveRAGHandler, WebSearchTool


def create_agentic_rag_graph(
    vectorstore: Chroma,
    llm: ChatOpenAI = None,
    max_iterations: int = 5,
    threshold_config: Optional[ThresholdConfig] = None,
    enable_web_search: bool = True
):
    """
    创建完整的 Agentic RAG 图 - 2025 最佳实践版

    Args:
        vectorstore: 向量数据库
        llm: LLM 实例
        max_iterations: 最大迭代次数
        threshold_config: 阈值配置（如果为None，使用默认配置）
        enable_web_search: 是否启用 Web 搜索（Corrective RAG）

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

    # 创建意图分类器（如果启用）
    intent_classifier = None
    if threshold_config.intent_classification.enable_intent_classification:
        intent_classifier = IntentClassifier(
            llm=llm,
            threshold_config=threshold_config
        )

    # 创建 CRAG 处理器（如果启用 Web 搜索）
    crag_handler = None
    if enable_web_search:
        web_search_tool = WebSearchTool(max_results=3)
        crag_handler = CorrectiveRAGHandler(
            web_search_tool=web_search_tool,
            llm=llm,
            quality_threshold=0.5,
            max_web_results=3
        )

    # 创建节点（传递 threshold_config）
    intent_node = None
    if intent_classifier:
        intent_node = create_intent_classification_node(intent_classifier, threshold_config=threshold_config)

    retrieve_node = create_retrieve_node(retriever, threshold_config=threshold_config)
    generate_node = create_generate_node(generator, threshold_config=threshold_config)

    # 创建决策节点（统一 API，通过 crag_handler 参数控制是否启用 Web Search）
    decision_node = create_decision_node(
        detector,
        crag_handler=crag_handler if enable_web_search else None,
        threshold_config=threshold_config
    )

    # 创建 Web Search 节点（如果启用）
    web_search_node = None
    if enable_web_search and crag_handler:
        web_search_node = create_web_search_node(crag_handler, threshold_config=threshold_config)

    # 创建状态图
    graph = StateGraph(AgenticRAGState)

    # 添加节点
    if intent_node:
        graph.add_node("intent", intent_node)
    graph.add_node("decision", decision_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    # 添加 Web Search 节点（如果启用）
    if web_search_node:
        graph.add_node("web_search", web_search_node)

    # 设置入口点：如果启用意图识别，先进行意图识别，否则直接进入决策节点
    if intent_node:
        graph.set_entry_point("intent")
        # 意图识别后进入决策节点
        graph.add_edge("intent", "decision")
    else:
        graph.set_entry_point("decision")

    # 添加条件边（包含 Web Search）
    if web_search_node:
        graph.add_conditional_edges(
            "decision",
            lambda state: state.get("next_action", "finish"),
            {
                "retrieve": "retrieve",
                "generate": "generate",
                "web_search": "web_search",
                "finish": END
            }
        )
        # Web Search 后回到决策节点
        graph.add_edge("web_search", "decision")
    else:
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
    max_iterations: int = 5
) -> AgenticRAGState:
    """
    创建初始状态 - 2025 最佳实践版

    Args:
        question: 用户问题
        max_iterations: 最大迭代次数

    Returns:
        初始状态
    """
    return {
        "question": question,
        "query_intent": None,  # 意图识别结果（将在intent节点中填充）
        "retrieved_docs": [],
        "retrieval_history": [],
        "retrieval_quality": 0.0,
        "retrieval_strategy": "semantic",
        "answer": "",
        "generation_history": [],
        "answer_quality": 0.0,
        "evaluation_feedback": "",
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "next_action": None,
        # Web Search (Corrective RAG) 相关
        "web_search_used": False,
        "web_search_results": [],
        "web_search_count": 0,
        # 元数据
        "error_message": "",
        "tools_used": []
    }
