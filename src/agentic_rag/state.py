"""Agentic RAG 状态定义"""
from typing import TypedDict, List, Literal, Optional, Dict, Any
from langchain_core.documents import Document


class AgenticRAGState(TypedDict):
    """Agentic RAG 完整状态定义

    2025 最佳实践：
    - 支持自适应多轮检索
    - 失败分析驱动的策略调整
    - 动态意图重识别
    """
    # 用户输入
    question: str

    # 意图识别相关
    query_intent: Optional[Dict[str, Any]]  # 查询意图信息（QueryIntent的字典形式）
    intent_reclassification_count: int  # 意图重识别次数（用于动态重识别）

    # 检索相关
    retrieved_docs: List[Document]  # 当前检索到的文档
    retrieval_history: List[List[Document]]  # 检索历史
    retrieval_quality: float  # 检索质量评分 (0-1)
    retrieval_strategy: str  # 当前使用的检索策略

    # 自适应检索相关 - 2025 企业级最佳实践
    failure_analysis: Optional[Dict[str, Any]]  # 检索失败分析结果
    query_variants: List[Dict[str, Any]]  # 查询变体列表（用于多轮检索）
    used_variants: List[str]  # 已使用过的查询变体（避免重复）
    current_round_config: Optional[Dict[str, Any]]  # 当前轮次的检索配置

    # 生成相关
    answer: str  # 当前生成的答案
    generation_history: List[str]  # 生成历史
    answer_quality: float  # 答案质量评分 (0-1)
    evaluation_feedback: str  # 评估反馈

    # 控制流
    iteration_count: int  # 迭代次数
    max_iterations: int  # 最大迭代次数
    next_action: Optional[Literal["retrieve", "generate", "web_search", "reclassify_intent", "finish"]]  # 下一步行动

    # Web Search (Corrective RAG) - 2025 最佳实践
    web_search_used: bool  # 是否使用了 Web 搜索
    web_search_results: List[Document]  # Web 搜索结果
    web_search_count: int  # Web 搜索次数

    # 元数据
    error_message: str  # 错误信息
    tools_used: List[str]  # 已使用的工具
