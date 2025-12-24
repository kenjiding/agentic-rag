"""检索失败分析器模块

该模块负责分析检索失败的原因，为后续的策略调整提供依据。

失败类型：
1. NO_RESULTS - 无结果：检索未返回任何文档
2. LOW_RELEVANCE - 低相关性：返回的文档与查询相关性低
3. INCOMPLETE - 不完整：只覆盖了部分信息需求
4. REDUNDANT - 冗余：多轮检索返回重复内容
5. SHALLOW - 浅层：信息太表面，缺乏深度
6. MISALIGNED - 偏离：检索结果与用户意图偏离

2025 最佳实践：
- 多维度失败原因分析
- 基于失败类型的针对性改进建议
- 支持失败历史追踪
"""

from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agentic_rag.threshold_config import ThresholdConfig


class FailureType(str, Enum):
    """检索失败类型枚举"""
    NO_RESULTS = "no_results"           # 无结果
    LOW_RELEVANCE = "low_relevance"     # 低相关性
    INCOMPLETE = "incomplete"           # 信息不完整
    REDUNDANT = "redundant"             # 内容冗余
    SHALLOW = "shallow"                 # 信息浅层
    MISALIGNED = "misaligned"           # 意图偏离
    UNKNOWN = "unknown"                 # 未知原因


class FailureAnalysisOutput(BaseModel):
    """LLM 分析输出结构

    注意：OpenAI structured output 要求所有字段都是必填的
    """

    failure_types: List[str] = Field(
        ...,
        description="检测到的失败类型列表"
    )

    missing_aspects: List[str] = Field(
        ...,
        description="缺失的信息方面（如果没有则为空列表）"
    )

    suggested_refinements: List[str] = Field(
        ...,
        description="建议的查询改进方向（如果没有则为空列表）"
    )

    alternative_angles: List[str] = Field(
        ...,
        description="可尝试的替代查询角度（如果没有则为空列表）"
    )

    needs_intent_reclassification: bool = Field(
        ...,
        description="是否需要重新进行意图识别"
    )

    reasoning: str = Field(
        ...,
        description="分析推理过程"
    )


@dataclass
class FailureAnalysisResult:
    """失败分析结果"""

    # 检测到的失败类型（可能有多个）
    failure_types: List[FailureType] = field(default_factory=list)

    # 主要失败类型
    primary_failure: FailureType = FailureType.UNKNOWN

    # 严重程度 (0-1, 1最严重)
    severity: float = 0.5

    # 缺失的信息方面
    missing_aspects: List[str] = field(default_factory=list)

    # 建议的改进方向
    suggested_refinements: List[str] = field(default_factory=list)

    # 替代查询角度
    alternative_angles: List[str] = field(default_factory=list)

    # 是否需要重新进行意图识别
    needs_intent_reclassification: bool = False

    # 当前轮次
    iteration: int = 0

    # 分析推理
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "failure_types": [ft.value for ft in self.failure_types],
            "primary_failure": self.primary_failure.value,
            "severity": self.severity,
            "missing_aspects": self.missing_aspects,
            "suggested_refinements": self.suggested_refinements,
            "alternative_angles": self.alternative_angles,
            "needs_intent_reclassification": self.needs_intent_reclassification,
            "iteration": self.iteration,
            "reasoning": self.reasoning
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureAnalysisResult":
        """从字典创建实例"""
        return cls(
            failure_types=[FailureType(ft) for ft in data.get("failure_types", [])],
            primary_failure=FailureType(data.get("primary_failure", "unknown")),
            severity=data.get("severity", 0.5),
            missing_aspects=data.get("missing_aspects", []),
            suggested_refinements=data.get("suggested_refinements", []),
            alternative_angles=data.get("alternative_angles", []),
            needs_intent_reclassification=data.get("needs_intent_reclassification", False),
            iteration=data.get("iteration", 0),
            reasoning=data.get("reasoning", "")
        )


class RetrievalFailureAnalyzer:
    """检索失败分析器

    分析检索失败的原因，提供针对性的改进建议。
    支持基于规则的快速分析和基于 LLM 的深度分析。
    """

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        threshold_config: Optional[ThresholdConfig] = None,
        enable_llm_analysis: bool = True,
        embeddings = None
    ):
        """
        初始化失败分析器

        Args:
            llm: LLM 实例（用于深度分析）
            threshold_config: 阈值配置
            enable_llm_analysis: 是否启用 LLM 深度分析
            embeddings: Embedding模型（用于语义相似度检测，企业级最佳实践）
        """
        self.threshold_config = threshold_config or ThresholdConfig.default()
        self.enable_llm_analysis = enable_llm_analysis
        self.embeddings = embeddings  # 用于语义相似度检测（企业级最佳实践）

        # 初始化LLM
        if llm is None and enable_llm_analysis:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        self.llm = llm

    def analyze(
        self,
        query: str,
        retrieved_docs: List[Document],
        retrieval_quality: float,
        query_intent: Optional[Dict[str, Any]] = None,
        retrieval_history: Optional[List[List[Document]]] = None,
        iteration: int = 0,
        previous_analysis: Optional[FailureAnalysisResult] = None
    ) -> FailureAnalysisResult:
        """
        分析检索失败原因

        Args:
            query: 原始查询
            retrieved_docs: 当前检索到的文档
            retrieval_quality: 检索质量评分
            query_intent: 查询意图（如果有）
            retrieval_history: 检索历史
            iteration: 当前迭代轮次
            previous_analysis: 上一轮的分析结果

        Returns:
            失败分析结果
        """
        # 第一步：基于规则的快速分析
        result = self._rule_based_analysis(
            query=query,
            retrieved_docs=retrieved_docs,
            retrieval_quality=retrieval_quality,
            query_intent=query_intent,
            retrieval_history=retrieval_history,
            iteration=iteration
        )

        # 第二步：如果启用且需要，进行 LLM 深度分析
        if self.enable_llm_analysis and self._needs_deep_analysis(result, iteration):
            result = self._llm_deep_analysis(
                query=query,
                retrieved_docs=retrieved_docs,
                retrieval_quality=retrieval_quality,
                query_intent=query_intent,
                rule_based_result=result,
                previous_analysis=previous_analysis
            )

        result.iteration = iteration
        return result

    def _rule_based_analysis(
        self,
        query: str,
        retrieved_docs: List[Document],
        retrieval_quality: float,
        query_intent: Optional[Dict[str, Any]],
        retrieval_history: Optional[List[List[Document]]],
        iteration: int
    ) -> FailureAnalysisResult:
        """基于规则的快速分析"""

        failure_types: List[FailureType] = []
        missing_aspects: List[str] = []
        suggested_refinements: List[str] = []
        alternative_angles: List[str] = []
        severity = 0.0

        # 获取阈值
        quality_threshold = self.threshold_config.retrieval.quality_threshold

        # 规则1：无结果检测
        if len(retrieved_docs) == 0:
            failure_types.append(FailureType.NO_RESULTS)
            severity = max(severity, 1.0)
            suggested_refinements.append("尝试使用更通用的关键词")
            suggested_refinements.append("考虑扩展查询范围")
            alternative_angles.append("使用同义词或相关概念")

        # 规则2：低相关性检测
        elif retrieval_quality < quality_threshold:
            failure_types.append(FailureType.LOW_RELEVANCE)
            severity = max(severity, 1.0 - retrieval_quality)

            if retrieval_quality < 0.3:
                suggested_refinements.append("查询可能与知识库内容不匹配")
                suggested_refinements.append("考虑使用更专业的术语")
            else:
                suggested_refinements.append("尝试更具体的表述")
                suggested_refinements.append("添加限定条件以提高精确度")

        # 规则3：冗余检测（与历史结果重复）
        if retrieval_history and len(retrieval_history) > 0:
            current_contents = self._get_doc_contents(retrieved_docs)
            for prev_docs in retrieval_history:
                prev_contents = self._get_doc_contents(prev_docs)
                overlap = self._calculate_overlap(current_contents, prev_contents)

                if overlap > 0.7:  # 70% 以上重复
                    failure_types.append(FailureType.REDUNDANT)
                    severity = max(severity, overlap)
                    suggested_refinements.append("需要从完全不同的角度查询")
                    alternative_angles.append("尝试查询相关但不同的主题")
                    break

        # 规则4：信息不完整检测（基于意图）
        if query_intent:
            sub_queries = query_intent.get("sub_queries", [])
            if sub_queries and len(sub_queries) > 1:
                # 检查是否覆盖了所有子查询的需求
                covered_aspects = self._estimate_coverage(retrieved_docs, sub_queries)
                if covered_aspects < len(sub_queries):
                    failure_types.append(FailureType.INCOMPLETE)
                    severity = max(severity, 0.5)

                    # 找出缺失的方面
                    for sq in sub_queries:
                        sq_query = sq.get("query", "") if isinstance(sq, dict) else str(sq)
                        if not self._is_aspect_covered(retrieved_docs, sq_query):
                            missing_aspects.append(sq_query)

        # 规则5：多轮失败 -> 建议重新识别意图
        needs_reclassification = False
        if iteration >= 2 and FailureType.LOW_RELEVANCE in failure_types:
            needs_reclassification = True
            suggested_refinements.append("初始意图识别可能不准确，建议重新分析")

        # 确定主要失败类型
        primary_failure = FailureType.UNKNOWN
        if failure_types:
            # 优先级：NO_RESULTS > REDUNDANT > LOW_RELEVANCE > INCOMPLETE
            priority_order = [
                FailureType.NO_RESULTS,
                FailureType.REDUNDANT,
                FailureType.LOW_RELEVANCE,
                FailureType.INCOMPLETE,
                FailureType.SHALLOW,
                FailureType.MISALIGNED
            ]
            for ft in priority_order:
                if ft in failure_types:
                    primary_failure = ft
                    break

        return FailureAnalysisResult(
            failure_types=failure_types,
            primary_failure=primary_failure,
            severity=severity,
            missing_aspects=missing_aspects,
            suggested_refinements=suggested_refinements,
            alternative_angles=alternative_angles,
            needs_intent_reclassification=needs_reclassification,
            reasoning="基于规则的快速分析"
        )

    def _needs_deep_analysis(
        self,
        rule_based_result: FailureAnalysisResult,
        iteration: int
    ) -> bool:
        """判断是否需要 LLM 深度分析"""

        # 条件1：规则分析未能确定失败原因
        if not rule_based_result.failure_types:
            return True

        # 条件2：多轮检索仍失败
        if iteration >= 2:
            return True

        # 条件3：严重程度高
        if rule_based_result.severity >= 0.8:
            return True

        # 条件4：检测到冗余
        if FailureType.REDUNDANT in rule_based_result.failure_types:
            return True

        return False

    def _llm_deep_analysis(
        self,
        query: str,
        retrieved_docs: List[Document],
        retrieval_quality: float,
        query_intent: Optional[Dict[str, Any]],
        rule_based_result: FailureAnalysisResult,
        previous_analysis: Optional[FailureAnalysisResult]
    ) -> FailureAnalysisResult:
        """使用 LLM 进行深度分析"""

        if not self.llm:
            return rule_based_result

        # 准备文档摘要
        doc_summaries = []
        for i, doc in enumerate(retrieved_docs[:5]):  # 最多取5个文档
            content = doc.page_content[:300]  # 每个文档最多300字
            doc_summaries.append(f"文档{i+1}: {content}...")

        docs_text = "\n".join(doc_summaries) if doc_summaries else "（无检索结果）"

        # 准备意图信息
        intent_info = ""
        if query_intent:
            intent_info = f"""
意图类型: {query_intent.get('intent_type', '未知')}
复杂度: {query_intent.get('complexity', '未知')}
子查询: {[sq.get('query', sq) if isinstance(sq, dict) else str(sq) for sq in query_intent.get('sub_queries', [])]}
"""

        # 准备历史分析信息
        history_info = ""
        if previous_analysis:
            history_info = f"""
上一轮分析:
- 失败类型: {[ft.value for ft in previous_analysis.failure_types]}
- 建议: {previous_analysis.suggested_refinements}
"""

        template = """你是一个专业的检索系统分析专家。请深入分析以下检索失败的原因。

# 原始查询
{query}

# 意图识别结果
{intent_info}

# 检索结果
质量评分: {quality}
{docs_text}

# 规则分析结果
检测到的问题: {rule_failures}
{history_info}

# 分析要求

请分析检索失败的深层原因，并提供改进建议。

## 失败类型（选择所有适用的）
- no_results: 无结果
- low_relevance: 相关性低
- incomplete: 信息不完整
- redundant: 内容冗余（与历史结果重复）
- shallow: 信息太浅层
- misaligned: 与用户意图偏离

## 需要回答的问题
1. 检测到哪些失败类型？
2. 缺失了哪些信息方面？
3. 建议如何改进查询？
4. 有哪些替代的查询角度？
5. 是否需要重新进行意图识别？

请使用与查询相同的语言回答。"""

        prompt = ChatPromptTemplate.from_template(template)

        try:
            # 使用 function_calling 方法以获得更好的兼容性
            structured_llm = self.llm.with_structured_output(
                FailureAnalysisOutput,
                method="function_calling"
            )
            chain = prompt | structured_llm

            output: FailureAnalysisOutput = chain.invoke({
                "query": query,
                "intent_info": intent_info or "（无意图识别结果）",
                "quality": f"{retrieval_quality:.2f}",
                "docs_text": docs_text,
                "rule_failures": [ft.value for ft in rule_based_result.failure_types],
                "history_info": history_info
            })

            # 转换失败类型
            failure_types = []
            for ft_str in output.failure_types:
                try:
                    failure_types.append(FailureType(ft_str))
                except ValueError:
                    pass

            # 合并规则分析和 LLM 分析的结果
            combined_failure_types = list(set(
                rule_based_result.failure_types + failure_types
            ))

            # 确定主要失败类型
            primary_failure = FailureType.UNKNOWN
            if combined_failure_types:
                priority_order = [
                    FailureType.NO_RESULTS,
                    FailureType.REDUNDANT,
                    FailureType.MISALIGNED,
                    FailureType.LOW_RELEVANCE,
                    FailureType.INCOMPLETE,
                    FailureType.SHALLOW
                ]
                for ft in priority_order:
                    if ft in combined_failure_types:
                        primary_failure = ft
                        break

            return FailureAnalysisResult(
                failure_types=combined_failure_types,
                primary_failure=primary_failure,
                severity=rule_based_result.severity,
                missing_aspects=output.missing_aspects or rule_based_result.missing_aspects,
                suggested_refinements=output.suggested_refinements or rule_based_result.suggested_refinements,
                alternative_angles=output.alternative_angles or rule_based_result.alternative_angles,
                needs_intent_reclassification=output.needs_intent_reclassification,
                reasoning=output.reasoning
            )

        except Exception as e:
            print(f"[失败分析器] LLM 分析错误: {e}")
            return rule_based_result

    def _get_doc_contents(self, docs: List[Document]) -> Set[str]:
        """提取文档内容集合（用于去重比较）"""
        contents = set()
        for doc in docs:
            # 使用内容的前200个字符作为标识
            content = doc.page_content[:200].strip()
            if content:
                contents.add(content)
        return contents

    def _calculate_overlap(self, set1: Set[str], set2: Set[str]) -> float:
        """计算两个集合的重叠度"""
        if not set1 or not set2:
            return 0.0

        # 基于内容相似度计算重叠
        overlap_count = 0
        for s1 in set1:
            for s2 in set2:
                # 简单的 Jaccard 相似度
                words1 = set(s1.split())
                words2 = set(s2.split())
                if words1 and words2:
                    jaccard = len(words1 & words2) / len(words1 | words2)
                    if jaccard > 0.5:  # 相似度阈值
                        overlap_count += 1
                        break

        return overlap_count / len(set1) if set1 else 0.0

    def _estimate_coverage(
        self,
        docs: List[Document],
        sub_queries: List[Any]
    ) -> int:
        """估算文档覆盖了多少子查询"""
        if not docs or not sub_queries:
            return 0

        covered = 0
        doc_text = " ".join([d.page_content for d in docs]).lower()

        for sq in sub_queries:
            sq_query = sq.get("query", "") if isinstance(sq, dict) else str(sq)
            # 企业级最佳实践：使用语义相似度检测，而不是简单的关键词匹配
            if self._is_query_covered_by_docs(docs, sq_query):
                covered += 1

        return covered

    def _is_aspect_covered(
        self,
        docs: List[Document],
        aspect_query: str
    ) -> bool:
        """
        检查某个方面是否被文档覆盖（企业级最佳实践）
        
        使用语义相似度检测，而不是简单的关键词匹配。
        
        Args:
            docs: 文档列表
            aspect_query: 方面查询
            
        Returns:
            如果文档覆盖了该方面，返回True
        """
        if not docs or not aspect_query:
            return False

        return self._is_query_covered_by_docs(docs, aspect_query)
    
    def _is_query_covered_by_docs(
        self,
        docs: List[Document],
        query: str,
        similarity_threshold: float = 0.3
    ) -> bool:
        """
        检查查询是否被文档覆盖（企业级最佳实践）
        
        企业级最佳实践：使用embedding语义相似度检测，而不是关键词匹配。
        支持多语言，理解语义而非字面匹配。
        
        Args:
            docs: 文档列表
            query: 查询文本
            similarity_threshold: 语义相似度阈值（默认0.3）
            
        Returns:
            如果文档覆盖了查询，返回True
        """
        if not docs or not query:
            return False

        # 企业级最佳实践：使用统一的语义相似度工具
        if self.embeddings:
            try:
                from src.agentic_rag.utils.semantic_similarity import SemanticSimilarityCalculator
                
                calculator = SemanticSimilarityCalculator(self.embeddings)
                max_similarity = calculator.calculate_max_similarity(
                    query=query,
                    documents=docs,
                    max_doc_length=1000
                )
                
                # 如果最高相似度超过阈值，认为文档覆盖了查询
                return max_similarity > similarity_threshold
                
            except Exception as e:
                # 如果embedding计算失败，回退到LLM判断
                if self.llm:
                    return self._llm_check_coverage(docs, query)
                # 如果LLM也不可用，使用简单的长度检查作为最后回退
                return len(" ".join([d.page_content for d in docs])) > len(query) * 2
        
        # 如果没有embedding，使用LLM判断
        if self.llm:
            return self._llm_check_coverage(docs, query)
        
        # 最后回退：简单的长度检查（非常不可靠，应该避免）
        return len(" ".join([d.page_content for d in docs])) > len(query) * 2
    
    def _llm_check_coverage(
        self,
        docs: List[Document],
        query: str
    ) -> bool:
        """
        使用LLM检查文档是否覆盖查询（企业级最佳实践）
        
        使用统一的LLM判断工具，避免代码重复。
        
        Args:
            docs: 文档列表
            query: 查询文本
            
        Returns:
            如果文档覆盖了查询，返回True
        """
        if not self.llm:
            return False
        
        try:
            from src.agentic_rag.utils.llm_judge import LLMJudge
            
            judge = LLMJudge(self.llm)
            return judge.check_coverage(
                query=query,
                documents=docs,
                max_docs=5,
                max_doc_length=500,
                confidence_threshold=0.5
            )
            
        except Exception as e:
            # LLM判断失败，返回False（保守策略）
            return False
    
