"""智能生成器实现"""
from typing import List, Optional
from agentic_rag.answer_evaluation import AnswerEvaluation
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.agentic_rag.threshold_config import ThresholdConfig, GeneratorThresholds


class IntelligentGenerator:
    """智能生成器"""
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        model_name: str = "gpt-4o-mini",
        temperature: Optional[float] = None,
        threshold_config: Optional[ThresholdConfig] = None
    ):
        """
        初始化生成器
        
        Args:
            llm: LLM 实例
            model_name: 模型名称
            temperature: 生成温度（如果为None，使用配置中的默认温度）
            threshold_config: 阈值配置
        """
        self.threshold_config = threshold_config
        
        # 获取生成器阈值配置
        generator_thresholds = threshold_config.generator if threshold_config else GeneratorThresholds()
        
        # 使用配置的温度（如果未提供）
        if temperature is None:
            temperature = generator_thresholds.default_temperature
        
        self.llm = llm or ChatOpenAI(
            model=model_name,
            temperature=temperature
            # 注意：降低 temperature 到 0.1 已经足够提高稳定性
            # seed 参数不是所有 API 都支持，如果需要可自行添加
        )
        self.output_parser = StrOutputParser()
        
    def format_context(self, docs: List[Document]) -> str:
        """
        格式化文档为上下文字符串
        
        注意：保持文档顺序稳定，确保相同输入产生相同输出
        
        Args:
            docs: 文档列表
            
        Returns:
            格式化后的上下文
        """
        if not docs:
            return "没有找到相关文档。"
        
        # 确保文档顺序稳定（按内容哈希排序，避免随机顺序）
        # 注意：这里不改变相似度排序，只是确保相同文档列表产生相同格式
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[文档 {i}]\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def generate(
        self,
        question: str,
        context: str,
        previous_answer: Optional[str] = None,
        feedback: Optional[str] = None,
        query_intent: Optional[dict] = None
    ) -> str:
        """
        生成答案
        
        Args:
            question: 用户问题
            context: 检索到的上下文
            previous_answer: 之前的答案（用于改进）
            feedback: 反馈信息（用于改进）
            query_intent: 查询意图信息（可选）
            
        Returns:
            生成的答案
        """
        if previous_answer and feedback:
            # 改进模式
            return self._improve_answer(question, context, previous_answer, feedback)
        else:
            # 首次生成模式（传递意图信息）
            return self._initial_generate(question, context, query_intent)
    
    def _initial_generate(self, question: str, context: str, query_intent: Optional[dict] = None) -> str:
        """首次生成答案"""
        
        # 根据意图类型调整prompt
        intent_specific_guidance = ""
        if query_intent:
            intent_type = query_intent.get("intent_type", "factual")
            decomposition_type = query_intent.get("decomposition_type")
            # 向后兼容：检查旧的 is_comparison 字段
            is_comparison = query_intent.get("is_comparison", False) or decomposition_type == "comparison"

            if intent_type == "comparison" or is_comparison:
                intent_specific_guidance = """
**特别说明（对比类查询）**：
1. **仔细查找所有相关时间点/对象的信息**：在上下文中找出所有提到的时间点和对应的数据
2. **明确对比**：明确指出每个时间点/对象的具体数值或状态
3. **计算变化**：如果有数值，计算并说明变化方向和幅度（上升/下降，增加/减少）
4. **引用来源**：明确引用每个数据来自哪个文档编号
5. **确保完整性**：必须同时找到所有需要对比的时间点/对象的信息，缺一不可
"""
            elif intent_type == "analytical":
                intent_specific_guidance = """
**特别说明（分析性查询）**：
1. **深入分析**：不仅提供事实，还要进行分析、推理和总结
2. **多角度思考**：从不同角度分析问题
3. **逻辑清晰**：确保分析过程逻辑清晰，结论有据可依
"""
            elif intent_type == "multi_hop":
                intent_specific_guidance = """
**特别说明（多跳查询）**：
1. **分步推理**：将复杂问题分解为多个步骤
2. **逐步验证**：每一步都要基于上下文中的信息
3. **整合结果**：将各步推理结果整合成最终答案
"""
        
        template = f"""你是一个专业的 AI 助手。严格基于以下上下文信息回答问题。

**工作流程**：
1. **仔细阅读所有上下文**：上下文可能包含多个文档，请逐一检查每个文档
2. **识别相关信息**：找出所有与问题相关或可能相关的信息（即使关键词不完全匹配）
3. **整合信息**：将分散在不同文档中的相关信息整合起来
4. **生成答案**：基于找到的信息，生成准确、完整、有条理的答案

{intent_specific_guidance}

**要求**：
1. **必须仔细检查所有文档**：不要遗漏任何可能相关的信息
2. **积极利用上下文**：如果上下文中包含任何与问题相关的信息（即使不完全匹配），都应该回答问题
3. **只使用上下文中的信息**：答案必须严格基于提供的上下文，不要编造信息，不要使用"可能"、"也许"等不确定的表述
4. **引用具体内容**：如果找到相关信息，请引用具体的文档编号和内容来支持你的答案
5. **只有在完全找不到相关信息时才说明无法回答**：只有在仔细检查所有上下文后，确实没有任何相关信息时，才说明"根据提供的上下文，我无法回答这个问题"

**重要提示**：
- 上下文可能包含多个文档，信息可能分散在不同文档中
- 关键词可能不完全匹配，但语义相关的内容也应该被考虑
- 请仔细分析上下文，不要过于保守
- 如果找到相关信息，必须明确回答问题，不要使用"可能"、"也许"等不确定的表述

上下文：
{{context}}

问题：{{question}}

**只有在上下文中存在的信息才应该被使用, 在上下文没有找到就是没有相关信息, 不要编造信息, 不要使用"可能"、"也许"等不确定的表述**

请严格按照上述流程，仔细检查上下文，然后回答问题："""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | self.output_parser
        
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        
        return answer
    
    def _improve_answer(
        self,
        question: str,
        context: str,
        previous_answer: str,
        feedback: str
    ) -> str:
        """改进答案"""
        template = """基于反馈改进之前的答案。

**工作流程**：
1. **分析反馈**：仔细阅读反馈，理解需要改进的地方
2. **检查上下文**：重新检查上下文，找出反馈中提到的遗漏信息
3. **整合信息**：将反馈指出的信息与上下文中的信息整合
4. **生成改进答案**：基于反馈和上下文，生成改进后的答案

原始问题：{question}

上下文：
{context}

之前的答案：
{previous_answer}

反馈：
{feedback}

**改进要求**：
1. **必须根据反馈改进**：反馈指出的问题必须全部解决
2. **充分利用上下文**：反馈中提到的信息必须在上下文中找到并引用
3. **保持正确部分**：原答案中正确的部分必须保留
4. **确保完整性**：改进后的答案必须完整回答所有问题
5. **明确表述**：使用确定的表述，避免"可能"、"也许"等不确定词汇
6. **引用具体内容**：引用具体的文档编号和内容来支持答案

**只有在上下文中存在的信息才应该被使用, 在上下文没有找到就是没有相关信息,不要编造信息, 不要使用"可能"、"也许"等不确定的表述**
请严格按照上述要求，生成改进后的答案："""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | self.output_parser
        
        improved_answer = chain.invoke({
            "question": question,
            "context": context,
            "previous_answer": previous_answer,
            "feedback": feedback
        })
        
        return improved_answer
    
    def evaluate_answer_quality(
        self,
        question: str,
        answer: str,
        context: str,
        threshold: Optional[float] = None
    ) -> tuple[float, bool, str]:
        """
        评估答案质量
        
        Args:
            question: 用户问题
            answer: 生成的答案
            context: 上下文
            threshold: 质量阈值（如果为None，使用配置中的阈值）
            
        Returns:
            (质量分数, 是否满足阈值, 评估反馈)
        """
        # 使用配置的阈值（如果未提供）
        if threshold is None:
            if self.threshold_config:
                threshold = self.threshold_config.generation.answer_quality_threshold
            else:
                threshold = 0.75  # 默认值
        template = """你是一个严格的答案质量评估器。请严格按照以下标准评估答案质量。

**评估原则**：
1. **客观性**：只根据事实评估，不要猜测或主观判断
2. **一致性**：相同质量的答案应该得到相同的分数
3. **明确性**：使用明确的评分标准，避免模糊判断

**评估标准**（每个维度 0-1 分）：

1. **相关性（Relevance）**：
   - 1.0：答案完全针对问题核心，无无关内容
   - 0.8-0.9：主要内容相关，但有轻微偏题或多余信息
   - 0.5-0.7：部分相关，但遗漏主要点或包含较多无关内容
   - 0.0-0.4：几乎或完全未回答问题

2. **准确性（Accuracy）**：
   - 1.0：所有信息均来自上下文且准确无误，无任何编造或矛盾
   - 0.8-0.9：绝大部分准确，仅有极小不影响理解的偏差
   - 0.5-0.7：存在明显事实错误或未忠实于上下文
   - 0.0-0.4：大量幻觉或严重扭曲上下文

3. **完整性（Completeness）**：
   - 1.0：全面覆盖上下文中的所有相关要点，无遗漏
   - 0.8-0.9：覆盖主要信息，遗漏次要但不影响整体理解的部分
   - 0.5-0.7：遗漏多个重要信息点，导致答案不完整
   - 0.0-0.4：严重遗漏，大部分关键信息未提及

4. **清晰性（Clarity）**：
   - 1.0：语言流畅、逻辑严谨、使用适当的段落/编号/强调，便于阅读
   - 0.8-0.9：整体清晰，偶有表述冗长或轻微混乱
   - 0.5-0.7：表达混乱、跳跃或重复较多，影响理解
   - 0.0-0.4：极度混乱、语法错误严重或完全无结构

**总体质量分数**：四个维度的算术平均值（直接计算，不要人为调整）

问题：{question}

上下文：
{context}

答案：
{answer}

请严格按照上述标准评估，确保评估结果客观、一致。"""
        
        prompt = ChatPromptTemplate.from_template(template)
        structured_llm = self.llm.with_structured_output(AnswerEvaluation, method="json_schema")
        chain = prompt | structured_llm
        
        try:
            response: AnswerEvaluation = chain.invoke({
                "question": question,
                "context": context,
                "answer": answer,
            })
            
            # 解析响应
            score = response.score
            feedback = response.feedback
            meets_threshold = score >= threshold
            
            return score, meets_threshold, feedback
            
        except Exception as e:
            print(f"[评估错误] {str(e)}")
            return 0.5, False, f"评估过程出错: {str(e)}"
    
    def generate_feedback(
        self,
        question: str,
        answer: str,
        context: str
    ) -> str:
        """
        生成改进反馈
        
        Args:
            question: 用户问题
            answer: 当前答案
            context: 上下文
            
        Returns:
            改进建议
        """
        template = """分析以下答案，指出可以改进的地方。

**分析要求**：
1. **仔细对比**：对比问题、答案和上下文，找出所有不一致或遗漏的地方
2. **明确指出**：明确指出答案中遗漏的上下文信息（包括具体位置）
3. **提供具体建议**：提供具体的改进建议，包括应该添加哪些信息

问题：{question}

上下文：
{context}

答案：
{answer}

请从以下方面分析并指出改进点：
1. **完整性**：答案是否完整回答了问题？是否遗漏了上下文中的重要信息？
   - 如果遗漏，请明确指出遗漏了哪些信息，这些信息在上下文的哪个位置
2. **准确性**：答案是否准确？是否有不准确或与上下文不符的地方？
   - 如果有，请明确指出错误的地方
3. **改进建议**：如何改进答案？
   - 请提供具体的改进建议，包括应该添加哪些信息，如何修正错误

**改进建议**（请详细说明）："""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | self.output_parser
        
        feedback = chain.invoke({
            "question": question,
            "context": context,
            "answer": answer
        })
        
        return feedback
