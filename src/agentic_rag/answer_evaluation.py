from pydantic import BaseModel, Field
from typing import Literal

class AnswerEvaluation(BaseModel):
    """RAG答案质量评估结果（Answer quality evaluation result for RAG）"""
    
    relevance: float = Field(
        ...,
        description=(
            "相关性分数（Relevance score）：评估答案是否直接、精准地回答了用户问题。"
            "1.0：答案完全针对问题核心，无无关内容；"
            "0.8-0.9：主要内容相关，但有轻微偏题或多余信息；"
            "0.5-0.7：部分相关，但遗漏主要点或包含较多无关内容；"
            "0.0-0.4：几乎或完全未回答问题。"
        ),
        ge=0.0,
        le=1.0
    )
    
    accuracy: float = Field(
        ...,
        description=(
            "准确性分数（Accuracy score）：评估答案中的事实陈述是否与提供的上下文一致，是否存在幻觉（hallucination）。"
            "1.0：所有信息均来自上下文且准确无误，无任何编造或矛盾；"
            "0.8-0.9：绝大部分准确，仅有极小不影响理解的偏差；"
            "0.5-0.7：存在明显事实错误或未忠实于上下文；"
            "0.0-0.4：大量幻觉或严重扭曲上下文。"
        ),
        ge=0.0,
        le=1.0
    )
    
    completeness: float = Field(
        ...,
        description=(
            "完整性分数（Completeness score）：评估答案是否覆盖了问题所需的所有关键信息点，而不遗漏重要内容。"
            "1.0：全面覆盖上下文中的所有相关要点，无遗漏；"
            "0.8-0.9：覆盖主要信息，遗漏次要但不影响整体理解的部分；"
            "0.5-0.7：遗漏多个重要信息点，导致答案不完整；"
            "0.0-0.4：严重遗漏，大部分关键信息未提及。"
        ),
        ge=0.0,
        le=1.0
    )
    
    clarity: float = Field(
        ...,
        description=(
            "清晰性分数（Clarity score）：评估答案的表达是否逻辑清晰、结构合理、易于理解。"
            "1.0：语言流畅、逻辑严谨、使用适当的段落/编号/强调，便于阅读；"
            "0.8-0.9：整体清晰，偶有表述冗长或轻微混乱；"
            "0.5-0.7：表达混乱、跳跃或重复较多，影响理解；"
            "0.0-0.4：极度混乱、语法错误严重或完全无结构。"
        ),
        ge=0.0,
        le=1.0
    )
    
    score: float = Field(
        ...,
        description=(
            "总体质量分数（Overall quality score）：四个维度（relevance, accuracy, completeness, clarity）的算术平均值。"
            "请直接计算并填写平均分，不要人为调整。"
        ),
        ge=0.0,
        le=1.0
    )
    
    feedback: str = Field(
        ...,
        description=(
            "简要评估反馈（Brief evaluation feedback）：用中文写100字以内，"
            "先指出主要优点，再指出不足及改进建议。例如："
            "'优点：高度相关且事实准确；不足：遗漏了X点，表达可更简洁。'"
        )
    )