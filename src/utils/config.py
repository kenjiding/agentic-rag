"""配置管理"""
from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

from src.agentic_rag.threshold_config import ThresholdConfig

load_dotenv()


@dataclass
class AgenticRAGConfig:
    """Agentic RAG 配置"""
    # LLM 配置
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    
    # 嵌入配置
    embedding_model: str = "text-embedding-3-small"
    
    # 检索配置（已迁移到 threshold_config，保留用于向后兼容）
    default_retrieval_k: int = 3
    retrieval_quality_threshold: float = 0.7
    
    # 生成配置（已迁移到 threshold_config，保留用于向后兼容）
    answer_quality_threshold: float = 0.7
    
    # 控制流配置
    max_iterations: int = 3
    early_stopping: bool = True
    
    # 向量数据库配置
    persist_directory: Optional[str] = None
    
    # 阈值配置（新增：统一的阈值管理）
    threshold_config: Optional[ThresholdConfig] = None
    
    # 其他配置
    verbose: bool = True
    enable_caching: bool = True
    
    def __post_init__(self):
        """初始化后处理"""
        # 如果没有提供阈值配置，使用默认配置
        if self.threshold_config is None:
            self.threshold_config = ThresholdConfig.default()
        
        # 向后兼容：如果提供了旧配置，同步到阈值配置
        if self.default_retrieval_k != 3:
            self.threshold_config.retrieval.default_k = self.default_retrieval_k
        if self.retrieval_quality_threshold != 0.7:
            self.threshold_config.retrieval.quality_threshold = self.retrieval_quality_threshold
        if self.answer_quality_threshold != 0.7:
            self.threshold_config.generation.answer_quality_threshold = self.answer_quality_threshold
    
    @classmethod
    def from_env(cls) -> "AgenticRAGConfig":
        """从环境变量创建配置"""
        return cls(
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "3")),
            persist_directory=os.getenv("PERSIST_DIRECTORY"),
            verbose=os.getenv("VERBOSE", "true").lower() == "true"
        )
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "embedding_model": self.embedding_model,
            "default_retrieval_k": self.default_retrieval_k,
            "retrieval_quality_threshold": self.retrieval_quality_threshold,
            "answer_quality_threshold": self.answer_quality_threshold,
            "max_iterations": self.max_iterations,
            "early_stopping": self.early_stopping,
            "persist_directory": self.persist_directory,
            "threshold_config": self.threshold_config.to_dict() if self.threshold_config else None,
            "verbose": self.verbose,
            "enable_caching": self.enable_caching
        }
    
    @classmethod
    def strict(cls) -> "AgenticRAGConfig":
        """严格配置：更高的阈值"""
        return cls(
            threshold_config=ThresholdConfig.strict()
        )
    
    @classmethod
    def lenient(cls) -> "AgenticRAGConfig":
        """宽松配置：较低的阈值"""
        return cls(
            threshold_config=ThresholdConfig.lenient()
        )