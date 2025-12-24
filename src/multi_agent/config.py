"""多Agent系统配置管理

本模块提供配置管理功能，支持从环境变量和配置文件加载配置。
"""
from typing import Optional
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class MultiAgentConfig:
    """多Agent系统配置
    
    Attributes:
        llm_model: 语言模型名称
        llm_temperature: 语言模型温度
        rag_persist_directory: RAG向量数据库持久化目录
        max_iterations: 最大迭代次数
        enable_web_search: 是否启用Web搜索
        log_level: 日志级别
    """
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    rag_persist_directory: str = "./tmp/chroma_db/agentic_rag"
    max_iterations: int = 10
    enable_web_search: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "MultiAgentConfig":
        """
        从环境变量加载配置
        
        Returns:
            配置实例
        """
        return cls(
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            rag_persist_directory=os.getenv("RAG_PERSIST_DIRECTORY", "./tmp/chroma_db/agentic_rag"),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "10")),
            enable_web_search=os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )

