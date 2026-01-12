"""统一的 LLM 封装类 - 2025-2026 企业级最佳实践

支持多种大模型，提供统一的接口，便于切换和扩展。
"""
from typing import Generator, Optional, Any, Union
from langchain_core.language_models import BaseChatModel
import os
import logging

logger = logging.getLogger(__name__)

# Optional dependencies - 按需导入，避免强制依赖
try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    ChatOpenAI = None  # type: ignore

try:
    from langchain_ollama import ChatOllama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    ChatOllama = None  # type: ignore

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    ChatGoogleGenerativeAI = None  # type: ignore

try:
    from langchain_anthropic import ChatAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    ChatAnthropic = None  # type: ignore

try:
    from langchain_community.chat_models import ChatTongyi
    HAS_TONGYI = True
except ImportError:
    HAS_TONGYI = False
    ChatTongyi = None  # type: ignore

try:
    from langchain_openai import ChatOpenAI as ChatMoonshot
    HAS_MOONSHOT = True
except ImportError:
    HAS_MOONSHOT = False
    ChatMoonshot = None  # type: ignore


class LLM:
    """统一的 LLM 封装类
    
    支持多种大模型提供商，通过统一的接口访问。
    模型名称格式: provider:model_name
    
    支持的提供商:
    - openai: OpenAI 模型 (gpt-4o-mini, gpt-4o, gpt-4-turbo 等)
    - anthropic: Claude 模型 (claude-3-5-sonnet-20241022, claude-3-opus 等)
    - gemini: Google Gemini 模型 (gemini-2.5-flash, gemini-pro 等)
    - ollama: Ollama 本地模型 (qwen3:8b, llama2 等)
    - tongyi: 阿里通义千问 (qwen-turbo, qwen-plus 等)
    - moonshot: Moonshot AI (moonshot-v1-8k, moonshot-v1-32k 等)
    
    示例:
        llm = LLM(model_name="openai:gpt-4o-mini", temperature=0.1)
        llm_instance = llm.get_llm()
        
        # 使用环境变量配置
        llm = LLM.from_env()
    """
    
    def __init__(
        self, 
        model_name: str = "openai:gpt-4o-mini", 
        temperature: float = 0.1,
        **kwargs
    ) -> None:
        """
        初始化 LLM
        
        Args:
            model_name: 模型名称，格式为 provider:model_name
            temperature: 温度参数，控制随机性
            **kwargs: 其他模型特定参数
        """
        if ":" not in model_name:
            # 兼容旧格式，默认为 OpenAI
            logger.warning(f"模型名称格式已更新，建议使用 'openai:{model_name}' 格式")
            model_name = f"openai:{model_name}"
        
        parts = model_name.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"模型名称格式错误，应为 'provider:model_name'，收到: {model_name}")
        
        self.provider = parts[0].lower()
        self.model_name = parts[1]
        self.temperature = temperature
        self.kwargs = kwargs
        
        # 根据提供商初始化对应的 LLM
        self.llm = self._init_llm()
    
    def _init_llm(self) -> BaseChatModel:
        """根据提供商初始化 LLM"""
        if self.provider == "openai":
            return self._init_openai()
        elif self.provider == "anthropic":
            return self._init_anthropic()
        elif self.provider == "gemini":
            return self._init_gemini()
        elif self.provider == "ollama":
            return self._init_ollama()
        elif self.provider == "tongyi":
            return self._init_tongyi()
        elif self.provider == "moonshot":
            return self._init_moonshot()
        else:
            raise ValueError(
                f"不支持的模型提供商: {self.provider}。"
                f"支持的提供商: openai, anthropic, gemini, ollama, tongyi, moonshot"
            )
    
    def _init_openai(self) -> BaseChatModel:
        """初始化 OpenAI 模型"""
        if not HAS_OPENAI or ChatOpenAI is None:
            raise ImportError(
                "langchain-openai 未安装。请运行: pip install langchain-openai"
            )
        
        api_key = self.kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = self.kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")
        
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=api_key,
            base_url=base_url,
            **{k: v for k, v in self.kwargs.items() if k not in ["api_key", "base_url"]}
        )
    
    def _init_anthropic(self) -> BaseChatModel:
        """初始化 Anthropic Claude 模型"""
        if not HAS_ANTHROPIC or ChatAnthropic is None:
            raise ImportError(
                "langchain-anthropic 未安装。请运行: pip install langchain-anthropic"
            )
        
        api_key = self.kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        
        return ChatAnthropic(
            model=self.model_name,
            temperature=self.temperature,
            api_key=api_key,
            **{k: v for k, v in self.kwargs.items() if k != "api_key"}
        )
    
    def _init_gemini(self) -> BaseChatModel:
        """初始化 Google Gemini 模型"""
        if not HAS_GEMINI or ChatGoogleGenerativeAI is None:
            raise ImportError(
                "langchain-google-genai 未安装。请运行: pip install langchain-google-genai"
            )
        
        api_key = self.kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
        
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=api_key,
            **{k: v for k, v in self.kwargs.items() if k != "api_key"}
        )
    
    def _init_ollama(self) -> BaseChatModel:
        """初始化 Ollama 本地模型"""
        if not HAS_OLLAMA or ChatOllama is None:
            raise ImportError(
                "langchain-ollama 未安装。请运行: pip install langchain-ollama"
            )
        
        base_url = self.kwargs.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        return ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            base_url=base_url,
            **{k: v for k, v in self.kwargs.items() if k not in ["base_url"]}
        )
    
    def _init_tongyi(self) -> BaseChatModel:
        """初始化阿里通义千问模型（使用 OpenAI 兼容接口）
        
        最佳实践：使用 OpenAI 兼容接口，统一接口规范，便于维护和扩展。
        """
        if not HAS_OPENAI or ChatOpenAI is None:
            raise ImportError(
                "langchain-openai 未安装。请运行: pip install langchain-openai"
            )
        
        # 获取 API Key（支持多种环境变量名称）
        api_key = (
            self.kwargs.get("api_key") 
            or os.getenv("TONGYI_API_KEY") 
            or os.getenv("DASHSCOPE_API_KEY")
        )
        
        # 获取 Base URL（默认使用兼容模式）
        base_url = (
            self.kwargs.get("base_url") 
            or os.getenv("TONGYI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        )
        
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=api_key,
            base_url=base_url,
            **{k: v for k, v in self.kwargs.items() if k not in ["api_key", "base_url"]}
        )
    
    def _init_moonshot(self) -> BaseChatModel:
        """初始化 Moonshot AI 模型（使用 OpenAI 兼容接口）"""
        if not HAS_MOONSHOT or ChatMoonshot is None:
            raise ImportError(
                "langchain-openai 未安装。请运行: pip install langchain-openai"
            )
        
        api_key = self.kwargs.get("api_key") or os.getenv("MOONSHOT_API_KEY")
        base_url = self.kwargs.get("base_url") or os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")
        
        return ChatMoonshot(
            model=self.model_name,
            temperature=self.temperature,
            api_key=api_key,
            base_url=base_url,
            **{k: v for k, v in self.kwargs.items() if k not in ["api_key", "base_url"]}
        )
    
    def generate(self, prompt: str) -> str:
        """生成文本（同步）"""
        return self.llm.invoke(prompt)
    
    def stream(self, prompt: str) -> Generator[str, None, None]:
        """流式生成文本"""
        return self.llm.stream(prompt)
    
    def get_llm(self) -> BaseChatModel:
        """获取 LangChain LLM 实例"""
        return self.llm
    
    @classmethod
    def from_env(cls, env_key: str = "LLM_MODEL_NAME", default: str = "openai:gpt-4o-mini") -> "LLM":
        """从环境变量创建 LLM 实例
        
        Args:
            env_key: 环境变量键名，默认为 LLM_MODEL_NAME
            default: 默认模型名称
            
        Returns:
            LLM 实例
        """
        model_name = os.getenv(env_key, default)
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        return cls(model_name=model_name, temperature=temperature)
    
    @classmethod
    def create_default(cls) -> "LLM":
        """创建默认 LLM 实例（用于向后兼容）"""
        return cls.from_env()
