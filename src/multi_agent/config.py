"""多Agent系统配置管理

本模块提供配置管理功能，支持从环境变量和配置文件加载配置。

2025最佳实践：
- 将硬编码值提取到配置中
- 支持通过配置扩展关键词和模式
- 保持向后兼容
"""
from typing import Optional, List, Dict
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# 关键词配置 - 用于意图检测和任务编排
# ============================================================

@dataclass
class KeywordsConfig:
    """关键词配置 - 可扩展的关键词列表

    支持多语言和自定义扩展。
    所有关键词列表都可以通过配置进行扩展。
    """

    # 购买/下单意图关键词
    order_intent_keywords: List[str] = field(default_factory=lambda: [
        # 中文
        "下单", "购买", "买", "订购", "我要", "帮我",
        # 英文
        "order", "buy", "purchase",
    ])

    # 商品相关关键词（通用类型）
    product_type_keywords: List[str] = field(default_factory=lambda: [
        # 中文通用
        "商品", "产品",
        # 电子产品
        "手机", "电脑", "笔记本", "平板", "电视", "相机",
        # 家电
        "家电", "冰箱", "洗衣机", "空调", "微波炉",
        # 英文
        "product", "item", "phone", "computer", "laptop", "tablet",
    ])

    # 品牌名称（可选配置，用于快速路径检测）
    brand_keywords: List[str] = field(default_factory=lambda: [
        # 电子产品品牌
        "华为", "苹果", "小米", "三星", "OPPO", "vivo", "联想", "戴尔", "惠普",
        # 家电品牌
        "西门子", "海尔", "格力", "美的", "松下", "索尼",
        # 英文品牌
        "Apple", "Samsung", "Huawei", "Xiaomi", "Dell", "HP", "Lenovo",
    ])

    # 确认关键词（用于确认机制）
    confirm_yes_keywords: List[str] = field(default_factory=lambda: [
        # 中文
        "确认", "是", "好的", "可以", "同意", "下单", "执行", "继续", "对",
        # 英文
        "yes", "confirm", "ok", "sure", "agree", "proceed",
    ])

    # 取消关键词
    confirm_no_keywords: List[str] = field(default_factory=lambda: [
        # 中文
        "不", "否", "取消", "不要", "算了", "不用",
        # 英文
        "no", "cancel", "abort", "stop", "never mind",
    ])

    # 选择相关关键词
    selection_keywords: List[str] = field(default_factory=lambda: [
        "选择", "确认", "第", "选",
        "1.", "2.", "3.", "4.", "5.",
        "1、", "2、", "3、", "4、", "5、",
        "select", "choose", "pick",
    ])

    # 取消选择的模式（正则表达式）
    cancel_selection_patterns: List[str] = field(default_factory=lambda: [
        r'^取消$',
        r'^取消选择',
        r'^不选了',
        r'^不要了',
        r'^算了$',
        r'^cancel$',
        r'^nevermind$',
    ])

    def get_all_product_keywords(self) -> List[str]:
        """获取所有商品相关关键词（类型 + 品牌）"""
        return self.product_type_keywords + self.brand_keywords

    def extend_order_keywords(self, keywords: List[str]) -> None:
        """扩展购买意图关键词"""
        self.order_intent_keywords.extend(keywords)

    def extend_product_keywords(self, keywords: List[str]) -> None:
        """扩展商品类型关键词"""
        self.product_type_keywords.extend(keywords)

    def extend_brand_keywords(self, keywords: List[str]) -> None:
        """扩展品牌关键词"""
        self.brand_keywords.extend(keywords)


# ============================================================
# LLM Prompt 模板配置
# ============================================================

@dataclass
class PromptTemplateConfig:
    """Prompt 模板配置 - 支持自定义场景描述"""

    # 任务类型描述（用于 LLM 检测）
    task_type_descriptions: Dict[str, str] = field(default_factory=lambda: {
        "order_with_search": "用户想要购买商品，但还没有选择具体的商品（需要先搜索）",
    })

    # 场景名称（用于 Prompt）
    scenario_name: str = "电商订单"

    # 场景描述（用于 Prompt）
    scenario_description: str = """在这个场景中，多步骤任务主要指：
- **order_with_search**: 用户想要购买商品，但还没有选择具体的商品
  - 特征：包含购买意图（下单/购买/买）+ 商品描述/关键词/品牌名 + 没有具体的 product_id"""


# ============================================================
# 全局配置实例
# ============================================================

# 关键词配置单例
_keywords_config: Optional[KeywordsConfig] = None

def get_keywords_config() -> KeywordsConfig:
    """获取关键词配置单例"""
    global _keywords_config
    if _keywords_config is None:
        _keywords_config = KeywordsConfig()
    return _keywords_config

def reset_keywords_config() -> None:
    """重置关键词配置（用于测试）"""
    global _keywords_config
    _keywords_config = None

# Prompt 模板配置单例
_prompt_config: Optional[PromptTemplateConfig] = None

def get_prompt_config() -> PromptTemplateConfig:
    """获取 Prompt 模板配置单例"""
    global _prompt_config
    if _prompt_config is None:
        _prompt_config = PromptTemplateConfig()
    return _prompt_config


# ============================================================
# 系统配置
# ============================================================

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

