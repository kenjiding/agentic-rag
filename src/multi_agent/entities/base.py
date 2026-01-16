"""企业级实体管理架构

设计原则：
1. 每个Agent有自己专门的实体Schema
2. 统一的实体管理器负责合并和转换
3. 类型安全，使用Pydantic确保数据验证
4. 可扩展，新增Agent只需定义自己的实体模型
"""

from typing import Dict, Any, List, Optional, Type, TypeVar
from pydantic import BaseModel, Field
from enum import Enum


# ==================== 类型变量 ====================

T = TypeVar('T', bound='BaseAgentEntities')


# ==================== Agent类型枚举 ====================

class AgentType(str, Enum):
    """Agent类型枚举"""
    PRODUCT = "product_agent"
    ORDER = "order_agent"
    CONSULTATION = "consultation_agent"
    RAG = "rag_agent"
    CHAT = "chat_agent"


# ==================== 基础实体模型 ====================

class BaseAgentEntities(BaseModel):
    """基础实体模型 - 所有Agent实体模型的基类

    提供通用的转换和合并方法。
    """

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于state.entities）"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseAgentEntities':
        """从字典创建实例（从state.entities恢复）"""
        return cls(**data)

    def merge(self, other: 'BaseAgentEntities') -> 'BaseAgentEntities':
        """合并另一个实体模型（默认实现：后者覆盖前者）

        子类可以重写此方法实现自定义合并逻辑。
        """
        merged_dict = self.to_dict()
        merged_dict.update(other.to_dict())
        return self.__class__(**merged_dict)


T = TypeVar('T', bound=BaseAgentEntities)


# ==================== 各Agent的实体模型 ====================

class ProductAgentEntities(BaseAgentEntities):
    """产品Agent的实体模型

    包含产品搜索、选择、对比所需的实体信息。
    """
    # 单个产品ID（用户已选定）
    product_id: Optional[int] = Field(
        default=None,
        description="用户明确指定的单个产品ID",
        ge=1
    )

    # 多个产品ID（用于对比场景）
    product_ids: List[int] = Field(
        default_factory=list,
        description="多个产品ID列表（用于产品对比场景）"
    )

    # 搜索关键词
    search_keyword: Optional[str] = Field(
        default=None,
        description="搜索关键词（品牌名、产品名或型号）"
    )

    # 购买数量
    quantity: Optional[int] = Field(
        default=None,
        description="购买数量",
        ge=1
    )

    # 品牌名
    brand: Optional[str] = Field(
        default=None,
        description="品牌名"
    )

    # 产品类别
    category: Optional[str] = Field(
        default=None,
        description="产品类别"
    )


class OrderAgentEntities(BaseAgentEntities):
    """订单Agent的实体模型

    包含订单查询、创建、取消所需的实体信息。
    """
    # 订单ID
    order_id: Optional[str] = Field(
        default=None,
        description="订单ID（字符串格式，如ORD123456）"
    )

    # 关联的产品ID
    product_id: Optional[int] = Field(
        default=None,
        description="关联的产品ID（创建订单时使用）",
        ge=1
    )

    # 购买数量
    quantity: Optional[int] = Field(
        default=None,
        description="购买数量",
        ge=1
    )

    # 用户确认状态
    user_confirmed: bool = Field(
        default=False,
        description="用户是否已确认订单"
    )

    # 收货地址
    shipping_address: Optional[str] = Field(
        default=None,
        description="收货地址"
    )

    # 联系电话
    phone: Optional[str] = Field(
        default=None,
        description="联系电话"
    )


class ConsultationAgentEntities(BaseAgentEntities):
    """咨询Agent的实体模型

    包含产品对比、参数查询、适配性确认所需的实体信息。
    """
    # 多个产品ID（用于对比）
    product_ids: List[int] = Field(
        default_factory=list,
        description="多个产品ID列表（至少2个）"
    )

    # 对比维度（价格、性能、功能等）
    comparison_aspects: List[str] = Field(
        default_factory=list,
        description="对比维度列表（如：价格、性能、尺寸等）"
    )

    # 查询的参数类型
    parameter_types: List[str] = Field(
        default_factory=list,
        description="查询的参数类型列表（如：尺寸、重量、功率等）"
    )

    # 设备信息（用于适配性查询）
    device_info: Optional[str] = Field(
        default=None,
        description="用户的设备信息（如车型、手机型号等）"
    )


class RAGAgentEntities(BaseAgentEntities):
    """RAG Agent的实体模型

    包含知识检索所需的实体信息。
    """
    # 查询关键词
    query_keywords: List[str] = Field(
        default_factory=list,
        description="查询关键词列表"
    )

    # 文档类型
    doc_types: List[str] = Field(
        default_factory=list,
        description="文档类型列表（如：说明书、技术文档、FAQ等）"
    )

    # 产品ID（用于检索特定产品的文档）
    product_id: Optional[int] = Field(
        default=None,
        description="关联的产品ID",
        ge=1
    )


class ChatAgentEntities(BaseAgentEntities):
    """Chat Agent的实体模型

    用于一般对话，实体较少。
    """
    # 情感倾向
    sentiment: Optional[str] = Field(
        default=None,
        description="用户情感倾向（positive, negative, neutral）"
    )

    # 意图类别（简单分类）
    intent_category: Optional[str] = Field(
        default=None,
        description="意图类别（greeting, thanks, complaint等）"
    )


# ==================== Agent实体模型映射 ====================

AGENT_ENTITY_MODELS: Dict[AgentType, Type[BaseAgentEntities]] = {
    AgentType.PRODUCT: ProductAgentEntities,
    AgentType.ORDER: OrderAgentEntities,
    AgentType.CONSULTATION: ConsultationAgentEntities,
    AgentType.RAG: RAGAgentEntities,
    AgentType.CHAT: ChatAgentEntities,
}


def get_entity_model_for_agent(agent_name: str) -> Type[BaseAgentEntities]:
    """根据Agent名称获取对应的实体模型

    Args:
        agent_name: Agent名称（如"product_agent"）

    Returns:
        对应的实体模型类

    Raises:
        ValueError: 如果Agent名称不匹配任何已知类型
    """
    try:
        agent_type = AgentType(agent_name)
        return AGENT_ENTITY_MODELS[agent_type]
    except ValueError:
        # 如果不是已知Agent，返回基础模型
        return BaseAgentEntities


