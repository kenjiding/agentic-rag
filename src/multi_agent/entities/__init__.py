"""实体管理模块

提供企业级的实体Schema和管理器。

使用方式：
    from src.multi_agent.entities import (
        ProductAgentEntities,
        OrderAgentEntities,
        EntityManager
    )

    # 创建实体模型
    product_entities = ProductAgentEntities(
        product_id=1,
        quantity=2
    )

    # 合并实体
    merged = EntityManager.merge_entities(
        product_entities,
        order_entities
    )

    # 提取Agent特定实体
    product_entities = EntityManager.extract_product_entities(state.entities)
"""

from src.multi_agent.entities.base import (
    # Agent类型
    AgentType,
    # 基础模型
    BaseAgentEntities,
    # Agent实体模型
    ProductAgentEntities,
    OrderAgentEntities,
    ConsultationAgentEntities,
    RAGAgentEntities,
    ChatAgentEntities,
    # 辅助函数
    get_entity_model_for_agent,
)

from src.multi_agent.entities.manager import EntityManager

__all__ = [
    # Agent类型
    "AgentType",

    # 基础模型
    "BaseAgentEntities",

    # Agent实体模型
    "ProductAgentEntities",
    "OrderAgentEntities",
    "ConsultationAgentEntities",
    "RAGAgentEntities",
    "ChatAgentEntities",

    # 实体管理器
    "EntityManager",

    # 辅助函数
    "get_entity_model_for_agent",
]
