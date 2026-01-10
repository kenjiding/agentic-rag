"""深度咨询工具

提供产品对比、参数提取、适配性确认等功能。
所有工具都使用LLM进行智能解析，适用于任何产品类别，不硬编码特定场景。
"""

import json
import logging
import asyncio
from typing import Annotated, Optional, List, Dict, Any
from decimal import Decimal

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import os
from src.db.engine import get_db_session
from src.db.crud import get_product_by_id, search_products
from src.db.test_data_loader import is_use_test_data
from src.schema.business_models import ProductDisplay

logger = logging.getLogger(__name__)

# 全局LLM实例（用于工具内部调用）
_default_llm = None


def get_default_llm() -> ChatOpenAI:
    """获取默认LLM实例（工具内部使用）"""
    global _default_llm
    if _default_llm is None:
        _default_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    return _default_llm


def run_async(coro):
    """运行异步函数（兼容同步和异步上下文）
    
    在同步上下文中运行异步函数。如果已经有运行中的事件循环，
    在新线程中创建新的事件循环运行；否则直接在当前线程运行。
    """
    try:
        # 尝试获取当前事件循环
        asyncio.get_running_loop()
        # 如果已经有运行中的事件循环，需要在新线程中运行
        import threading
        from asyncio import new_event_loop, set_event_loop
        
        result = None
        exception = None
        event = threading.Event()
        
        def run_in_new_loop():
            nonlocal result, exception
            try:
                new_loop = new_event_loop()
                set_event_loop(new_loop)
                result = new_loop.run_until_complete(coro)
                new_loop.close()
            except Exception as e:
                exception = e
            finally:
                event.set()
        
        thread = threading.Thread(target=run_in_new_loop, daemon=True)
        thread.start()
        
        # 等待完成，设置超时
        if not event.wait(timeout=60):
            raise TimeoutError("工具调用超时（60秒）")
        
        if exception:
            raise exception
        return result
    except RuntimeError:
        # 如果没有运行中的事件循环，直接运行
        return asyncio.run(coro)


# ============== Pydantic模型定义 ==============

class ProductSpecifications(BaseModel):
    """产品参数结构化输出"""
    specifications: Dict[str, Any] = Field(
        description="结构化产品参数，根据产品类别智能提取关键参数"
    )
    category: Optional[str] = Field(
        default=None,
        description="产品类别（如相机、手机、冰箱等）"
    )
    key_features: List[str] = Field(
        default_factory=list,
        description="关键特性列表"
    )


class ComparisonResult(BaseModel):
    """产品对比结果结构化输出"""
    comparison_aspects: List[str] = Field(
        description="对比维度列表（如价格、性能、夜景拍摄等）。这是必需字段，必须提供至少2个对比维度。"
    )
    comparison_details: Dict[str, Dict[str, str]] = Field(
        description="各产品在各维度上的详细对比信息（必需字段）。这是一个嵌套字典：外层键是维度名（如'价格'、'夜景拍摄'），内层字典的键是产品名称，值是该产品在此维度的详细描述。必须为 comparison_aspects 中的每个维度都提供数据，且每个维度必须包含所有对比产品的信息。示例：{'价格': {'产品A名称': '¥8000', '产品B名称': '¥12000'}, '夜景拍摄': {'产品A名称': 'ISO 12800', '产品B名称': 'ISO 25600'}}"
    )
    scenario_analysis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="场景化分析结果（如果有用户场景），包含场景名称、各产品评分、推荐理由等。格式：{'场景': '场景名', '评分': {'产品名': 分数}, '推荐理由': '理由'}。"
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="综合推荐建议和理由，应基于对比分析给出明确的建议。"
    )


# ============== 工具实现 ==============

async def _extract_product_specifications_async(
    product_id: Annotated[
        int,
        Field(description="产品ID", examples=[1, 2, 100])
    ],
    aspect: Annotated[
        Optional[str],
        Field(
            default=None,
            description="关注维度（可选），如'夜景拍摄'、'续航能力'、'拍照性能'等。如果指定，将重点提取和分析该维度的相关参数。"
        )
    ] = None,
) -> str:
    """从产品描述中提取结构化参数（通用工具，适用于任何产品类别）

    功能说明：
    - 从产品的description和features字段中智能提取结构化参数
    - 根据产品类别自动识别关键参数（如相机类关注传感器、ISO等；手机类关注处理器、内存等）
    - 如果指定aspect，将重点关注该维度的相关参数
    - 提取的参数会缓存到数据库的specifications字段，提升后续查询性能

    设计原则：
    - 通用性：不硬编码特定产品类别的参数，由LLM自主判断提取哪些参数
    - 智能性：适应不同格式的产品描述，自动识别关键信息
    - 可扩展：支持指定关注维度，进行针对性分析

    Args:
        product_id: 产品ID
        aspect: 关注维度（可选），如"夜景拍摄"、"续航能力"等

    Returns:
        JSON格式：包含提取的参数、产品类别、关键特性，以及针对aspect的分析（如果指定）
    """
    try:
        # 运行时动态检查是否使用测试数据
        if is_use_test_data():
            # 测试数据模式：不需要数据库会话，db参数传入None
            product = get_product_by_id(None, product_id)
        else:
            # 真实数据模式：使用数据库会话
            with get_db_session() as db:
                product = get_product_by_id(db, product_id)
        
        if not product:
            return json.dumps({
                "text": f"未找到ID为 {product_id} 的产品",
                "specifications": None,
                "error": "product_not_found"
            }, ensure_ascii=False)

        # 安全访问属性（兼容测试数据和真实数据）
        product_name = getattr(product, 'name', '未知产品')
        product_model = getattr(product, 'model_number', None) or '未提供'
        brand_name = getattr(getattr(product, 'brand', None), 'name', None) if hasattr(product, 'brand') else None
        main_cat_name = getattr(getattr(product, 'main_category', None), 'name', None) if hasattr(product, 'main_category') else None
        sub_cat_name = getattr(getattr(product, 'sub_category', None), 'name', None) if hasattr(product, 'sub_category') else None
        product_description = getattr(product, 'description', None) or '无描述'
        product_features = getattr(product, 'features', None) or '无特性信息'
        product_price = getattr(product, 'price', None)
        product_specifications = getattr(product, 'specifications', None)

        # 如果已有specifications缓存且没有指定aspect，直接返回
        if product_specifications and not aspect:
            return json.dumps({
                "text": f"产品 {product_name} 的参数信息：\n{json.dumps(product_specifications, ensure_ascii=False, indent=2)}",
                "specifications": product_specifications,
                "category": main_cat_name,
                "cached": True
            }, ensure_ascii=False)

        # 使用LLM提取参数
        # 注意：由于 ProductSpecifications 包含 Dict[str, Any] 动态字典字段，
        # OpenAI 的结构化输出功能无法自动生成符合要求的 schema，
        # 因此使用 method="function_calling" 方法，这是处理动态字典的标准做法
        llm = get_default_llm()
        structured_llm = llm.with_structured_output(
            ProductSpecifications,
            method="function_calling"
        )

        # 构建产品信息上下文
        price_str = f"{product_price}元" if product_price else "价格面议"
        product_info = f"""
产品名称：{product_name}
产品型号：{product_model}
品牌：{brand_name or '未知'}
分类：{main_cat_name or '未知'} / {sub_cat_name or '未知'}
描述：{product_description}
特性：{product_features}
价格：{price_str}
"""

        # 构建prompt - 注意：ChatPromptTemplate 使用模板变量，不使用 f-string
        aspect_prompt_text = f"\n\n【重点关注】用户特别关注'{aspect}'这一维度，请重点提取和分析与该维度相关的参数。" if aspect else ""

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的产品参数提取专家。请从产品描述中提取结构化参数。

**核心原则**：
1. **通用性**：不硬编码特定产品类别的参数，根据产品类别智能识别关键参数
2. **智能性**：适应不同格式的产品描述，自动识别关键信息
3. **结构化**：返回标准化的参数结构，方便后续对比和分析

**提取规则**：
- 根据产品类别（如相机、手机、冰箱等）自主判断提取哪些参数
- 对于相机类产品，可能关注：传感器尺寸、像素、ISO范围、防抖、镜头等
- 对于手机类产品，可能关注：处理器、内存、存储、拍照能力、续航等
- 对于家电类产品，可能关注：容量、功率、能效、功能特性等
- 其他类别产品，根据描述智能判断关键参数

**输出要求**：
- specifications字段：包含所有提取的参数，使用中文字段名
- category字段：识别产品类别
- key_features字段：列出3-5个关键特性

**示例**：
相机产品示例（JSON格式）：
{{
  "specifications": {{
    "传感器": "1英寸",
    "有效像素": "2000万",
    "ISO范围": "ISO 100-12800",
    "防抖": "5轴防抖",
    "连拍速度": "20张/秒",
    "视频": "4K 60fps",
    "重量": "650g"
  }},
  "category": "相机",
  "key_features": ["1英寸大传感器", "5轴防抖", "4K视频"]
}}"""),
                ("user", """请从以下产品信息中提取结构化参数：{aspect_prompt}

{product_info}""")
        ])

        # 调用LLM提取参数
        result = await structured_llm.ainvoke(
            prompt_template.format_messages(
                aspect_prompt=aspect_prompt_text,
                product_info=product_info
            )
        )

        # 如果有指定aspect，进行针对性分析
        aspect_analysis = None
        if aspect:
            specifications_json = json.dumps(result.specifications, ensure_ascii=False, indent=2)
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个产品分析专家。请分析产品在特定维度上的表现。"),
                ("user", """产品：{product_name}
关注维度：{aspect}
产品参数：{specifications_json}

请分析该产品在'{aspect}'这一维度上的表现，包括：
1. 相关参数和指标
2. 优势点
3. 可能存在的不足

返回JSON格式：{{"analysis": "详细分析文本", "strengths": ["优势1", "优势2"], "weaknesses": ["不足1"]}}
""")
            ])
            analysis_result = await llm.ainvoke(
                analysis_prompt.format_messages(
                    product_name=product_name,
                    aspect=aspect,
                    specifications_json=specifications_json
                )
            )
            try:
                aspect_analysis = json.loads(analysis_result.content)
            except json.JSONDecodeError:
                aspect_analysis = {"analysis": analysis_result.content}

        # 更新数据库（异步更新，不阻塞返回）
        # 注意：测试数据模式下不更新数据库
        if not is_use_test_data():
            try:
                with get_db_session() as db:
                    db_product = get_product_by_id(db, product_id)
                    if db_product and hasattr(db_product, 'specifications'):
                        db_product.specifications = result.specifications
                        db.commit()
            except Exception as e:
                logger.warning(f"更新产品参数到数据库失败: {e}")

        # 构建返回结果
        result_dict = {
            "text": f"已提取产品 {product_name} 的结构化参数",
            "specifications": result.specifications,
            "category": result.category,
            "key_features": result.key_features,
            "aspect_analysis": aspect_analysis
        }

        # 生成人类可读文本
        text_parts = [f"📋 产品：{product_name}"]
        if result.category:
            text_parts.append(f"类别：{result.category}")
        if result.key_features:
            text_parts.append(f"关键特性：{', '.join(result.key_features)}")
        text_parts.append("\n📊 详细参数：")
        for key, value in result.specifications.items():
            text_parts.append(f"  • {key}：{value}")
        if aspect_analysis:
            text_parts.append(f"\n🎯 {aspect}维度分析：")
            text_parts.append(aspect_analysis.get("analysis", ""))
            if aspect_analysis.get("strengths"):
                text_parts.append("\n✅ 优势：")
                for strength in aspect_analysis["strengths"]:
                    text_parts.append(f"  • {strength}")
            if aspect_analysis.get("weaknesses"):
                text_parts.append("\n⚠️ 不足：")
                for weakness in aspect_analysis["weaknesses"]:
                    text_parts.append(f"  • {weakness}")

        result_dict["text"] = "\n".join(text_parts)

        return json.dumps(result_dict, ensure_ascii=False)

    except Exception as e:
        logger.error(f"提取产品参数失败: {e}", exc_info=True)
        return json.dumps({
            "text": f"提取产品参数时出错: {str(e)}",
            "specifications": None,
            "error": str(e)
        }, ensure_ascii=False)


@tool
def extract_product_specifications(
    product_id: Annotated[
        int,
        Field(description="产品ID", examples=[1, 2, 100])
    ],
    aspect: Annotated[
        Optional[str],
        Field(
            default=None,
            description="关注维度（可选），如'夜景拍摄'、'续航能力'、'拍照性能'等。如果指定，将重点提取和分析该维度的相关参数。"
        )
    ] = None,
) -> str:
    """从产品描述中提取结构化参数（通用工具，适用于任何产品类别）

    功能说明：
    - 从产品的description和features字段中智能提取结构化参数
    - 根据产品类别自动识别关键参数（如相机类关注传感器、ISO等；手机类关注处理器、内存等）
    - 如果指定aspect，将重点关注该维度的相关参数
    - 提取的参数会缓存到数据库的specifications字段，提升后续查询性能

    设计原则：
    - 通用性：不硬编码特定产品类别的参数，由LLM自主判断提取哪些参数
    - 智能性：适应不同格式的产品描述，自动识别关键信息
    - 可扩展：支持指定关注维度，进行针对性分析

    Args:
        product_id: 产品ID
        aspect: 关注维度（可选），如"夜景拍摄"、"续航能力"等

    Returns:
        JSON格式：包含提取的参数、产品类别、关键特性，以及针对aspect的分析（如果指定）
    """
    # 运行异步函数
    return run_async(_extract_product_specifications_async(product_id, aspect))


async def _compare_products_async(
    product_ids: Annotated[
        List[int],
        Field(
            description="要对比的产品ID列表（至少2个，最多5个）",
            examples=[[1, 2], [1, 2, 3]]
        )
    ],
    comparison_aspects: Annotated[
        Optional[List[str]],
        Field(
            default=None,
            description="对比维度（可选），如['价格', '性能', '夜景拍摄']。如果未指定，将自动识别关键对比维度。"
        )
    ] = None,
    user_scenario: Annotated[
        Optional[str],
        Field(
            default=None,
            description="用户使用场景（可选），如'VLOG拍摄'、'夜景拍摄'、'旅行使用'等。如果指定，将根据场景进行针对性推荐。"
        )
    ] = None,
) -> str:
    """对比多个产品，支持多维度分析和场景化推荐（通用工具，适用于任何产品类别）

    功能说明：
    - 提取各产品的参数信息（如果未提取，会自动提取）
    - 如果未指定对比维度，使用LLM自动识别关键对比维度
    - 进行多维度对比分析（价格、性能、适用场景等）
    - 如果有用户场景，进行场景化评分和推荐

    设计原则：
    - 灵活性：对比维度可由LLM自动识别，也可由用户指定
    - 场景化：支持根据用户具体场景进行智能推荐
    - 通用性：适用于任何产品类别的对比

    Args:
        product_ids: 产品ID列表（至少2个，最多5个）
        comparison_aspects: 对比维度（可选），如["价格", "性能", "夜景拍摄"]
        user_scenario: 用户使用场景（可选），如"VLOG拍摄"、"夜景拍摄"

    Returns:
        JSON格式：包含对比结果、各维度分析、场景化推荐（如果有场景）
    """
    try:
        if len(product_ids) < 2:
            return json.dumps({
                "text": "对比至少需要2个产品",
                "error": "insufficient_products"
            }, ensure_ascii=False)

        if len(product_ids) > 5:
            return json.dumps({
                "text": "一次最多对比5个产品",
                "error": "too_many_products"
            }, ensure_ascii=False)

        # 获取产品信息
        products_data = []
        
        # 运行时动态检查是否使用测试数据
        if is_use_test_data():
            # 测试数据模式：不需要数据库会话
            for pid in product_ids:
                product = get_product_by_id(None, pid)  # db参数为None
                if not product:
                    return json.dumps({
                        "text": f"未找到ID为 {pid} 的产品",
                        "error": f"product_{pid}_not_found"
                    }, ensure_ascii=False)
                
                # 安全访问属性
                product_specifications = getattr(product, 'specifications', None)

                # 如果没有specifications，先提取
                if not product_specifications:
                    spec_result = await _extract_product_specifications_async(pid, None)
                    try:
                        spec_json = json.loads(spec_result)
                        product_specifications = spec_json.get("specifications")
                        # 更新specifications属性（如果对象支持）
                        if hasattr(product, 'specifications'):
                            product.specifications = product_specifications
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析产品 {pid} 的参数提取结果，跳过")

                # 安全访问所有属性
                products_data.append({
                    "id": getattr(product, 'id', pid),
                    "name": getattr(product, 'name', '未知产品'),
                    "model_number": getattr(product, 'model_number', None),
                    "brand": getattr(getattr(product, 'brand', None), 'name', None) if hasattr(product, 'brand') and product.brand else None,
                    "category": getattr(getattr(product, 'main_category', None), 'name', None) if hasattr(product, 'main_category') and product.main_category else None,
                    "price": float(getattr(product, 'price', 0)) if getattr(product, 'price', None) else None,
                    "rating": float(getattr(product, 'rating', 0)) if hasattr(product, 'rating') else 0.0,
                    "specifications": product_specifications or {},
                    "description": getattr(product, 'description', None),
                    "features": getattr(product, 'features', None)
                })
        else:
            # 真实数据模式：使用数据库会话
            with get_db_session() as db:
                for pid in product_ids:
                    product = get_product_by_id(db, pid)
                    if not product:
                        return json.dumps({
                            "text": f"未找到ID为 {pid} 的产品",
                            "error": f"product_{pid}_not_found"
                        }, ensure_ascii=False)
                    
                    # 如果没有specifications，先提取
                    if not product.specifications:
                        spec_result = await _extract_product_specifications_async(pid, None)
                        try:
                            spec_json = json.loads(spec_result)
                            product.specifications = spec_json.get("specifications")
                        except json.JSONDecodeError:
                            logger.warning(f"无法解析产品 {pid} 的参数提取结果，跳过")

                    products_data.append({
                        "id": product.id,
                        "name": product.name,
                        "model_number": product.model_number,
                        "brand": product.brand.name if product.brand else None,
                        "category": product.main_category.name if product.main_category else None,
                        "price": float(product.price) if product.price else None,
                        "rating": product.rating,
                        "specifications": product.specifications or {},
                        "description": product.description,
                        "features": product.features
                    })

        # 使用LLM进行对比分析
        # 注意：由于 ComparisonResult 包含 Dict[str, Dict[str, str]] 深层嵌套字典结构，
        # function_calling 方法可能无法稳定生成完整的 schema。
        # 因此优先使用 JSON mode + 手动解析的方式，这样更可靠且可控
        llm = get_default_llm()

        # 构建产品信息 - 使用实际产品名称，便于LLM在对比结果中使用
        product_names_list = []  # 用于在prompt中明确列出产品名称
        products_info_parts = []
        for i, p in enumerate(products_data, 1):
            product_name = p['name']
            product_names_list.append(f"  - {product_name}")
            products_info_parts.append(f"""{product_name}（ID: {p['id']}）：
名称：{product_name}
品牌：{p['brand'] or '未知'}
型号：{p['model_number'] or '未提供'}
价格：{p['price']}元
评分：{p['rating']:.1f}分
参数：{json.dumps(p['specifications'], ensure_ascii=False, indent=2)}
""")
        products_info = "\n\n".join(products_info_parts)
        product_names_list_text = "\n".join(product_names_list)

        # 构建prompt - 注意：ChatPromptTemplate 使用模板变量，不使用 f-string
        aspects_prompt_text = f"\n对比维度：{', '.join(comparison_aspects)}" if comparison_aspects else "\n对比维度：请自动识别关键对比维度（如价格、性能、适用场景等）"
        scenario_prompt_text = f"\n用户使用场景：{user_scenario}\n请根据该场景进行针对性分析和推荐。" if user_scenario else ""

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的产品对比分析专家。请对多个产品进行多维度对比分析。

**核心原则**：
1. **多维度分析**：不只看价格或单一指标，综合考虑多个维度
2. **场景化推理**：根据用户的具体使用场景进行针对性分析
3. **通用性**：适用于任何产品类别的对比，不硬编码特定场景

**分析要求**：
- 如果未指定对比维度，自动识别关键对比维度（如价格、性能、适用场景、关键参数等）
- 对每个维度进行详细对比，找出各产品的优劣势
- 如果有用户场景，根据场景需求进行评分和推荐
- 给出明确的推荐建议和理由

**输出要求（必须严格遵循）**：

**字段1：comparison_aspects**（必需）
- 类型：字符串数组
- 内容：对比维度列表
- 示例：["价格", "传感器", "夜景拍摄", "防抖", "便携性"]
- 要求：至少提供2个对比维度

**字段2：comparison_details**（必需，这是最重要的字段）
- 类型：对象（字典），结构为：{{"维度名": {{"产品名称": "描述"}}}}
- **关键要求**：
  1. 这是必需字段，绝对不能为空
  2. 必须为 comparison_aspects 中的每个维度都创建一个键值对
  3. 每个维度的值必须是一个对象（字典），包含所有对比产品的信息
  4. 产品名称必须使用实际的产品名称（见下方产品列表），不能使用占位符
  5. 描述应该详细、具体，突出该产品在此维度上的特点

对比的产品名称列表（必须使用这些实际名称）：
{product_names_list}

**字段3：scenario_analysis**（可选，如果有用户场景则推荐提供）
- 类型：对象
- 结构：{{"场景": "场景名", "评分": {{"产品名": 分数}}, "推荐理由": "理由"}}
- 产品名称必须使用实际产品名称

**字段4：recommendation**（推荐提供）
- 类型：字符串
- 内容：综合推荐建议和理由

**完整示例格式**（注意：这里使用"产品A"、"产品B"只是格式示例，实际输出时必须使用真实产品名称）：
{{
  "comparison_aspects": ["价格", "传感器", "夜景拍摄"],
  "comparison_details": {{
    "价格": {{
      "实际产品名称1": "该产品在此维度的详细描述",
      "实际产品名称2": "该产品在此维度的详细描述"
    }},
    "传感器": {{
      "实际产品名称1": "该产品在此维度的详细描述",
      "实际产品名称2": "该产品在此维度的详细描述"
    }},
    "夜景拍摄": {{
      "实际产品名称1": "该产品在此维度的详细描述",
      "实际产品名称2": "该产品在此维度的详细描述"
    }}
  }},
  "scenario_analysis": {{
    "场景": "VLOG拍摄",
    "评分": {{
      "实际产品名称1": 8.5,
      "实际产品名称2": 9.0
    }},
    "推荐理由": "推荐理由说明"
  }},
  "recommendation": "综合推荐建议"
}}

**关键提醒**：
1. comparison_details 是必需字段，绝对不能缺失或为空
2. 必须为每个对比维度提供所有产品的对比信息
3. 使用实际产品名称，不要使用"产品A"、"产品1"等占位符"""),
            ("user", """请对比以下产品并返回JSON格式的结果：{aspects_prompt}{scenario_prompt}

{products_info}

**重要**：请严格按照上面定义的JSON格式返回对比结果，不要添加任何额外的说明文字。直接返回JSON对象，格式如下：
{{
  "comparison_aspects": [...],
  "comparison_details": {{...}},
  "scenario_analysis": {{...}},
  "recommendation": "..."
}}
""")
        ])

        # 调用LLM进行对比分析
        # 使用 JSON mode 作为主要方法，因为嵌套字典结构在 function_calling 中不够稳定
        # 这样可以获得更好的控制力和可靠性
        try:
            # 使用 JSON response format（如果支持）或要求返回纯JSON
            json_response = await llm.ainvoke(
                prompt_template.format_messages(
                    aspects_prompt=aspects_prompt_text,
                    scenario_prompt=scenario_prompt_text,
                    products_info=products_info,
                    product_names_list=product_names_list_text
                )
            )
            
            # 从响应中提取JSON
            content = json_response.content if hasattr(json_response, 'content') else str(json_response)
            
            # 尝试提取JSON（可能在markdown代码块中）
            import re
            # 匹配 markdown 代码块中的 JSON（支持多行）
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL | re.MULTILINE)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试匹配第一个 { 到最后一个 } 之间的内容（处理嵌套JSON）
                brace_count = 0
                start_idx = -1
                for i, char in enumerate(content):
                    if char == '{':
                        if start_idx == -1:
                            start_idx = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and start_idx != -1:
                            json_str = content[start_idx:i+1]
                            break
                else:
                    # 如果没找到完整JSON，尝试直接解析整个内容
                    json_str = content.strip()
            
            # 解析JSON
            comparison_data = json.loads(json_str)
            
            # 验证必需字段
            if "comparison_aspects" not in comparison_data:
                raise ValueError("响应中缺少 comparison_aspects 字段")
            if "comparison_details" not in comparison_data:
                raise ValueError("响应中缺少 comparison_details 字段（必需）")
            if not comparison_data.get("comparison_details"):
                raise ValueError("comparison_details 字段为空（必需）")
            
            # 构建 ComparisonResult 对象（用于类型验证和后续处理）
            comparison_result = ComparisonResult(
                comparison_aspects=comparison_data.get("comparison_aspects", []),
                comparison_details=comparison_data.get("comparison_details", {}),
                scenario_analysis=comparison_data.get("scenario_analysis"),
                recommendation=comparison_data.get("recommendation")
            )
            
            # 验证 comparison_details 是否包含所有维度的数据
            missing_aspects = set(comparison_result.comparison_aspects) - set(comparison_result.comparison_details.keys())
            if missing_aspects:
                logger.warning(f"comparison_details 缺少以下维度的数据: {missing_aspects}，将使用现有数据继续")
                
        except json.JSONDecodeError as json_error:
            logger.error(f"JSON解析失败: {json_error}，响应内容: {content[:500] if 'content' in locals() else 'N/A'}", exc_info=True)
            raise ValueError(f"产品对比分析失败：无法解析LLM返回的JSON格式。错误: {str(json_error)}") from json_error
        except Exception as e:
            logger.error(f"产品对比分析失败: {e}", exc_info=True)
            raise ValueError(f"产品对比分析失败: {str(e)}") from e

        # 构建返回结果
        result_dict = {
            "text": "",  # 将在下面生成
            "comparison_aspects": comparison_result.comparison_aspects,
            "comparison_details": comparison_result.comparison_details,
            "scenario_analysis": comparison_result.scenario_analysis,
            "recommendation": comparison_result.recommendation,
            "products": products_data
        }

        # 生成人类可读的对比报告
        text_parts = ["🔍 产品对比分析报告\n"]
        
        # 产品基本信息
        text_parts.append("📦 对比产品：")
        for i, p in enumerate(products_data, 1):
            price_info = f"¥{p['price']:.2f}" if p['price'] else "价格面议"
            text_parts.append(f"{i}. {p['name']}（{p['brand'] or '未知品牌'}）- {price_info}")
        
        # 对比维度分析
        text_parts.append("\n📊 对比维度：")
        for aspect in comparison_result.comparison_aspects:
            text_parts.append(f"\n【{aspect}】")
            if aspect in comparison_result.comparison_details:
                for product_name, detail in comparison_result.comparison_details[aspect].items():
                    text_parts.append(f"  • {product_name}：{detail}")
        
        # 场景化分析
        if comparison_result.scenario_analysis:
            text_parts.append(f"\n🎯 场景化分析（{comparison_result.scenario_analysis.get('场景', user_scenario)}）：")
            if "评分" in comparison_result.scenario_analysis:
                for product_name, score in comparison_result.scenario_analysis["评分"].items():
                    text_parts.append(f"  • {product_name}：{score}/10分")
            if "推荐理由" in comparison_result.scenario_analysis:
                text_parts.append(f"  推荐理由：{comparison_result.scenario_analysis['推荐理由']}")
        
        # 综合推荐
        if comparison_result.recommendation:
            text_parts.append(f"\n💡 推荐建议：\n{comparison_result.recommendation}")

        result_dict["text"] = "\n".join(text_parts)

        return json.dumps(result_dict, ensure_ascii=False)

    except Exception as e:
        logger.error(f"产品对比失败: {e}", exc_info=True)
        return json.dumps({
            "text": f"产品对比时出错: {str(e)}",
            "error": str(e)
        }, ensure_ascii=False)


@tool
def compare_products(
    product_ids: Annotated[
        List[int],
        Field(
            description="要对比的产品ID列表（至少2个，最多5个）",
            examples=[[1, 2], [1, 2, 3]]
        )
    ],
    comparison_aspects: Annotated[
        Optional[List[str]],
        Field(
            default=None,
            description="对比维度（可选），如['价格', '性能', '夜景拍摄']。如果未指定，将自动识别关键对比维度。"
        )
    ] = None,
    user_scenario: Annotated[
        Optional[str],
        Field(
            default=None,
            description="用户使用场景（可选），如'VLOG拍摄'、'夜景拍摄'、'旅行使用'等。如果指定，将根据场景进行针对性推荐。"
        )
    ] = None,
) -> str:
    """对比多个产品，支持多维度分析和场景化推荐（通用工具，适用于任何产品类别）

    功能说明：
    - 提取各产品的参数信息（如果未提取，会自动提取）
    - 如果未指定对比维度，使用LLM自动识别关键对比维度
    - 进行多维度对比分析（价格、性能、适用场景等）
    - 如果有用户场景，进行场景化评分和推荐

    设计原则：
    - 灵活性：对比维度可由LLM自动识别，也可由用户指定
    - 场景化：支持根据用户具体场景进行智能推荐
    - 通用性：适用于任何产品类别的对比

    Args:
        product_ids: 产品ID列表（至少2个，最多5个）
        comparison_aspects: 对比维度（可选），如["价格", "性能", "夜景拍摄"]
        user_scenario: 用户使用场景（可选），如"VLOG拍摄"、"夜景拍摄"

    Returns:
        JSON格式：包含对比结果、各维度分析、场景化推荐（如果有场景）
    """
    # 运行异步函数
    return run_async(_compare_products_async(product_ids, comparison_aspects, user_scenario))


def get_consultation_tools() -> list:
    """获取所有咨询工具列表"""
    return [
        extract_product_specifications,
        compare_products,
    ]
