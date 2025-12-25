"""FastAPI 服务器 - 为前端提供流式 API 接口"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import asyncio
import logging
from src.multi_agent.graph import MultiAgentGraph
from src.multi_agent.config import MultiAgentConfig

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Agent API", version="1.0.0")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 MultiAgentGraph 实例
_graph: Optional[MultiAgentGraph] = None
_graph_initializing = False
_graph_lock = asyncio.Lock()


async def get_graph() -> MultiAgentGraph:
    """获取或创建 MultiAgentGraph 实例（异步，支持并发安全）"""
    global _graph, _graph_initializing
    
    if _graph is not None:
        return _graph
    
    async with _graph_lock:
        # 双重检查，避免重复初始化
        if _graph is not None:
            return _graph
        
        if _graph_initializing:
            # 如果正在初始化，等待完成
            while _graph_initializing:
                await asyncio.sleep(0.1)
            return _graph
        
        _graph_initializing = True
        try:
            config = MultiAgentConfig()
            loop = asyncio.get_event_loop()
            
            def init_graph():
                return MultiAgentGraph(
                    llm=None,
                    max_iterations=config.max_iterations
                )
            
            _graph = await loop.run_in_executor(None, init_graph)
            return _graph
        except Exception as e:
            logger.error(f"MultiAgentGraph 初始化失败: {e}", exc_info=True)
            raise
        finally:
            _graph_initializing = False


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    session_id: Optional[str] = "default"
    stream: bool = True


def format_state_update(state_update: Dict[str, Any]) -> Dict[str, Any]:
    """格式化状态更新为前端友好的格式

    返回统一的结构化响应：
    - 有结构化数据时：content 为简短描述，数据�� response_data 中
    - 无结构化数据时：content 为 AI 生成的完整回复
    """
    result = {
        "type": "state_update",
        "data": {
            "response_type": "text",
            "response_data": {}
        }
    }

    # 提取消息 - 查找最后一条 AI 消息
    messages = state_update.get("messages", [])
    has_products = False
    has_orders = False

    if messages:
        from langchain_core.messages import AIMessage, ToolMessage

        # 先提取工具结果中的结构化数据
        for message in messages:
            if isinstance(message, ToolMessage):
                try:
                    tool_content = message.content
                    if isinstance(tool_content, str):
                        try:
                            tool_result = json.loads(tool_content)
                        except:
                            continue

                        if isinstance(tool_result, dict):
                            if "products" in tool_result:
                                products = tool_result.get("products", [])
                                if products:
                                    result["data"]["response_data"]["products"] = products
                                    has_products = True
                            if "orders" in tool_result:
                                orders = tool_result.get("orders", [])
                                if orders:
                                    result["data"]["response_data"]["orders"] = orders
                                    has_orders = True
                except Exception:
                    pass

        # 提取文本内容
        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        if ai_messages:
            last_ai_message = ai_messages[-1]
            if hasattr(last_ai_message, "content") and last_ai_message.content:
                ai_content = last_ai_message.content

                # 当有结构化数据时，content 已经是工具的简短描述
                # 直接使用即可，无需额外处理
                result["data"]["content"] = ai_content
                result["data"]["role"] = "assistant"
        elif isinstance(messages[-1], dict):
            result["data"]["content"] = messages[-1].get("content", "")
            result["data"]["role"] = messages[-1].get("type", "assistant")

    # 确定响应类型
    if has_products and has_orders:
        result["data"]["response_type"] = "mixed"
    elif has_products:
        result["data"]["response_type"] = "product_list"
    elif has_orders:
        result["data"]["response_type"] = "order_list"

    # 提取其他信息
    if current_agent := state_update.get("current_agent"):
        result["data"]["current_agent"] = current_agent
    if tools_used := state_update.get("tools_used", []):
        result["data"]["tools_used"] = tools_used

    return result


async def stream_chat_response(question: str, session_id: str):
    """流式生成聊天响应"""
    try:
        graph = await get_graph()
        accumulated_state = {}
        execution_steps: list[str] = []
        step_details: list[dict] = []

        def add_step(step_name: str, detail: str = "") -> bool:
            """添加执行步骤，返回是否是新步骤"""
            if step_name and step_name not in execution_steps:
                execution_steps.append(step_name)
                step_details.append({"name": step_name, "detail": detail, "status": "running"})
                # 更新之前步骤的状态为完成
                for i in range(len(step_details) - 1):
                    step_details[i]["status"] = "completed"
                return True
            return False

        # 立即发送初始状态
        initial_step = "🚀 开始分析您的问题"
        add_step(initial_step)
        yield f"data: {json.dumps({'type': 'state_update', 'data': {'execution_steps': execution_steps, 'step_details': step_details}}, ensure_ascii=False)}\n\n"

        # 使用 updates 模式获取每个节点的更新
        async for state_update in graph.astream(question, stream_mode="updates"):
            # LangGraph 返回的格式是 {node_name: {updated_fields}}
            for node_name, node_update in state_update.items():
                # 跳过特殊节点
                if node_name in ("__start__", "__end__"):
                    continue

                # 生成步骤名称和详情
                step_name = _format_step_name(node_name, node_update)
                step_detail = _format_step_detail(node_name, node_update)

                if step_name:
                    is_new_step = add_step(step_name, step_detail)
                    # 如果有新步骤，立即发送
                    if is_new_step:
                        yield f"data: {json.dumps({'type': 'state_update', 'data': {'execution_steps': execution_steps, 'step_details': step_details}}, ensure_ascii=False)}\n\n"

                # 累积状态
                if isinstance(node_update, dict):
                    if "messages" in node_update and "messages" in accumulated_state:
                        existing_messages = accumulated_state.get("messages", [])
                        new_messages = node_update.get("messages", [])
                        existing_ids = {id(msg) if hasattr(msg, 'id') else str(msg) for msg in existing_messages}
                        for msg in new_messages:
                            msg_id = id(msg) if hasattr(msg, 'id') else str(msg)
                            if msg_id not in existing_ids:
                                existing_messages.append(msg)
                                existing_ids.add(msg_id)
                        accumulated_state["messages"] = existing_messages
                    else:
                        accumulated_state.update(node_update)

                # 发送状态更新
                formatted = format_state_update(accumulated_state)
                formatted["data"]["execution_steps"] = execution_steps
                formatted["data"]["step_details"] = step_details
                yield f"data: {json.dumps(formatted, ensure_ascii=False)}\n\n"

        # 标记所有步骤为完成
        for detail in step_details:
            detail["status"] = "completed"

        # 发送最终状态
        yield f"data: {json.dumps({'type': 'state_update', 'data': {'execution_steps': execution_steps, 'step_details': step_details}}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"


def _format_step_name(node_name: str, node_update: Dict[str, Any]) -> Optional[str]:
    """格式化执行步骤名称"""
    step_map = {
        "intent_recognition": "🎯 意图识别",
        "supervisor": "🧠 路由决策",
        "rag_agent": "📚 知识检索",
        "chat_agent": "💬 对话处理",
        "product_agent": "🛍️ 商品搜索",
        "order_agent": "📦 订单管理",
    }

    # 检查是否有路由决策信息
    if node_name == "supervisor" and isinstance(node_update, dict):
        selected_agent = node_update.get("current_agent")
        if selected_agent:
            agent_name = step_map.get(selected_agent, selected_agent)
            return f"🧠 路由到: {agent_name}"

    return step_map.get(node_name)


def _format_step_detail(node_name: str, node_update: Dict[str, Any]) -> str:
    """格式化执行步骤的详细描述"""
    detail_map = {
        "intent_recognition": "正在分析您的问题意图...",
        "supervisor": "智能路由正在选择最合适的助手...",
        "rag_agent": "正在从知识库中检索相关信息...",
        "chat_agent": "正在生成回答...",
        "product_agent": "正在搜索商品信息...",
        "order_agent": "正在查询订单信息...",
    }

    # 特殊处理：supervisor 路由决策
    if node_name == "supervisor" and isinstance(node_update, dict):
        selected_agent = node_update.get("current_agent")
        routing_reason = node_update.get("routing_reason", "")
        if selected_agent:
            agent_descriptions = {
                "rag_agent": "知识库检索助手",
                "chat_agent": "智能对话助手",
                "product_agent": "商品搜索助手",
                "order_agent": "订单管理助手",
            }
            desc = agent_descriptions.get(selected_agent, selected_agent)
            if routing_reason:
                return f"已选择 {desc}，原因：{routing_reason[:50]}..."
            return f"已选择 {desc}"

    # 检查是否有工具调用信息
    if isinstance(node_update, dict):
        tools_used = node_update.get("tools_used", [])
        if tools_used:
            tool_names = [t.get("tool", "").split("_")[-1] for t in tools_used if t.get("tool")]
            if tool_names:
                return f"正在使用工具：{', '.join(tool_names)}"

    return detail_map.get(node_name, "正在处理...")


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """聊天接口 - 支持流式响应"""
    if request.stream:
        return StreamingResponse(
            stream_chat_response(request.message, request.session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # 非流式响应（同步）
        graph = await get_graph()
        final_state = await graph.ainvoke(request.message)
        
        # 格式化响应
        response = format_state_update(final_state)
        return response


@app.get("/api/health")
async def health():
    """健康检查"""
    graph_status = "initialized" if _graph is not None else ("initializing" if _graph_initializing else "not_started")
    return {
        "status": "ok",
        "service": "ai-agent-api",
        "graph_status": graph_status
    }


@app.get("/")
async def root():
    """根路径"""
    return {"message": "AI Agent API Server", "version": "1.0.0"}


@app.on_event("startup")
async def startup_event():
    """应用启动时预初始化 MultiAgentGraph（后台进行，不阻塞启动）"""
    async def init_graph_background():
        try:
            await get_graph()
        except Exception as e:
            logger.error(f"MultiAgentGraph 后台初始化失败: {e}", exc_info=True)
    
    asyncio.create_task(init_graph_background())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

