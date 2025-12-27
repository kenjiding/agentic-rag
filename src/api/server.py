"""FastAPI 服务器 - 为前端提供流式 API 接口"""
import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import api_router
from src.api.graph_manager import get_graph
from src.api.executors import register_confirmation_executors
from src.confirmation import get_confirmation_manager

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

# 注册 API 路由
app.include_router(api_router)


@app.get("/")
async def root():
    """根路径"""
    return {"message": "AI Agent API Server", "version": "1.0.0"}


@app.on_event("startup")
async def startup_event():
    """应用启动时预初始化 MultiAgentGraph 和 ConfirmationManager"""
    # 初始化确认管理器并注册执行器
    manager = get_confirmation_manager()
    register_confirmation_executors(manager)

    # 后台初始化 MultiAgentGraph
    async def init_graph_background():
        try:
            await get_graph()
        except Exception as e:
            logger.error(f"MultiAgentGraph 后台初始化失败: {e}", exc_info=True)

    asyncio.create_task(init_graph_background())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
