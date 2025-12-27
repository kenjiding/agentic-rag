"""健康检查路由"""
from fastapi import APIRouter
from src.api.graph_manager import get_graph_status

router = APIRouter()


@router.get("/health")
async def health():
    """健康检查"""
    graph_status = get_graph_status()
    return {
        "status": "ok",
        "service": "ai-agent-api",
        "graph_status": graph_status
    }

