"""API 路由模块"""
from fastapi import APIRouter
from . import chat, confirmation, selection, health

# 创建主路由
api_router = APIRouter(prefix="/api")

# 注册子路由
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(confirmation.router, tags=["confirmation"])
api_router.include_router(selection.router, tags=["selection"])
api_router.include_router(health.router, tags=["health"])

