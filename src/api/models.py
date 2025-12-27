"""API 请求和响应模型"""
from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    session_id: Optional[str] = "default"
    stream: bool = True


class ConfirmationResolveRequest(BaseModel):
    """确认解析请求"""
    confirmation_id: str
    confirmed: bool


class SelectionResolveRequest(BaseModel):
    """选择解析请求"""
    selection_id: str
    selected_option_id: str


class SelectionCancelRequest(BaseModel):
    """取消选择请求"""
    selection_id: str

