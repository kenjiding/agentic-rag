"""确认相关路由"""
import logging
from fastapi import APIRouter, HTTPException
from src.api.models import ConfirmationResolveRequest
from src.confirmation import (
    get_confirmation_manager,
    ConfirmationNotFoundError,
    ConfirmationExpiredError,
    ConfirmationAlreadyResolvedError,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/confirmation/resolve")
async def resolve_confirmation(request: ConfirmationResolveRequest):
    """解析确认操作

    用户点击确认/取消按钮后调用此接口
    """
    try:
        manager = get_confirmation_manager()
        result = await manager.resolve_confirmation(
            request.confirmation_id,
            request.confirmed,
        )

        return {
            "success": True,
            "status": result.status,
            "action_type": result.action_type,
            "message": result.execution_result.get("text", "操作已完成") if result.execution_result else "操作已取消",
            "data": result.execution_result,
            "error": result.error,
        }
    except ConfirmationNotFoundError:
        raise HTTPException(status_code=404, detail="确认不存在")
    except ConfirmationExpiredError:
        raise HTTPException(status_code=410, detail="确认已过期")
    except ConfirmationAlreadyResolvedError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"解析确认失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/confirmation/pending/{session_id}")
async def get_pending_confirmation(session_id: str):
    """获取会话的待确认操作

    前端重新连接时检查是否有待确认操作
    """
    try:
        manager = get_confirmation_manager()
        confirmation = await manager.get_pending_confirmation(session_id)

        if confirmation:
            return {
                "has_pending": True,
                "confirmation": {
                    "confirmation_id": confirmation.confirmation_id,
                    "action_type": confirmation.action_type,
                    "display_message": confirmation.display_message,
                    "display_data": confirmation.display_data,
                    "expires_at": confirmation.expires_at.isoformat() if confirmation.expires_at else None,
                },
            }
        return {"has_pending": False}
    except Exception as e:
        logger.error(f"获取待确认操作失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

