"""Health Controller: 健康检查"""
import logging

from fastapi import APIRouter

from app.common.response import APIResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=APIResponse[dict])
async def health_check():
    """健康检查"""
    return APIResponse(data={"status": "healthy"})
