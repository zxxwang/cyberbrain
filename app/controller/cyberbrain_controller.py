"""
CyberBrain Controller: 大脑主接口

提供 react 接口，直接透传给 cyber_brain.react() 方法
用于测试和集成调用
"""
import logging
from typing import Optional

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field

from app.service.cyber_brain import cyber_brain

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/brain", tags=["CyberBrain"])


class ReactRequest(BaseModel):
    """大脑反应请求"""
    agent_id: str = Field(..., description="Agent ID")
    signal: str = Field(..., description="外部输入信号")
    speaker: Optional[str] = Field(None, description="说话人姓名，用于关系查询、情绪反应和工作记忆标记")


class ReactResponse(BaseModel):
    """大脑反应响应"""
    response: str = Field(..., description="说出口的话")
    lies: list[str] = Field(default_factory=list, description="被标记为谎言的句子")
    silent: bool = Field(False, description="是否选择不说话（此时 response 为空）")
    route: str = Field(..., description="认知路径：fast / memory / deliberate")
    route_reason: str = Field("", description="选择该路径的原因：needs_reasoning / think_prob / hostile_to_known / low_action_confidence / knowledge / needs_memory / intuitive")
    emotion: dict = Field(..., description="System1 本能反应与认知门控")
    decision: Optional[dict] = Field(None, description="System2 社交决策（仅 deliberate 路径）")
    brief: list = Field(default_factory=list, description="召回的 Brief 记忆")
    episodic: list = Field(default_factory=list, description="召回的情景记忆")
    knowledge: list = Field(default_factory=list, description="召回的专业知识")
    requires_knowledge: bool = Field(..., description="是否需要专业知识")
    subjectivity: dict = Field(default_factory=dict, description="主体性观测：situation 进了哪些系统、是否有 overall_state")
    timings_ms: dict = Field(..., description="各阶段耗时（毫秒）")


@router.post("/react", response_model=ReactResponse, summary="大脑反应")
async def react(request: ReactRequest = Body(...)):
    """
    大脑主方法：接收外部信号，返回响应

    这是 CyberBrain 的核心接口，完整流程：
    1. 召回 Brief（信条/事实）
    2. System1：tev1 快速情绪量化打分
    3. 召回 Episodic（情景记忆）
    4. 如果需要知识，召回 Knowledge（知识记忆）
    5. 信号进入 PFC 短时记忆队列
    6. System2：结合所有信息，生成最终响应

    Args:
        request: 包含 agent_id, signal, speaker

    Returns:
        完整的响应结果，包含 response、emotion、记忆召回、耗时统计等
    """
    logger.info(f"CyberBrain.react: agent_id={request.agent_id}, speaker={request.speaker}, "
                f"signal={request.signal[:50]}...")

    result = await cyber_brain.react(
        agent_id=request.agent_id,
        signal=request.signal,
        speaker=request.speaker
    )

    return result

