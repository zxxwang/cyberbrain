"""
Agent Life Event Service: 标准化的事件记录接口
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from app.repository.agent_life_event_repository import agent_life_event_repository

logger = logging.getLogger(__name__)


class AgentLifeEventService:
    """Agent 生命事件记录服务"""

    # 事件类型常量
    EVENT_TYPE_REACT_INPUT = "react_input"      # 听到的话
    EVENT_TYPE_REACT_OUTPUT = "react_output"    # 说过的话
    EVENT_TYPE_THOUGHT = "thought"              # 产生的念头
    EVENT_TYPE_TODO = "todo"                    # 待办任务
    EVENT_TYPE_SITUATION = "situation"          # 看到的事实
    EVENT_TYPE_MEMORY = "memory"                # 记忆写入
    EVENT_TYPE_EMOTION = "emotion"              # 情绪变化
    EVENT_TYPE_TASK_EXEC = "task_execution"     # 任务执行

    async def record_react_input(
        self,
        agent_id: str,
        signal: str,
        speaker: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        记录 react 输入（听到的话）

        Args:
            agent_id: Agent ID
            signal: 输入信号
            speaker: 说话人
            metadata: 额外元数据
        """
        return await agent_life_event_repository.insert_event(
            agent_id=agent_id,
            event_type=self.EVENT_TYPE_REACT_INPUT,
            data={
                "signal": signal,
                "speaker": speaker
            },
            metadata=metadata
        )

    async def record_react_output(
        self,
        agent_id: str,
        response: str,
        signal: str,
        speaker: str,
        metadata: Optional[Dict[str, Any]] = None,
        rt: Optional[float] = None,
        trace: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        记录 react 输出（说过的话）

        Args:
            agent_id: Agent ID
            response: 回复内容
            signal: 触发的输入
            speaker: 对话对象
            metadata: 额外元数据
            rt: 本次 react 调用的耗时（毫秒），写到文档顶层
            trace: 本轮完整认知过程（结构见 cyber_brain._build_react_trace），写到文档顶层。
                   返回的文档 ID 供异步写入的 emotion 事件通过 metadata.trace_id 关联回这一轮
        """
        return await agent_life_event_repository.insert_event(
            agent_id=agent_id,
            event_type=self.EVENT_TYPE_REACT_OUTPUT,
            data={
                "response": response,
                "signal": signal,
                "speaker": speaker
            },
            metadata=metadata,
            rt=rt,
            trace=trace,
        )

    async def record_thought(
        self,
        agent_id: str,
        thought: str,
        thought_type: str,
        context: Optional[Dict[str, Any]] = None,
        rt: Optional[float] = None
    ) -> Optional[str]:
        """
        记录念头/思考

        Args:
            agent_id: Agent ID
            thought: 念头内容
            thought_type: 念头类型（planning, reasoning, reflection, etc.）
            context: 上下文信息
            rt: 生成这条念头的耗时（毫秒）；手动写入（如初始化）时为 None
        """
        return await agent_life_event_repository.insert_event(
            agent_id=agent_id,
            event_type=self.EVENT_TYPE_THOUGHT,
            data={
                "thought": thought,
                "thought_type": thought_type,
                "context": context or {}
            },
            rt=rt,
        )

    async def record_todo(
        self,
        agent_id: str,
        task_id: str,
        content: str,
        action: str,
        task_data: Optional[Dict[str, Any]] = None,
        rt: Optional[float] = None
    ) -> Optional[str]:
        """
        记录待办任务事件

        Args:
            agent_id: Agent ID
            task_id: 任务ID
            content: 任务内容
            action: 动作（created, started, completed, cancelled）
            task_data: 任务详细数据
            rt: 产生这条待办的耗时（毫秒，created 时为漏斗判断耗时）；completed/cancelled 为 None
        """
        return await agent_life_event_repository.insert_event(
            agent_id=agent_id,
            event_type=self.EVENT_TYPE_TODO,
            data={
                "task_id": task_id,
                "content": content,
                "action": action,
                "task_data": task_data or {}
            },
            rt=rt,
        )

    async def record_situation(
        self,
        agent_id: str,
        situation: Dict[str, Any],
        change_type: str = "update"
    ) -> Optional[str]:
        """
        记录 Situation 变化

        Args:
            agent_id: Agent ID
            situation: Situation 数据
            change_type: 变化类型（update, major_change, etc.）
        """
        return await agent_life_event_repository.insert_event(
            agent_id=agent_id,
            event_type=self.EVENT_TYPE_SITUATION,
            data={
                "situation": situation,
                "change_type": change_type
            }
        )

    async def record_memory(
        self,
        agent_id: str,
        memory_type: str,
        memory_content: str,
        importance: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        记录记忆写入

        Args:
            agent_id: Agent ID
            memory_type: 记忆类型（episodic, brief, knowledge）
            memory_content: 记忆内容
            importance: 重要性评分
            metadata: 额外元数据
        """
        return await agent_life_event_repository.insert_event(
            agent_id=agent_id,
            event_type=self.EVENT_TYPE_MEMORY,
            data={
                "memory_type": memory_type,
                "content": memory_content,
                "importance": importance
            },
            metadata=metadata
        )

    async def record_emotion(
        self,
        agent_id: str,
        emotion_state: Dict[str, Any],
        trigger: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        记录情绪变化

        Args:
            agent_id: Agent ID
            emotion_state: 情绪状态（mood, stress, etc.）；对话触发时含 before / after / delta
            trigger: 触发原因
            metadata: 额外元数据（对话触发时带 trace_id，即对应 react_output 事件的文档 ID）
        """
        return await agent_life_event_repository.insert_event(
            agent_id=agent_id,
            event_type=self.EVENT_TYPE_EMOTION,
            data={
                "emotion_state": emotion_state,
                "trigger": trigger
            },
            metadata=metadata
        )

    async def record_task_execution(
        self,
        agent_id: str,
        task_id: str,
        task_content: str,
        result: Dict[str, Any]
    ) -> Optional[str]:
        """
        记录任务执行

        Args:
            agent_id: Agent ID
            task_id: 任务ID
            task_content: 任务内容
            result: 执行结果
        """
        return await agent_life_event_repository.insert_event(
            agent_id=agent_id,
            event_type=self.EVENT_TYPE_TASK_EXEC,
            data={
                "task_id": task_id,
                "content": task_content,
                "result": result
            }
        )

    async def get_agent_timeline(
        self,
        agent_id: str,
        event_types: Optional[List[str]] = None,
        page: int = 1,
        page_size: int = 50
    ) -> Dict[str, Any]:
        """
        获取 Agent 事件时间线（分页）

        Args:
            agent_id: Agent ID
            event_types: 事件类型过滤
            page: 页码（从1开始）
            page_size: 每页数量

        Returns:
            {
                "total": 总数,
                "page": 当前页,
                "page_size": 每页数量,
                "events": 事件列表
            }
        """
        skip = (page - 1) * page_size

        # 获取事件列表
        events = await agent_life_event_repository.find_events(
            agent_id=agent_id,
            event_types=event_types,
            limit=page_size,
            skip=skip
        )

        # 获取总数
        total = await agent_life_event_repository.count_events(
            agent_id=agent_id,
            event_types=event_types
        )

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "events": events
        }


# 全局单例
agent_life_event_service = AgentLifeEventService()
