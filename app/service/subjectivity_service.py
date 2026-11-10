"""
Subjectivity Service: 主体性服务

职责：
- 维护Agent的主体性上下文（Subject Context）
- 包含客观处境（situation）、主观感受（overall_state）、内在念头（thoughts）

主体性 = 事实处境 + 主观感受 + 内在念头
"""
import asyncio
import json
import logging
import random
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from app.client.doubao_client import doubao_client
from app.client.embedding_client import embedding_client
from app.client.redis_client import redis_client
from app.config.settings import settings
from app.service.agent_life_event_service import agent_life_event_service

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构
# ============================================================================

class SubjectContext:
    """主体性上下文：Agent的主观世界"""

    def __init__(
        self,
        agent_id: str,
        situation: Dict = None,
        thoughts: List[Dict] = None,
        overall_state: Dict = None,
        pending_tasks: List[Dict] = None,
        recent_done: List[Dict] = None
    ):
        self.agent_id = agent_id
        self.situation = situation or {}
        self.thoughts = thoughts or []
        self.overall_state = overall_state or {
            "mood": 0.0,
            "stress": 0.3,
            "energy": 0.7
        }
        self.pending_tasks = pending_tasks or []
        # 近期完成/取消的任务（只留去重需要的字段：内容、向量、完成时间），完成的任务本身已从 pending_tasks 移除
        self.recent_done = recent_done or []
        self.last_update = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "situation": self.situation,
            "thoughts": self.thoughts,
            "overall_state": self.overall_state,
            "pending_tasks": self.pending_tasks,
            "recent_done": self.recent_done,
            "last_update": self.last_update
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SubjectContext":
        return cls(
            agent_id=data["agent_id"],
            situation=data.get("situation", {}),
            thoughts=data.get("thoughts", []),
            overall_state=data.get("overall_state", {}),
            pending_tasks=data.get("pending_tasks", []),
            recent_done=data.get("recent_done", [])
        )


# ============================================================================
# Service
# ============================================================================

class SubjectivityService:
    """主体性服务"""

    def __init__(self):
        self.redis_key_prefix = "subject_context"
        # Thoughts: 永久保留，只保留最近3个

    # ------------------------------------------------------------------------
    # Subject Context CRUD
    # ------------------------------------------------------------------------

    async def get_context(self, agent_id: str) -> Optional[SubjectContext]:
        """获取主体性上下文"""
        key = f"{self.redis_key_prefix}:{agent_id}"
        data = await redis_client.get(key)

        if not data:
            return None

        try:
            return SubjectContext.from_dict(json.loads(data))
        except Exception as e:
            logger.error(f"Failed to parse subject context for {agent_id}: {e}")
            return None

    async def save_context(self, context: SubjectContext):
        """保存主体性上下文"""
        key = f"{self.redis_key_prefix}:{context.agent_id}"
        context.last_update = datetime.now().isoformat()

        data = json.dumps(context.to_dict(), ensure_ascii=False)
        await redis_client.set(key, data, ex=30 * 86400)  # 30天过期

    def format_for_prompts(self, context: Optional[SubjectContext]) -> tuple[str, str]:
        """
        格式化主体性上下文为 prompt 字符串

        Returns:
            (state_and_situation, state_only)
            - state_and_situation: System1/2 使用（状态 + 处境）
            - state_only: System3 使用（仅状态）
        """
        if not context:
            return "", ""

        # State（内在状态）
        state_lines = []
        if context.overall_state:
            state_lines.append("## 整体状态")
            state_lines.append(f"心情：{context.overall_state.get('mood', 0.0):.1f}")
            state_lines.append(f"压力：{context.overall_state.get('stress', 0.3):.1f}")
            state_lines.append(f"精力：{context.overall_state.get('energy', 0.7):.1f}")

        state_only = "\n".join(state_lines) if state_lines else ""

        # Situation（客观处境）
        situation_lines = []
        if context.situation:
            situation_lines.append("\n## 客观处境")
            if context.situation.get('current_activity'):
                situation_lines.append(f"当前活动：{context.situation['current_activity']}")
            if context.situation.get('time_sensitive_events'):
                events = [f"- {e.get('description') or e.get('event')} ({e.get('deadline')})" for e in context.situation.get('time_sensitive_events', [])[:3]]
                if events:
                    situation_lines.append(f"时间敏感事件：\n" + "\n".join(events))
            if context.situation.get('ongoing_states'):
                situation_lines.append(f"持续状态：{', '.join(context.situation.get('ongoing_states', [])[:3])}")

        situation_str = "\n".join(situation_lines) if situation_lines else ""

        state_and_situation = state_only + situation_str

        return state_and_situation, state_only

    async def list_agents(self) -> List[str]:
        """列出所有有主体性上下文的Agent"""
        pattern = f"{self.redis_key_prefix}:*"
        keys = await redis_client.scan_keys(pattern)
        return [k.replace(f"{self.redis_key_prefix}:", "") for k in keys]

    # ------------------------------------------------------------------------
    # Situation（客观处境）
    # ------------------------------------------------------------------------

    async def set_situation(
        self,
        agent_id: str,
        current_activity: Optional[str] = None,
        time_sensitive_events: Optional[List[Dict]] = None,
        ongoing_states: Optional[List[str]] = None,
        attention_focus: Optional[List[str]] = None
    ):
        """设置客观处境"""
        context = await self.get_context(agent_id)
        if not context:
            context = SubjectContext(agent_id=agent_id)

        situation = {}
        if current_activity is not None:
            situation["current_activity"] = current_activity
        if time_sensitive_events is not None:
            situation["time_sensitive_events"] = time_sensitive_events
        if ongoing_states is not None:
            situation["ongoing_states"] = ongoing_states
        if attention_focus is not None:
            situation["attention_focus"] = attention_focus

        context.situation = situation
        await self.save_context(context)

        logger.info(f"Set situation for {agent_id}")

        # 🆕 记录 Situation 设置事件
        await agent_life_event_service.record_situation(
            agent_id=agent_id,
            situation=situation,
            change_type="set"
        )

    async def replace_situation(self, agent_id: str, situation: Dict):
        """
        用一个完整的处境字典整体替换当前处境（Memory Gate、待办执行使用）

        与 set_situation 的区别：set_situation 按四个标准字段逐个传参；这里原样保存整个字典，
        不会丢掉 recent_contacts 等额外字段
        """
        context = await self.get_context(agent_id)
        if not context:
            context = SubjectContext(agent_id=agent_id)
        context.situation = dict(situation)
        await self.save_context(context)
        logger.info(f"Replaced situation for {agent_id}")

        await agent_life_event_service.record_situation(
            agent_id=agent_id,
            situation=context.situation,
            change_type="replace"
        )

    async def update_situation(
        self,
        agent_id: str,
        **updates
    ):
        """更新客观处境（部分字段）"""
        context = await self.get_context(agent_id)
        if not context:
            context = SubjectContext(agent_id=agent_id)

        for key, value in updates.items():
            if value is not None:
                context.situation[key] = value

        await self.save_context(context)

        # 🆕 记录 Situation 更新事件
        await agent_life_event_service.record_situation(
            agent_id=agent_id,
            situation=context.situation,
            change_type="update"
        )

    async def get_situation(self, agent_id: str) -> Dict:
        """获取客观处境"""
        context = await self.get_context(agent_id)
        return context.situation if context else {}

    async def clean_expired_events(self, agent_id: str):
        """清理过期的时间敏感事件"""
        context = await self.get_context(agent_id)
        if not context or not context.situation.get("time_sensitive_events"):
            return

        today = datetime.now().date()
        events = context.situation["time_sensitive_events"]

        # 过滤掉deadline < today的事件；无deadline或格式错误的保留
        valid_events = []
        for e in events:
            try:
                if datetime.strptime(e["deadline"], "%Y-%m-%d").date() < today:
                    continue
            except (KeyError, TypeError, ValueError):
                pass
            valid_events.append(e)

        if len(valid_events) < len(events):
            context.situation["time_sensitive_events"] = valid_events
            await self.save_context(context)
            logger.info(f"Cleaned {len(events) - len(valid_events)} expired events for {agent_id}")

    async def format_situation_for_prompt(self, agent_id: str) -> str:
        """格式化situation用于prompt"""
        situation = await self.get_situation(agent_id)
        if not situation:
            return ""

        lines = []

        if situation.get("current_activity"):
            lines.append(f"- {situation['current_activity']}")

        if situation.get("time_sensitive_events"):
            today = datetime.now().date()
            for event in situation["time_sensitive_events"]:
                # 过期、无deadline或格式错误的事件不展示
                try:
                    deadline = datetime.strptime(event["deadline"], "%Y-%m-%d").date()
                except (KeyError, TypeError, ValueError):
                    continue
                days_left = (deadline - today).days
                if days_left < 0:
                    continue

                if days_left == 0:
                    time_str = "今天"
                elif days_left == 1:
                    time_str = "明天"
                else:
                    time_str = f"还有{days_left}天"

                lines.append(f"- {event['description']}（{deadline.strftime('%m月%d日')}，{time_str}）")

        if situation.get("ongoing_states"):
            for state in situation["ongoing_states"]:
                lines.append(f"- {state}")

        if situation.get("attention_focus"):
            lines.append(f"- 当前关注：{', '.join(situation['attention_focus'])}")

        return "\n".join(lines) if lines else ""

    # ------------------------------------------------------------------------
    # Narrative（个人叙事）
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # Thoughts（内在念头）
    # ------------------------------------------------------------------------

    async def add_thought(
        self,
        agent_id: str,
        thought: str,
        related_to: Optional[str] = None,
        target_person: Optional[str] = None,
        urgency: float = 0.0,
        rt: Optional[float] = None
    ):
        """
        添加念头（永久保留，只保留最近3个）

        Args:
            rt: 生成这条念头的耗时（毫秒），由调用方计时后传入；手动写入（如初始化）时为 None
        """
        context = await self.get_context(agent_id)
        if not context:
            context = SubjectContext(agent_id=agent_id)

        new_thought = {
            "time": datetime.now().isoformat(),
            "thought": thought,
            "related_to": related_to,
            "target_person": target_person,
            "urgency": urgency,
            "expressed": False
        }

        context.thoughts.append(new_thought)

        # 按时间排序，只保留最近3个
        context.thoughts.sort(key=lambda t: t["time"], reverse=True)
        context.thoughts = context.thoughts[:3]

        await self.save_context(context)
        logger.info(f"Added thought for {agent_id}: {thought[:30]}... (keeping latest 3)")

        # 🆕 记录念头事件
        await agent_life_event_service.record_thought(
            agent_id=agent_id,
            thought=thought,
            thought_type="spontaneous",
            context={
                "related_to": related_to,
                "target_person": target_person,
                "urgency": urgency
            },
            rt=rt,
        )

    async def get_unexpressed_thoughts(
        self,
        agent_id: str,
        target_person: Optional[str] = None
    ) -> List[Dict]:
        """获取未表达的念头"""
        context = await self.get_context(agent_id)
        if not context:
            return []

        thoughts = [t for t in context.thoughts if not t.get("expressed", False)]

        if target_person:
            thoughts = [t for t in thoughts if t.get("target_person") == target_person]

        # 按urgency排序
        thoughts.sort(key=lambda t: t.get("urgency", 0.0), reverse=True)

        return thoughts

    async def get_thoughts_for_speaker(
        self,
        agent_id: str,
        speaker: str,
        limit: int = 3
    ) -> List[Dict]:
        """和对话人有关的最近几个未表达念头：指向他的，以及没有特定对象的；指向别人的不取"""
        context = await self.get_context(agent_id)
        if not context:
            return []

        thoughts = [
            t for t in context.thoughts
            if not t.get("expressed", False) and t.get("target_person") in (None, "", speaker)
        ]
        thoughts.sort(key=lambda t: t["time"], reverse=True)
        return thoughts[:limit]

    async def mark_thought_expressed(
        self,
        agent_id: str,
        thought_time: str
    ):
        """标记念头已表达"""
        context = await self.get_context(agent_id)
        if not context:
            return

        for thought in context.thoughts:
            if thought["time"] == thought_time:
                thought["expressed"] = True
                thought["expressed_at"] = datetime.now().isoformat()
                break

        await self.save_context(context)

    async def generate_thought(
        self,
        agent_id: str,
        trigger: str = "spontaneous"
    ) -> Optional[str]:
        """生成内在念头

        Args:
            agent_id: Agent ID
            trigger: 触发原因 (spontaneous/situation_change/deadline_approaching)

        Returns:
            生成的念头内容
        """
        context = await self.get_context(agent_id)
        if not context:
            return None

        # 构建prompt
        situation_str = await self.format_situation_for_prompt(agent_id)

        if not situation_str:
            return None

        now = datetime.now()
        prompt = f"""你是这个Agent的内心世界。

                    现在是 {now.strftime('%Y年%m月%d日 %H:%M')}。

                    【客观处境】
                    {situation_str}

                    【整体状态】
                    - 心情：{context.overall_state.get('mood', 0.0):.2f}
                    - 压力：{context.overall_state.get('stress', 0.3):.2f}
                    - 精力：{context.overall_state.get('energy', 0.7):.2f}

                    此刻内心会冒出什么念头？（内心独白，20字以内）
                    有没有想要做的事或想找的人？

                    输出JSON：
                    {{
                      "thought": "念头内容",
                      "target_person": "想找谁（如果有）",
                      "urgency": 0.0-1.0,
                      "related_to": "关联的事件（如果有）"
                    }}

                    只输出JSON，不要其他内容。"""

        try:
            response = await doubao_client.generate(prompt, max_tokens=300)

            # 去除可能的markdown代码块
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            thought_data = json.loads(response)
            return thought_data
        except Exception as e:
            logger.error(f"Failed to generate thought: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------------
    # Overall State（整体状态）
    # ------------------------------------------------------------------------

    async def set_overall_state(
        self,
        agent_id: str,
        mood: float,
        stress: float,
        energy: float
    ):
        """
        直接设置overall_state（用于初始化）

        Args:
            agent_id: Agent ID
            mood: 心情值 [0.0, 1.0]
            stress: 压力值 [0.0, 1.0]
            energy: 精力值 [0.0, 1.0]
        """
        context = await self.get_context(agent_id)
        if not context:
            context = SubjectContext(agent_id=agent_id)

        context.overall_state["mood"] = max(0.0, min(1.0, mood))
        context.overall_state["stress"] = max(0.0, min(1.0, stress))
        context.overall_state["energy"] = max(0.0, min(1.0, energy))

        await self.save_context(context)
        logger.info(f"Set overall_state for {agent_id}: mood={mood:.2f}, stress={stress:.2f}, energy={energy:.2f}")

    async def adjust_overall_state(
        self,
        agent_id: str,
        mood_delta: float = 0.0,
        stress_delta: float = 0.0,
        reason: str = ""
    ):
        """
        调整overall_state（由Memory Gate调用）

        Args:
            agent_id: Agent ID
            mood_delta: 心情变化 [-1.0, 1.0]
            stress_delta: 压力变化 [-1.0, 1.0]
            reason: 调整原因

        Returns:
            {"before": 调整前的 overall_state, "after": 调整后的 overall_state}
        """
        context = await self.get_context(agent_id)
        if not context:
            context = SubjectContext(agent_id=agent_id)
        before = dict(context.overall_state)

        # 调整心情
        current_mood = context.overall_state.get("mood", 0.0)
        new_mood = max(-1.0, min(1.0, current_mood + mood_delta))
        context.overall_state["mood"] = new_mood

        # 调整压力
        current_stress = context.overall_state.get("stress", 0.3)
        new_stress = max(0.0, min(1.0, current_stress + stress_delta))
        context.overall_state["stress"] = new_stress

        # 重新计算精力
        context.overall_state["energy"] = max(0.1, 1.0 - new_stress * 0.7)

        await self.save_context(context)
        logger.info(
            f"Adjusted overall_state for {agent_id}: "
            f"mood {current_mood:.2f} -> {new_mood:.2f}, "
            f"stress {current_stress:.2f} -> {new_stress:.2f} "
            f"({reason})"
        )
        return {"before": before, "after": dict(context.overall_state)}

    # ------------------------------------------------------------------------
    # 用于Prompt的格式化
    # ------------------------------------------------------------------------

    async def format_for_system1(self, agent_id: str) -> str:
        """格式化为System1的输入（简洁版）

        System1 只需要：
        - Overall State（当前状态的数值）
        - Situation（客观背景）
        """
        context = await self.get_context(agent_id)
        if not context:
            return ""

        lines = []

        # 整体状态
        state = context.overall_state
        lines.append("【当前整体状态】")
        lines.append(f"- 心情：{state.get('mood', 0.0):.2f}")
        lines.append(f"- 压力：{state.get('stress', 0.3):.2f}")
        lines.append(f"- 精力：{state.get('energy', 0.7):.2f}")
        lines.append("")

        # 客观处境
        situation_str = await self.format_situation_for_prompt(agent_id)
        if situation_str:
            lines.append("【客观处境】")
            lines.append(situation_str)

        return "\n".join(lines).rstrip()


    # ------------------------------------------------------------------------
    # Pending Tasks（待办事项）
    # ------------------------------------------------------------------------

    async def add_todo(self, agent_id: str, content: str, source: str, context: Dict = None) -> Optional[str]:
        """
        智能添加待办（多层漏斗 + 去重）。创建成功时，漏斗判断到写入的耗时记为 todo 事件的 rt

        - pending_tasks 只存待执行的任务，不设数量上限；完成/取消后从中移除
        - 去重：新任务与待执行任务、TODO_DEDUP_WINDOW_HOURS 内完成/取消的任务比较。
          内容完全相同直接跳过；否则余弦 ≥ TODO_DEDUP_SIMILARITY 的作为候选，由 System One 判断是否同一件事
        """
        t_start = time.perf_counter()
        try:
            ctx = context or {}
            if source == "thought":
                result = await self._thought_funnel(content, ctx)
            elif source == "dialogue":
                result = await self._dialogue_funnel(content, ctx)
            else:
                return None
            if not result:
                return None

            # 向量在锁外计算（远程调用），锁内只做比较和写入
            vector = await self._task_vector(result["content"])

            lock = self._tasks_lock_key(agent_id)
            if not await self._acquire_tasks_lock(lock):
                logger.warning(f"add_todo skipped for {agent_id}: tasks lock busy")
                return None
            try:
                subject_ctx = await self.get_context(agent_id)
                if not subject_ctx:
                    subject_ctx = SubjectContext(agent_id=agent_id)
                self._prune_recent_done(subject_ctx)

                duplicate = await self._find_duplicate_task(subject_ctx, result["content"], vector)
                if duplicate:
                    logger.info(
                        f"Todo dedup for {agent_id}: '{result['content']}' ≈ '{duplicate['content']}' "
                        f"(sim={duplicate['similarity']:.3f}, {duplicate['status']}), skipped"
                    )
                    return None

                task = {
                    "task_id": f"task_{uuid.uuid4().hex}",
                    "content": result["content"],
                    "source": source,
                    "priority": result.get("priority", 0.6),
                    "deadline": result.get("deadline"),
                    "related_person": result.get("related_person"),
                    "interaction_type": result.get("interaction_type"),
                    "created_at": datetime.now().isoformat(),
                    "status": "pending",
                }
                if vector is not None:
                    task["vector"] = vector
                subject_ctx.pending_tasks.append(task)
                subject_ctx.pending_tasks.sort(key=lambda t: t.get("priority", 0), reverse=True)
                await self.save_context(subject_ctx)
            finally:
                await redis_client.release_lock(lock)

            logger.info(f"✅ Todo: {task['content']} ({source}, p={task['priority']:.2f})")
            await agent_life_event_service.record_todo(
                agent_id=agent_id,
                task_id=task["task_id"],
                content=task["content"],
                action="created",
                task_data=self._public_task(task),
                rt=(time.perf_counter() - t_start) * 1000,
            )
            return task["task_id"]
        except Exception as e:
            logger.error(f"add_todo error: {e}", exc_info=True)
            return None

    # ---------------- 待办：去重与并发控制 ----------------

    @staticmethod
    def _tasks_lock_key(agent_id: str) -> str:
        return f"subject_context:{agent_id}:tasks_lock"

    @staticmethod
    async def _acquire_tasks_lock(lock: str, timeout: float = 5.0) -> bool:
        """
        pending_tasks 的读-改-写要串行：react 的对话待办、念头生成、任务完成可能同时修改同一个 agent，
        不加锁时后写的会覆盖先写的。锁 10 秒自动过期，防止进程崩溃后死锁
        """
        deadline = time.monotonic() + timeout
        while True:
            if await redis_client.acquire_lock(lock, 10):
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.05)

    @staticmethod
    async def _task_vector(content: str) -> Optional[List[float]]:
        """任务内容的向量；embedding 失败时返回 None，去重退化为内容完全相同才算重复"""
        try:
            return await asyncio.to_thread(embedding_client.embed, content)
        except Exception as e:
            logger.warning(f"Task embedding failed, dedup falls back to exact match: {e}")
            return None

    @staticmethod
    def _prune_recent_done(ctx: "SubjectContext") -> None:
        """清掉超出去重窗口的已完成/已取消记录"""
        cutoff = (datetime.now() - timedelta(hours=settings.TODO_DEDUP_WINDOW_HOURS)).isoformat()
        ctx.recent_done = [d for d in ctx.recent_done if d.get("done_at", "") >= cutoff]

    # 去重的两步：embedding 粗筛候选 + System One 判断是否同一件事。
    # 只靠阈值分不开：同一件事换个说法余弦常在 0.65 上下，"明早吃药"与"明早开会"却有 0.78
    DEDUP_MAX_CANDIDATES = 3
    RECENT_DONE_MAX = 20  # recent_done 最多保留的条数（完成任务时截断，留最近的）
    DEDUP_FALLBACK_SIMILARITY = 0.9  # System One 不可用时，只把余弦 ≥ 此值的当作重复（宁可漏判，不误吞新任务）

    async def _find_duplicate_task(
        self, ctx: "SubjectContext", content: str, vector: Optional[List[float]]
    ) -> Optional[Dict]:
        """
        在待执行任务和近期完成/取消的任务里找重复，返回重复的那条（含 similarity、status），没有返回 None。
        早期创建的任务没有向量，第一次比较时补算并写回（随后由调用方 save_context 一起保存）
        """
        candidates = [(t, t.get("status", "pending")) for t in ctx.pending_tasks]
        candidates += [(d, d.get("status", "completed")) for d in ctx.recent_done]

        scored = []
        for item, status in candidates:
            if item.get("content", "").strip() == content.strip():
                return {"content": item["content"], "similarity": 1.0, "status": status}
            if vector is None:
                continue
            other = item.get("vector")
            if other is None and item.get("content"):
                other = await self._task_vector(item["content"])
                if other is not None:
                    item["vector"] = other
            if other is None:
                continue
            a, b = np.asarray(vector), np.asarray(other)
            sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            if sim >= settings.TODO_DEDUP_SIMILARITY:
                scored.append({"content": item["content"], "similarity": sim, "status": status})

        if not scored:
            return None
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return await self._judge_same_task(content, scored[:self.DEDUP_MAX_CANDIDATES])

    async def _judge_same_task(self, content: str, candidates: List[Dict]) -> Optional[Dict]:
        """让 System One 判断新待办是否与某个候选是同一件事；判断失败时退化为高阈值"""
        criteria = {"无": "都不是同一件事"}
        for i, c in enumerate(candidates, 1):
            criteria[str(i)] = c["content"]
        try:
            resp = await doubao_client.decide(
                state=f"新待办：{content}",
                questions={"same_as": {
                    "type": "choice",
                    "instructions": "新待办和下面哪个已有待办是同一件事？同一件事指要做的事、针对的人和目的都相同，只是说法不同。"
                                    "动作或对象不同就不是同一件事（如提醒吃药 vs 提醒开会、发周报 vs 发合同），选'无'。",
                    "criteria": criteria,
                }},
            )
            choice = resp.get("answers", {}).get("same_as", {}).get("choice", "无")
        except Exception as e:
            logger.warning(f"Todo dedup judge failed, fallback to similarity >= {self.DEDUP_FALLBACK_SIMILARITY}: {e}")
            best = candidates[0]
            return best if best["similarity"] >= self.DEDUP_FALLBACK_SIMILARITY else None

        if choice in criteria and choice != "无":
            return candidates[int(choice) - 1]
        return None

    @staticmethod
    def _public_task(task: Dict) -> Dict:
        """对外展示/写事件用的任务字段（去掉内部的向量）"""
        return {k: v for k, v in task.items() if k != "vector"}

    async def _thought_funnel(self, content: str, ctx: Dict) -> Optional[Dict]:
        """Thought 3层: 关键词 → decide → LLM"""
        keywords = ["得","要","应该","需要","必须","找","联系","问","谈","说","做","写","改","查","看","准备","安排","计划","约"]
        if any(k in content for k in keywords):
            logger.info(f"Thought L1(kw): {content[:20]}")
            return {"content": content, "priority": 0.6, "related_person": self._extract_person(content)}

        # decide
        decide_result = await doubao_client.decide(
            state=f"念头内容：{content}\n强度：{ctx.get('intensity', 0.5):.2f}",
            questions={
                "need_task": {
                    "type": "choice",
                    "instructions": "这个念头是否需要转为待办任务？",
                    "criteria": {
                        "yes": "包含明确行动意图，指向具体的人或事",
                        "no": "纯情绪、抽象担忧、无明确行动"
                    }
                }
            }
        )
        choice = decide_result.get("answers", {}).get("need_task", {}).get("choice", "no")
        confidence = decide_result.get("answers", {}).get("need_task", {}).get("confidence", 0.0)

        if choice == "yes":
            if confidence > 0.7:
                logger.info(f"Thought L2(decide={confidence:.2f}): {content[:20]}")
                return {"content": content, "priority": confidence}
            elif confidence > 0.5:
                logger.info(f"Thought L3(LLM, decide={confidence:.2f}): {content[:20]}")
                llm = await self._llm_judge_thought(content, ctx)
                if llm.get("should_add"):
                    return {"content": llm.get("task_content") or content, "priority": llm.get("priority",0.6), "related_person": llm.get("related_person")}
        logger.debug(f"Thought reject: {content[:20]}")
        return None

    async def _dialogue_funnel(self, content: str, ctx: Dict) -> Optional[Dict]:
        """
        对话待办 2 层：System One 粗判是否有承诺 → 置信度 > 0.5 时交给 lite 模型
        再确认一次是否有任务、把原话提炼成任务描述、按语义推出截止日期
        """
        signal, response, speaker = ctx.get("signal", content), ctx.get("response", ""), ctx.get("speaker", "用户")

        decide_result = await doubao_client.decide(
            state=f"{speaker}：{signal} Agent：{response}",
            questions={
                "has_commitment": {
                    "type": "choice",
                    "instructions": "Agent是否承诺了需要后续完成的待办任务？",
                    "criteria": {
                        "yes": "说话人明确请求Agent完成某个行动（做、汇报、提交、联系等） 且 Agent明确承诺（好的/我会/可以/记住了）",
                        "no": "闲聊、询问情况、让对方继续说话、Agent拒绝、已完成的事、无明确承诺、纯粹的说明请求"
                    }
                },
                "needs_interaction": {
                    "type": "choice",
                    "instructions": "任务是否需要主动联系/找某人沟通？",
                    "criteria": {
                        "yes": "需要找人说话、联系某人、通知某人、跟某人沟通、汇报给某人、向某人反馈、告诉某人、回复某人",
                        "no": "自己能完成、不涉及联系他人、已经完成的对话"
                    }
                }
            }
        )
        answers = decide_result.get("answers", {})
        choice = answers.get("has_commitment", {}).get("choice", "no")
        confidence = answers.get("has_commitment", {}).get("confidence", 0.0)
        needs_interaction = answers.get("needs_interaction", {}).get("choice", "no")
        logger.info(f"🔍 Dialogue funnel: has_commitment={choice}({confidence:.2f}), "
                    f"needs_interaction={needs_interaction}, signal={signal[:30]}")

        if choice != "yes" or confidence <= 0.5:
            return None

        extracted = await self._llm_extract_dialogue_task(signal, response, speaker)
        if not extracted:
            logger.info(f"Dialogue funnel: LLM found no task, signal={signal[:30]}")
            return None

        task = {
            "content": extracted["content"],
            "priority": 0.8,
            "deadline": extracted["deadline"],
            "related_person": speaker,
        }
        if needs_interaction == "yes":
            task["interaction_type"] = "dialogue"
        logger.info(f"Dialogue task (decide={confidence:.2f}): {task['content']} deadline={task['deadline']}")
        return task

    # 提示词尽量短、输出只有一两行纯文本（不用 JSON），减少输入和输出 token
    DIALOGUE_TASK_PROMPT = """今天是{today}（{weekday}）。
{speaker}：{signal}
我：{response}

我答应了要去做某件事吗？没有就只输出：无
有就输出两行，不加任何前缀：
任务描述（不照抄原话，写清做什么，30字内）
截止日期 YYYY-MM-DD（按话里的时间推算；没提时间写：无）"""

    WEEKDAYS = "一二三四五六日"

    async def _llm_extract_dialogue_task(self, signal: str, response: str, speaker: str) -> Optional[Dict]:
        """lite 模型确认任务、提炼描述、推算截止日期。没有任务或调用失败返回 None"""
        today = datetime.now()
        prompt = self.DIALOGUE_TASK_PROMPT.format(
            today=today.strftime("%Y-%m-%d"), weekday=f"星期{self.WEEKDAYS[today.weekday()]}",
            speaker=speaker, signal=signal, response=response or "（没有回应）",
        )
        try:
            resp = await doubao_client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=80, model=settings.DOUBAO_MODEL_LITE,
            )
        except Exception as e:
            logger.error(f"Dialogue task extraction failed: {e}", exc_info=True)
            return None

        # 模型偶尔会加"任务描述：""截止日期："之类的前缀，去掉；序号前缀必须带标点（"1."），避免吃掉日期开头的数字
        prefix = re.compile(r"^\s*(?:[12][.、:：)]|任务描述|任务|截止日期|截止)\s*[:：]?\s*")
        lines = [prefix.sub("", l).strip() for l in (resp or "").strip().splitlines() if l.strip()]
        if not lines or lines[0].rstrip("。.") in ("无", "没有"):
            return None

        deadline = None
        if len(lines) > 1:
            m = re.search(r"\d{4}-\d{2}-\d{2}", lines[1])
            if m:
                try:
                    deadline = datetime.strptime(m.group(0), "%Y-%m-%d").date().isoformat()
                except ValueError:
                    pass
        return {"content": lines[0], "deadline": deadline}

    def _extract_person(self, content: str) -> Optional[str]:
        import re
        if "和" in content:
            m = re.search(r'和([^\s，。、]+)', content)
            if m: return m.group(1)
        return None

    async def _llm_judge_thought(self, content: str, ctx: Dict) -> Dict:
        prompt = f'念头：{content}\n强度：{ctx.get("intensity",0.5):.2f}\n\n标准：明确行动意图，指向具体人或事。\n反例：纯情绪、抽象担忧。\n\n输出JSON：{{"should_add":true/false,"task_content":"任务(50字内)","priority":0.6,"related_person":"人名或null"}}\n\n只输出JSON。'
        try:
            resp = await doubao_client.generate(prompt, max_tokens=200)
            resp = resp.strip()
            if resp.startswith("```"): resp = "\n".join(resp.split("\n")[1:-1])
            return json.loads(resp)
        except: return {"should_add": False}

    async def get_pending_tasks(self, agent_id: str, status: str = "pending") -> List[Dict]:
        """
        待执行任务（不含内部向量）。完成/取消的任务已从列表移除，
        status 参数保留兼容：传 "pending" 或 "all" 都返回全部待执行任务
        """
        ctx = await self.get_context(agent_id)
        if not ctx:
            return []
        tasks = [self._public_task(t) for t in ctx.pending_tasks]
        if status == "all":
            return tasks
        return [t for t in tasks if t.get("status", "pending") == status]

    async def complete_task(self, agent_id: str, task_id: str) -> bool:
        return await self._finish_task(agent_id, task_id, "completed")

    async def cancel_task(self, agent_id: str, task_id: str) -> bool:
        return await self._finish_task(agent_id, task_id, "cancelled")

    async def _finish_task(self, agent_id: str, task_id: str, status: str) -> bool:
        """
        完成/取消任务：从 pending_tasks 移除，内容和向量转入 recent_done 用于去重窗口，
        完整的任务记录写入 todo 生命事件（MongoDB），历史不丢
        """
        lock = self._tasks_lock_key(agent_id)
        if not await self._acquire_tasks_lock(lock):
            logger.error(f"{status} task {task_id} failed for {agent_id}: tasks lock busy")
            return False
        try:
            ctx = await self.get_context(agent_id)
            if not ctx:
                return False
            task = next((t for t in ctx.pending_tasks if t.get("task_id") == task_id), None)
            if task is None:
                return False

            now = datetime.now().isoformat()
            ctx.pending_tasks = [t for t in ctx.pending_tasks if t.get("task_id") != task_id]
            ctx.recent_done.append({
                "task_id": task_id,
                "content": task.get("content", ""),
                "vector": task.get("vector"),
                "status": status,
                "done_at": now,
            })
            self._prune_recent_done(ctx)
            # 每条带 1024 维向量（约 22KB），只保留最近 RECENT_DONE_MAX 条，避免 subject_context 过大
            if len(ctx.recent_done) > self.RECENT_DONE_MAX:
                ctx.recent_done.sort(key=lambda d: d.get("done_at", ""))
                ctx.recent_done = ctx.recent_done[-self.RECENT_DONE_MAX:]
            await self.save_context(ctx)
        except Exception as e:
            logger.error(f"{status} task error: {e}", exc_info=True)
            return False
        finally:
            await redis_client.release_lock(lock)

        logger.info(f"Task {status}: {task.get('content')}")
        record = self._public_task(task)
        record.update({"status": status, f"{status}_at": now})
        await agent_life_event_service.record_todo(
            agent_id=agent_id,
            task_id=task_id,
            content=task.get("content", ""),
            action=status,
            task_data=record,
        )
        return True

    async def migrate_finished_tasks(self) -> Dict[str, int]:
        """
        一次性迁移旧数据：旧版本完成的任务仍留在 pending_tasks 里（status=completed/cancelled），
        把它们移入 recent_done（完成时间在去重窗口内的才保留）。幂等，可重复执行
        """
        moved = {}
        for agent_id in await self.list_agents():
            lock = self._tasks_lock_key(agent_id)
            if not await self._acquire_tasks_lock(lock):
                continue
            try:
                ctx = await self.get_context(agent_id)
                if not ctx:
                    continue
                finished = [t for t in ctx.pending_tasks if t.get("status", "pending") != "pending"]
                if not finished:
                    continue
                ctx.pending_tasks = [t for t in ctx.pending_tasks if t.get("status", "pending") == "pending"]
                for t in finished:
                    ctx.recent_done.append({
                        "task_id": t.get("task_id"),
                        "content": t.get("content", ""),
                        "vector": t.get("vector"),
                        "status": t.get("status"),
                        "done_at": t.get("completed_at") or t.get("cancelled_at") or t.get("created_at", ""),
                    })
                self._prune_recent_done(ctx)
                await self.save_context(ctx)
                moved[agent_id] = len(finished)
            finally:
                await redis_client.release_lock(lock)
        if moved:
            logger.info(f"Migrated finished tasks out of pending_tasks: {moved}")
        return moved

# 单例
subjectivity_service = SubjectivityService()
