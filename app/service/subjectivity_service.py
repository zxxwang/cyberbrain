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
import math
import random
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

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

# 整体状态的取值范围：心情 -1（最差）~ 1（最好），压力、精力 0 ~ 1
STATE_RANGE = {"mood": (-1.0, 1.0), "stress": (0.0, 1.0), "energy": (0.0, 1.0)}
# 没有配置基线（旧数据）时用的中性基线
DEFAULT_BASELINE = {"mood": 0.0, "stress": 0.3, "energy": 0.7}


def format_state_lines(state: Dict[str, float], prefix: str = "") -> List[str]:
    """整体状态 → 提示词里的三行，带刻度：心情的范围是 -1 ~ 1，不标出来模型分不清 0.2 是好是坏"""
    return [
        f"{prefix}心情：{state.get('mood', 0.0):.2f}（-1 最差，0 平常，1 最好）",
        f"{prefix}压力：{state.get('stress', 0.3):.2f}（0 ~ 1，越高压力越大）",
        f"{prefix}精力：{state.get('energy', 0.7):.2f}（0 ~ 1，越低越疲惫）",
    ]


def describe_state(state: Optional[Dict[str, float]]) -> str:
    """
    整体状态 → 一句话（"心情不太好，压力很大，有点累"）。给 System1 放在这句话旁边用：
    小模型对数字不敏感、对紧挨着信号的文字敏感，三行数字放在上下文开头时它几乎不用（实测去掉状态，判断不变）
    """
    if not state:
        return ""
    mood, stress, energy = state.get("mood", 0.0), state.get("stress", 0.3), state.get("energy", 0.7)
    m = ("心情很差" if mood <= -0.5 else "心情不太好" if mood <= -0.15 else "心情平常" if mood < 0.15
         else "心情不错" if mood < 0.5 else "心情很好")
    s = "压力很大" if stress >= 0.75 else "有些压力" if stress >= 0.5 else "没什么压力"
    e = "非常疲惫" if energy <= 0.25 else "有点累" if energy <= 0.45 else "精力充沛" if energy >= 0.75 else "精力还行"
    return f"{m}，{s}，{e}"


def _clamp(key: str, value: float) -> float:
    lo, hi = STATE_RANGE[key]
    return max(lo, min(hi, float(value)))


def _diminish(key: str, current: float, delta: float) -> float:
    """
    边际递减：增量乘以"离边界还剩多少"（占整个取值范围的比例）。
    越接近边界越难再往前推，往回拉则几乎不受影响，所以连续同向刺激会饱和，不会几句话就贴边
    """
    lo, hi = STATE_RANGE[key]
    room = (hi - current) if delta > 0 else (current - lo)
    return delta * max(0.0, room) / (hi - lo)


class SubjectContext:
    """主体性上下文：Agent的主观世界"""

    def __init__(
        self,
        agent_id: str,
        situation: Dict = None,
        thoughts: List[Dict] = None,
        overall_state: Dict = None,
        pending_tasks: List[Dict] = None,
        recent_done: List[Dict] = None,
        baseline: Dict = None,
        state_updated_at: Optional[str] = None,
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
        # 整体状态的基线（这个人平时的状态）：情绪会随时间回到这里。由初始化/更新接口设定
        self.baseline = baseline or dict(DEFAULT_BASELINE)
        # 整体状态最后一次按时间结算的时刻。和 last_update 分开：last_update 任何保存都会刷新（如加一条待办），
        # 用它算"过了多久"会把回归计时错误地清零
        self.state_updated_at = state_updated_at or datetime.now().isoformat()
        self.last_update = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "situation": self.situation,
            "thoughts": self.thoughts,
            "overall_state": self.overall_state,
            "pending_tasks": self.pending_tasks,
            "recent_done": self.recent_done,
            "baseline": self.baseline,
            "state_updated_at": self.state_updated_at,
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
            recent_done=data.get("recent_done", []),
            baseline=data.get("baseline"),
            # 旧数据没有这个字段：从读到的这一刻开始计时，不对过去补算
            state_updated_at=data.get("state_updated_at"),
        )


# ============================================================================
# Service
# ============================================================================

# 每类处境最多展示几条：关注点和事件太多会撑大 prompt，也会稀释真正重要的那几件
SITUATION_MAX_ITEMS = 3


def _event_due(event: Dict[str, Any], today) -> Optional[tuple]:
    """(deadline, 剩余天数)；无 deadline、格式错误或已过期返回 None"""
    try:
        deadline = datetime.strptime(str(event.get("deadline")), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    days_left = (deadline - today).days
    return (deadline, days_left) if days_left >= 0 else None


def render_situation(situation: Optional[Dict[str, Any]]) -> str:
    """
    把客观处境渲染成 prompt 文本。System1/2（format_for_prompts）和念头生成（format_situation_for_prompt）共用，
    以前两处各写一份，System1/2 那份漏了 attention_focus、也不过滤过期事件。

    - 时间敏感事件：过期、无日期或日期格式错误的不展示；按截止日期由近到远取前 3 条，
      每条标上"今天 / 明天 / 还有 N 天"——只给日期时，模型分不清这件事是过去还是将来
    - 持续状态、当前关注：各最多 3 条
    """
    if not situation:
        return ""
    lines: List[str] = []
    if situation.get("current_activity"):
        lines.append(f"当前活动：{situation['current_activity']}")

    today = datetime.now().date()
    upcoming = []
    for e in situation.get("time_sensitive_events") or []:
        due = _event_due(e, today)
        desc = e.get("description") or e.get("event")
        if due and desc:
            upcoming.append((due[0], due[1], desc))
    upcoming.sort(key=lambda x: x[0])
    if upcoming:
        lines.append("时间敏感事件：")
        for deadline, days_left, desc in upcoming[:SITUATION_MAX_ITEMS]:
            when = "今天" if days_left == 0 else "明天" if days_left == 1 else f"还有{days_left}天"
            lines.append(f"- {desc}（{deadline.strftime('%m月%d日')}，{when}）")

    states = [s for s in situation.get("ongoing_states") or [] if s][:SITUATION_MAX_ITEMS]
    if states:
        lines.append(f"持续状态：{'、'.join(states)}")

    focus = [f for f in situation.get("attention_focus") or [] if f][:SITUATION_MAX_ITEMS]
    if focus:
        lines.append(f"当前关注：{'、'.join(focus)}")
    return "\n".join(lines)


def settle_state(context: "SubjectContext", now: Optional[datetime] = None) -> None:
    """
    把"上次结算到现在"这段时间的变化补算进整体状态（就地修改，不保存）：
    - 心情、压力：指数回归到基线，x += (基线 - x)·(1 - e^(-小时/τ))，τ = STATE_REGRESSION_TAU_HOURS
    - 精力：向基线恢复，压力越大恢复越慢（τ 乘以 1 + 压力）

    读取时补算，不需要定时任务：没人读的值不必实时更新，读的时候算出来就是对的。
    结算后把 state_updated_at 改成 now：之后若保存，保存的就是结算后的值和对应时刻；不保存，下次从原时刻重算，
    结果一样（指数回归分段算和一次算相同）
    """
    now = now or datetime.now()
    try:
        last = datetime.fromisoformat(context.state_updated_at)
    except (TypeError, ValueError):
        context.state_updated_at = now.isoformat()
        return
    hours = (now - last).total_seconds() / 3600
    if hours <= 0:
        return
    state, base = context.overall_state, context.baseline
    k = 1 - math.exp(-hours / settings.STATE_REGRESSION_TAU_HOURS)
    for key in ("mood", "stress"):
        cur = state.get(key, base.get(key, DEFAULT_BASELINE[key]))
        state[key] = round(_clamp(key, cur + (base.get(key, DEFAULT_BASELINE[key]) - cur) * k), 4)
    energy = state.get("energy", base.get("energy", DEFAULT_BASELINE["energy"]))
    tau_e = settings.ENERGY_RECOVERY_TAU_HOURS * (1 + state.get("stress", 0.3))
    k_e = 1 - math.exp(-hours / tau_e)
    state["energy"] = round(_clamp("energy", energy + (base.get("energy", DEFAULT_BASELINE["energy"]) - energy) * k_e), 4)
    context.state_updated_at = now.isoformat()


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
            context = SubjectContext.from_dict(json.loads(data))
        except Exception as e:
            logger.error(f"Failed to parse subject context for {agent_id}: {e}")
            return None
        # 读到的整体状态先按流逝的时间结算（情绪回落、精力恢复）
        settle_state(context)
        return context

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
            state_lines.extend(format_state_lines(context.overall_state))

        state_only = "\n".join(state_lines) if state_lines else ""

        # Situation（客观处境）：与念头生成共用 render_situation，保证两边看到的处境一致
        body = render_situation(context.situation)
        situation_str = f"\n## 客观处境\n{body}" if body else ""

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
        """格式化situation用于念头生成等 prompt（与 System1/2 使用同一个 render_situation）"""
        return render_situation(await self.get_situation(agent_id))

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
        # target_person 必须是关系表里的名字，否则无法据此主动联系对方
        from app.repository.relationship_repository import relationship_repository
        people = await relationship_repository.load_all(agent_id)
        people_str = "、".join(f"{n}（{r['relationship']}）" for n, r in people.items()) or "（无）"
        state_block = "\n".join(format_state_lines(context.overall_state, prefix="                    - "))
        prompt = f"""你是这个Agent的内心世界。

                    现在是 {now.strftime('%Y年%m月%d日 %H:%M')}。

                    【客观处境】
                    {situation_str}

                    【整体状态】
{state_block}

                    【我认识的人】{people_str}

                    此刻内心会冒出什么念头？（内心独白，20字以内）
                    有没有想要做的事或想找的人？想找的人只能从【我认识的人】里选，写名字原文

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

            # 模型偶尔在 JSON 前后带说明文字：只取第一个 { 到最后一个 }；仍然解析失败就放弃这次念头（不影响对话）
            if "{" in response and "}" in response:
                response = response[response.index("{"): response.rindex("}") + 1]
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                logger.warning(f"Thought skipped for {agent_id}: LLM returned invalid JSON ({e})")
                return None
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
        energy: float,
        set_baseline: bool = True,
    ):
        """
        直接设置整体状态（初始化 / 管理接口）。默认同时设为基线：初始化和人工设定的都是"这个人平时的状态"，
        情绪波动之后会随时间回到这里

        Args:
            mood: 心情 [-1, 1]；stress、energy: [0, 1]
        """
        context = await self.get_context(agent_id)
        if not context:
            context = SubjectContext(agent_id=agent_id)
        state = {"mood": _clamp("mood", mood), "stress": _clamp("stress", stress), "energy": _clamp("energy", energy)}
        context.overall_state = dict(state)
        if set_baseline:
            context.baseline = dict(state)
        context.state_updated_at = datetime.now().isoformat()
        await self.save_context(context)
        logger.info(f"Set overall_state for {agent_id}: {state} (baseline={'yes' if set_baseline else 'no'})")

    async def adjust_overall_state(
        self,
        agent_id: str,
        mood_delta: float = 0.0,
        stress_delta: float = 0.0,
        energy_delta: float = 0.0,
        reason: str = ""
    ):
        """
        按增量调整整体状态（对话情绪、处境变化）。

        - 先按流逝的时间结算（get_context 已做），再加增量
        - 增量做边际递减：越接近边界越难再推（见 _diminish）
        - 精力独立变化：只由 energy_delta（对话消耗）和时间恢复决定，不再由压力换算

        Returns:
            {"before": 调整前（已按时间结算）, "after": 调整后}
        """
        context = await self.get_context(agent_id)
        if not context:
            context = SubjectContext(agent_id=agent_id)
        state = context.overall_state
        before = dict(state)
        for key, delta in (("mood", mood_delta), ("stress", stress_delta), ("energy", energy_delta)):
            if not delta:
                continue
            cur = state.get(key, context.baseline.get(key, DEFAULT_BASELINE[key]))
            state[key] = round(_clamp(key, cur + _diminish(key, cur, delta)), 4)
        context.state_updated_at = datetime.now().isoformat()
        await self.save_context(context)
        logger.info(f"Adjusted overall_state for {agent_id}: {before} -> {dict(state)} ({reason})")
        return {"before": before, "after": dict(state)}

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
        lines.extend(format_state_lines(state, prefix="- "))
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
            # 没有联网能力的 agent（如拟人角色），漏斗里不问"要不要联网 / 要不要现在就做"，任务里两项都是 False
            from app.repository.capability_repository import CAP_WEB_SEARCH, capability_repository
            can_web = await capability_repository.has(agent_id, CAP_WEB_SEARCH)
            if source == "thought":
                result = await self._thought_funnel(content, ctx, can_web)
            elif source == "dialogue":
                result = await self._dialogue_funnel(content, ctx, can_web)
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
                    # 是否需要联网搜索补充信息：由漏斗里的决策模型判断，执行阶段据此走联网搜索
                    "needs_web_search": bool(result.get("needs_web_search")),
                    # 对话中判断为"现在就该做"：写入后由调用方立即执行（只限有真实执行能力的任务，见 task_executor）
                    "execute_now": bool(result.get("execute_now")),
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

    # 是否需要联网：和漏斗里已有的问题放在同一次 System One 请求里，不额外增加调用
    NEEDS_WEB_SEARCH_QUESTION = {
        "type": "choice",
        "instructions": "完成这件事，是否需要上网搜索、查资料来获取我现在还不知道的信息？",
        "criteria": {
            "yes": "需要查询、了解、研究、核实某个知识或事实，或需要最新的信息（新闻、价格、政策、研究进展等）",
            "no": "做事、联系人、提醒、整理已有的东西，或凭已知的就能完成",
        },
    }

    @staticmethod
    def _yes_prob(answers: Dict, key: str) -> float:
        """
        System One 选择题里"yes"的概率。注意不能用返回的 confidence：它表示判断有多确定（约为 1 - 熵），
        不是"yes 的概率"。例如 yes 概率 0.755 时 confidence 只有 0.197
        """
        a = answers.get(key, {})
        probs = a.get("probabilities") or {}
        if "yes" in probs:
            return float(probs["yes"])
        return 1.0 if a.get("choice") == "yes" else 0.0

    @staticmethod
    def _needs_web_search(answers: Dict) -> bool:
        return answers.get("needs_web_search", {}).get("choice", "no") == "yes"

    async def _thought_funnel(self, content: str, ctx: Dict, can_web: bool = False) -> Optional[Dict]:
        """
        念头 → 待办：一次 System One 请求同时判断"是否转待办"和"是否需要联网"（只在有联网能力时问）。
        yes 概率 ≥ 0.8 直接创建；0.6–0.8 再交给 LLM 确认；更低不创建。

        原来的第一层是关键词匹配（得、要、说、做、看……），单字关键词几乎匹配所有念头，
        命中就直接创建、不经过决策，是待办重复的主要来源，已去掉
        """
        questions = {
            "need_task": {
                "type": "choice",
                "instructions": "这个念头是否需要转为待办任务？",
                "criteria": {
                    "yes": "有想去做的具体事情：找某人、提醒某人、做某件事，或者想去了解、查明、补充某个具体的知识或信息",
                    "no": "纯情绪、感慨、抽象担忧、对已经发生的事的回味，没有要去做的事"
                }
            },
        }
        if can_web:
            questions["needs_web_search"] = self.NEEDS_WEB_SEARCH_QUESTION
        decide_result = await doubao_client.decide(
            state=f"念头内容：{content}\n强度：{ctx.get('intensity', 0.5):.2f}",
            questions=questions,
        )
        answers = decide_result.get("answers", {})
        p_yes = self._yes_prob(answers, "need_task")
        needs_web = self._needs_web_search(answers)

        if p_yes >= 0.8:
            logger.info(f"Thought L1(p_yes={p_yes:.2f}, web={needs_web}): {content[:20]}")
            return {"content": content, "priority": round(p_yes, 2), "needs_web_search": needs_web,
                    "related_person": self._extract_person(content)}
        if p_yes >= 0.6:
            logger.info(f"Thought L2(LLM, p_yes={p_yes:.2f}, web={needs_web}): {content[:20]}")
            llm = await self._llm_judge_thought(content, ctx)
            if llm.get("should_add"):
                return {"content": llm.get("task_content") or content, "priority": llm.get("priority", 0.6),
                        "related_person": llm.get("related_person"), "needs_web_search": needs_web}
        logger.debug(f"Thought reject: {content[:20]}")
        return None

    async def _dialogue_funnel(self, content: str, ctx: Dict, can_web: bool = False) -> Optional[Dict]:
        """
        对话待办 2 层：System One 粗判是否有承诺 → 置信度 > 0.5 时交给 lite 模型
        再确认一次是否有任务、把原话提炼成任务描述、按语义推出截止日期

        needs_web_search / execute_now 只在有联网能力时问：目前能立即执行的只有联网任务，
        没有联网能力时两项都没有意义，答案里没有这两项，解析结果即为 False
        """
        signal, response, speaker = ctx.get("signal", content), ctx.get("response", ""), ctx.get("speaker", "用户")
        history = ctx.get("recent_dialogue", "")

        questions = {
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
            },
        }
        if can_web:
            questions["needs_web_search"] = self.NEEDS_WEB_SEARCH_QUESTION
            questions["execute_now"] = {
                "type": "choice",
                "instructions": "这件事是不是现在就该马上着手去做？",
                "criteria": {
                    "yes": "对方要求现在、马上去做；或者没提时间，而且是立刻就能动手的事（如查资料、核实信息、搜索）",
                    "no": "指定了以后的时间（今晚、明天、下周、某个日期之前），或者要等某个条件满足，或者是到时候的提醒",
                },
            }
        decide_result = await doubao_client.decide(
            state=(f"刚才的对话：\n{history}\n\n此刻：" if history else "") + f"{speaker}：{signal} Agent：{response}",
            questions=questions,
        )
        answers = decide_result.get("answers", {})
        # 用 yes 概率而不是 confidence（见 _yes_prob）。原来的 confidence > 0.5 约等于 yes 概率 ≥ 0.89，这里保持同样严格
        p_commit = self._yes_prob(answers, "has_commitment")
        needs_interaction = answers.get("needs_interaction", {}).get("choice", "no")
        needs_web = self._needs_web_search(answers)
        execute_now = answers.get("execute_now", {}).get("choice", "no") == "yes"
        logger.info(f"🔍 Dialogue funnel: p_commit={p_commit:.2f}, needs_interaction={needs_interaction}, "
                    f"needs_web_search={needs_web}, execute_now={execute_now}, signal={signal[:30]}")

        if p_commit < 0.85:
            return None

        extracted = await self._llm_extract_dialogue_task(signal, response, speaker, history)
        if not extracted:
            logger.info(f"Dialogue funnel: LLM found no task, signal={signal[:30]}")
            return None

        task = {
            "content": extracted["content"],
            "priority": 0.8,
            "deadline": extracted["deadline"],
            "related_person": speaker,
            "needs_web_search": needs_web,
            "execute_now": execute_now,
        }
        if needs_interaction == "yes":
            task["interaction_type"] = "dialogue"
        logger.info(f"Dialogue task (p_commit={p_commit:.2f}): {task['content']} deadline={task['deadline']}")
        return task

    # 提示词尽量短、输出只有一两行纯文本（不用 JSON），减少输入和输出 token
    # 只提炼，不再判断"有没有任务"：System One 已判断有承诺（yes 概率 ≥ 0.85）。
    # 实测让 lite 再确认一次反而不稳：同一输入、温度 0，约 1/4 的概率回答"无"，导致明确的承诺被漏掉
    DIALOGUE_TASK_PROMPT = """今天是{today}（{weekday}）。
刚才的对话：
{history}

此刻：
{speaker}：{signal}
我：{response}

我答应了要去做一件事。输出两行，不加任何前缀：
任务描述（不照抄原话，写清做什么；这句话省略了要做的事时，从刚才的对话里补全具体对象，30字内）
截止日期 YYYY-MM-DD（按话里的时间推算；没提时间写：无）"""

    WEEKDAYS = "一二三四五六日"

    async def _llm_extract_dialogue_task(self, signal: str, response: str, speaker: str, history: str = "") -> Optional[Dict]:
        """lite 模型提炼任务描述、推算截止日期。调用失败或输出为空时返回 None"""
        today = datetime.now()
        prompt = self.DIALOGUE_TASK_PROMPT.format(
            today=today.strftime("%Y-%m-%d"), weekday=f"星期{self.WEEKDAYS[today.weekday()]}",
            speaker=speaker, signal=signal, response=response or "（没有回应）",
            history=history or "（无）",
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
        # 偶尔会把日期写在描述开头（"2026-11-15提醒……"），去掉
        lines[0] = re.sub(r"^\d{4}-\d{2}-\d{2}\s*", "", lines[0]).strip()

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
