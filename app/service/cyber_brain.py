"""
Cyber Brain: 整合各脑区能力的最终模块

react() 是大脑接收信号、返回响应的主方法：
1. 召回brief（信条/事实），System1(tev1)给出本能反应：情绪、冲动、投入意愿，
   以及认知门控（对方意图、直觉行为、是否要回忆、是否要深思）
2. 按门控选择认知路径：
   - fast：直觉行为直接交给System3说出口
   - memory：带上想起的往事，交给System3
   - deliberate：System2结合人格、记忆、技能做社交决策（不说话），再交给System3
3. System3：把已决定的行为转成"我"的口语，只管怎么说，不加内容

执行上 brief→System1 与 episodic、knowledge预取、工作记忆并行；fast路径不等episodic。
"""
import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.client.doubao_client import doubao_client, start_usage_collection
from app.client.embedding_client import embedding_client
from app.config.prompts import (
    SYSTEM2_PROMPT, SYSTEM2_USER_TEMPLATE, SYSTEM3_PROMPT, SYSTEM3_USER_TEMPLATE,
)
from app.config.settings import settings
from app.repository.memory_repository import memory_repository
from app.repository.relationship_repository import relationship_repository
from app.service.memory_gate_service import memory_gate_service
from app.service.subjectivity_service import subjectivity_service
from app.service.persona_manager import persona_manager
from app.service.agent_life_event_service import agent_life_event_service
from app.service.prefrontal_lobe_memory import prefrontal_lobe_memory, format_records
from app.service.subjectivity_service import subjectivity_service
from app.service.knowledge_memory_service import knowledge_memory_service
from app.service.system1_tev1 import system1_tev1, QUICK_ACTIONS

logger = logging.getLogger(__name__)

# System1解析失败时的中性情绪；认知门控取保守值，保证走System2兜底
NEUTRAL_EMOTION = {
    "emotion": "平静", "intensity": 0.0, "impulse": "无", "engagement": 0.5,
    "user_intent": "其他", "quick_action": "随口接话", "action_confidence": 0.0, "think_prob": 1.0,
    "needs_memory": 0.5, "needs_reasoning": 1.0,
}

# 认知路径
ROUTE_FAST = "fast"            # System1 -> System3：凭直觉接话
ROUTE_MEMORY = "memory"        # System1 -> 回忆 -> System3：想起点事，但不用深思
ROUTE_DELIBERATE = "deliberate"  # System1 -> 回忆 -> System2 -> System3：得想清楚

# System2说话时写入PFC的身份标识
SELF_SPEAKER = "我"
# 选择不说话时写入工作记忆的内容，下一轮知道自己刚才没接话
SILENCE_RECORD = "（没有接话）"

# 括号包裹的动作/神态/语气旁白，如"（叹气）""(语气带着火气)"
STAGE_DIRECTION = re.compile(r"[（(][^（）()]{1,30}[）)]")

# 谎言标记：System2在撒谎的句子末尾加[谎言]（兼容全角括号）
LIE_TAG = re.compile(r"\s*[\[【]谎言[\]】]")
# 句子切分：按中英文句末标点，保留标点在句内
SENTENCE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]*")


def parse_lies(text: str) -> Tuple[str, List[str]]:
    """
    解析System2输出中的[谎言]标记

    Returns:
        (去掉标记后说出口的话, 被标记为谎言的句子列表)
    """
    lies = []
    for sentence in SENTENCE.findall(text):
        if LIE_TAG.search(sentence):
            clean = LIE_TAG.sub("", sentence).strip()
            if clean:
                lies.append(clean)
    return LIE_TAG.sub("", text).strip(), lies


def _format_memories(memories: List[Dict[str, Any]], empty: str) -> str:
    lines = [f"- {m.get('memory', '')}" for m in memories if m.get("memory")]
    return "\n".join(lines) if lines else empty


def _format_knowledge(knowledge_list: List[Dict[str, Any]], empty: str) -> str:
    """技能payload是title/content，不是memory字段"""
    blocks = [f"《{s['title']}》\n{s.get('content', '')[:settings.BRAIN_KNOWLEDGE_CONTENT_CHARS]}"
              for s in knowledge_list if s.get("title")]
    return "\n\n".join(blocks) if blocks else empty


# 信号几乎不含中文时视为外语，单独提醒System2用同一语言回复（中文上下文会压过提示词里的语言规则）
CJK = re.compile(r"[一-鿿]")


def _language_hint(signal: str) -> str:
    if len(CJK.findall(signal)) <= len(signal) * 0.1:
        return "\n\n（对方用的不是中文，用对方的语言回复）"
    return ""




def _format_engagement_state(engagement: float) -> str:
    """将投入意愿分数映射为自然语言描述"""
    if engagement < 0.2:
        return "完全不想聊"
    elif engagement < 0.4:
        return "只想应付一两句"
    elif engagement < 0.6:
        return "正常交流"
    elif engagement < 0.8:
        return "愿意多聊几句"
    else:
        return "话匣子打开了"


def _format_conflict_state(conflict: float) -> str:
    """将话题冲突程度分数映射为自然语言描述"""
    if conflict < 0.2:
        return "完全无冲突"
    elif conflict < 0.4:
        return "轻微分歧"
    elif conflict < 0.6:
        return "有点不同意"
    elif conflict < 0.8:
        return "明显冲突"
    else:
        return "触犯底线"


def _format_instinct(emotion: Dict[str, Any]) -> str:
    """System1本能反应的自然语言版本（不给VAD数值，数字会被LLM当成要'体现'的指标）"""
    return (f"情绪：{emotion.get('emotion', '平静')}（强度 {emotion.get('intensity', 0.0):.1f}）\n"
            f"第一冲动：{emotion.get('impulse', '无')}\n"
            f"投入意愿：{_format_engagement_state(emotion.get('engagement', 0.5))}")


def _trace_memories(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """召回结果的精简版（id / 分数 / 文本），不带向量和 payload 冗余字段"""
    out = []
    for m in memories:
        score = m.get("rerank_score", m.get("score"))
        out.append({
            "id": m.get("id"),
            "score": round(score, 4) if isinstance(score, (int, float)) else None,
            "text": m.get("memory") or m.get("title") or m.get("content") or "",
        })
    return out


def _build_react_trace(
    *, speaker_info: str, lies: List[str], silent: bool,
    route: str, route_reason: str, emotion: Dict[str, Any], decision: Optional[Dict[str, Any]], requires_knowledge: bool,
    brief: List[Dict[str, Any]], episodic: List[Dict[str, Any]], knowledge: List[Dict[str, Any]],
    overall_state: Optional[Dict[str, Any]], working_memory_chars: int,
    timings: Dict[str, float], llm_usage: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    一轮 react 的完整中间状态（与测试报告的观测维度一致），写入 react_output 事件的顶层 trace 字段：
    System1 全字段、路由、System2 决策、召回（含分数）、对话前 overall_state、各阶段耗时、LLM token。
    signal / speaker / response 已在同一条事件的 data 里，这里不重复
    """
    return {
        "speaker_info": speaker_info,
        "lies": lies,
        "silent": silent,
        "route": route,
        "route_reason": route_reason,
        "requires_knowledge": requires_knowledge,
        "system1": dict(emotion),
        "system2": decision,
        "recall": {
            "brief": _trace_memories(brief),
            "episodic": _trace_memories(episodic),
            "knowledge": _trace_memories(knowledge),
        },
        # 对话前的状态；对话引起的变化由异步写入的 emotion 事件记录（metadata.trace_id 关联）
        "overall_state_before": dict(overall_state) if overall_state else None,
        "working_memory_chars": working_memory_chars,
        "timings_ms": {k: round(v, 1) for k, v in timings.items()},
        "llm_usage": list(llm_usage),
    }


# 对抗性的直觉行为：对认识的人选了这些、话题又没有冲突，多半是 System1 理解错了
HOSTILE_ACTIONS = {"直接拒绝", "顶回去"}


def _recent_dialogue(records: List[Dict[str, Any]], turns: int) -> str:
    """工作记忆末尾的几条原文（压缩块和当日摘要不算），给 System1 理解省略回答"""
    if turns <= 0:
        return ""
    raw = [r for r in records if r.get("type") == "raw"][-turns:]
    return format_records(raw)


def _choose_route(emotion: Dict[str, Any], requires_knowledge: bool, speaker_known: bool = False) -> Tuple[str, str]:
    """
    认知路径选择，返回 (路径, 原因)。

    - 要不要深思（System2）看门控：需要推理，或直觉上"得想一想"的倾向占多数
    - 直觉拿不准（直觉行为置信度低）时不走 fast：一个接近随机的选择不应该直接变成回答
    - 对关系表里的人选了拒绝/顶回去、但话题几乎没有冲突：多半是 System1 理解错了（如没看懂省略回答），交给 System2 复核
    - 知识命中只决定"带不带知识"，不再强制进 System2；但命中知识的话题最低走 memory 路径，
      保证把相关经历一起召回（fast 会取消 episodic，聊书时容易变浅或编造）
    """
    if emotion.get("needs_reasoning", 1.0) >= settings.BRAIN_ROUTE_REASONING_THRESHOLD:
        return ROUTE_DELIBERATE, "needs_reasoning"
    if emotion.get("think_prob", 1.0) >= settings.BRAIN_ROUTE_THINK_THRESHOLD:
        return ROUTE_DELIBERATE, "think_prob"
    if (speaker_known and emotion.get("quick_action") in HOSTILE_ACTIONS
            and emotion.get("conflict", 1.0) < settings.BRAIN_ROUTE_HOSTILE_MAX_CONFLICT):
        return ROUTE_DELIBERATE, "hostile_to_known"
    if emotion.get("action_confidence", 0.0) < settings.BRAIN_ROUTE_MIN_ACTION_CONFIDENCE:
        return ROUTE_DELIBERATE, "low_action_confidence"
    if requires_knowledge:
        return ROUTE_MEMORY, "knowledge"
    if emotion.get("needs_memory", 0.0) >= settings.BRAIN_ROUTE_MEMORY_THRESHOLD:
        return ROUTE_MEMORY, "needs_memory"
    return ROUTE_FAST, "intuitive"


def _quick_plan(
    emotion: Dict[str, Any],
    episodic: Optional[List[Dict[str, Any]]] = None,
    knowledge: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    fast/memory路径的行为计划：直接来自System1的直觉行为。
    memory路径是门控判断"要接这句话必须想起具体往事"，所以想起的事是回应的内容本身，不是可选点缀。
    知识不再强制走 System2，所以命中的知识也要放进计划，否则 System3 拿不到
    """
    action = emotion.get("quick_action", "随口接话")
    lines = [f"对方是在：{emotion.get('user_intent', '其他')}",
             f"我的第一反应：{action}（{QUICK_ACTIONS.get(action, '')}）"]
    if episodic:
        lines.append("对方说的这件事，我记得的是（回应要落在这些具体的事上；这里没有的细节就是记不清了）：\n"
                     + _format_memories(episodic[:3], ""))
    if knowledge:
        lines.append("我对这个话题的了解（用自己的话自然带出，不要背书）：\n"
                     + _format_knowledge(knowledge, ""))
    if not episodic and not knowledge:
        lines.append("凭直觉回应，根据行为提示、对话上下文、我和对方的关系，给出符合我性格的自然回答。")
    # fast/memory 路径没有 System2 的事实检查，只能在这里约束：别把"可以帮你做"说成"已经做了"
    lines.append("只能说已经知道的事。没有真正做过的事（联系了谁、发了什么、安排好了什么）不能说成已经做了；"
                 "可以说打算去做，或者问对方要不要做。")
    if emotion.get("user_intent") == "提问":
        # 不经过 System2 时，System1 的"反问对方"会让 System3 只抛回一句"你觉得呢？"而不回答
        lines.append("对方在问我问题：先正面回答（用上面记得的事和了解的东西），想反问也放在回答之后。")
    return "\n".join(lines)


# System2 按编号输出（不写字段名，省输出token），编号 -> 内部键；编号定义见 SYSTEM2_PROMPT【输出】
DECISION_FIELDS = {
    "1": "sanity", "2": "thought", "3": "action", "4": "core", "5": "extra",
    "6": "withhold", "7": "manner", "8": "lies", "9": "source",
}
# 兼容模型偶尔仍带字段名的写法："4 核心意思：xxx" / "4. xxx" / "4、xxx"
DECISION_FIELD_NAMES = ("这话合理吗", "心里想", "怎么做", "核心意思", "补充", "不说出口", "分寸", "假话", "事实来源")
DECISION_LINE = re.compile(r"^\s*([1-9])\s*[.、:：)）]?\s*(.*)$", re.M)
DECISION_NAME_PREFIX = re.compile(rf"^({'|'.join(DECISION_FIELD_NAMES)})\s*[:：]\s*")
NONE_VALUES = {"", "无", "没有", "无。", "-"}
SILENCE_WORDS = ("不说话", "沉默", "不吭声", "不搭理", "不理会", "不回应")


def _parse_decision(text: str) -> Optional[Dict[str, Any]]:
    """解析System2的编号决策；缺"怎么做"(3)且缺"核心意思"(4)时视为失败。省略的可选项按空处理"""
    decision = {}
    for num, value in DECISION_LINE.findall(text):
        key = DECISION_FIELDS[num]
        if key in decision:
            continue  # 同一编号重复出现时取第一次
        value = DECISION_NAME_PREFIX.sub("", value.strip()).strip()
        decision[key] = "" if value in NONE_VALUES else value
    if "core" not in decision and "action" not in decision:
        return None
    decision["lies"] = [x.strip() for x in re.split(r"[；;]", decision.get("lies", "")) if x.strip()]
    core = decision.get("core", "")
    decision["silent"] = (not core or any(w in core for w in SILENCE_WORDS)) and \
        any(w in decision.get("action", "") + core for w in SILENCE_WORDS)
    return decision


def _format_decision(decision: Dict[str, Any]) -> str:
    """System2决策 -> System3的行为计划。"心里想"只服务于决策，不交给System3，避免它把理由都说出来"""
    lines = [f"我决定：{decision.get('action', '')}",
             f"核心意思：{decision.get('core') or '（无）'}"]
    if decision.get("extra"):
        lines.append(f"补充（可说可不说）：{decision['extra']}")
    if decision.get("withhold"):
        lines.append(f"心里有但不说出口：{decision['withhold']}")
    if decision.get("manner"):
        lines.append(f"分寸：{decision['manner']}")
    if decision.get("lies"):
        lines.append(f"要说的假话：{'；'.join(decision['lies'])}")
    if decision.get("source"):
        lines.append(f"事实来源：{decision['source']}")
    return "\n".join(lines)


class CyberBrain:
    def __init__(self):
        # 后台任务的引用：asyncio 只持有弱引用，不保存的话任务可能执行到一半被 GC 回收
        self._background: set = set()
        self.knowledge_top_k = getattr(settings, 'BRAIN_KNOWLEDGE_TOP_K', 5)  # 默认召回5个知识

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _format_speaker_info(self, speaker_name: Optional[str], agent_id: str) -> Tuple[str, bool]:
        """
        格式化说话人信息：
        1. relationship 表有记录 → 按关系描述
        2. working memory（昨天+今天）出现过 → "有过简单接触的人"
        3. 都不在 → "一个不认识的纯陌生人"（强防备）

        Returns:
            (说话人描述, 是否在关系表里)。后者给路由用：对认识的人选了拒绝类直觉行为时要复核
        """
        if not speaker_name:
            return "一个不认识的纯陌生人", False

        # 1. 查 relationship 表
        try:
            relationship = await relationship_repository.get_relationship(agent_id, speaker_name)
            if relationship:
                return relationship, True
        except Exception as e:
            logger.warning(f"Relationship lookup failed for {speaker_name}: {e}")

        # 2. 查 working memory（昨天+今天的 compress/daily/raw）
        try:
            wm_records = await prefrontal_lobe_memory.get_records(agent_id)
            for record in wm_records:
                if record.get("type") == "raw":
                    # raw 记录直接查 speaker 字段
                    if record.get("speaker") == speaker_name:
                        return "有过简单接触的人", False
                else:
                    # compress/daily 查 speakers 列表
                    if speaker_name in record.get("speakers", []):
                        return "有过简单接触的人", False
        except Exception as e:
            logger.warning(f"Working memory lookup failed: {e}")

        # 3. 纯陌生人
        return "一个不认识的纯陌生人", False

    async def react(
        self,
        agent_id: str,
        signal: str,
        speaker: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        大脑主方法：接收外部信号，返回响应

        Args:
            agent_id: 大脑（agent）标识
            signal: 外部输入信号
            speaker: 说话人姓名，用于关系查询、情绪反应和工作记忆标记

        Returns:
            {"response": 说出口的话（不含标记，沉默时为空）, "lies": 其中被标记为谎言的句子, "silent": 是否选择不说话,
             "route": fast/memory/deliberate, "emotion": System1本能反应与认知门控,
             "decision": System2社交决策（仅deliberate路径）, "brief": [...], "episodic": [...],
             "knowledge": [...], "requires_knowledge": bool, "timings_ms": 各阶段耗时}
        """
        react_start = time.perf_counter()
        llm_usage = start_usage_collection()  # 本轮所有 chat 调用的 token 用量，写入 react_output 的 trace

        # 🆕 记录 react 输入事件（听到的话）
        await agent_life_event_service.record_react_input(
            agent_id=agent_id,
            signal=signal,
            speaker=speaker or "user"
        )

        timings: Dict[str, float] = {}

        # === 前置步骤：串行执行，避免重复IO和竞态条件 ===
        # 1. 获取说话人关系信息
        speaker_info, speaker_known = await self._format_speaker_info(speaker, agent_id)

        # 2. 获取主体性上下文（读一次，按系统分发：System1/2 状态+处境，System3 只给状态）
        t_subject = time.perf_counter()
        subject_context = await subjectivity_service.get_context(agent_id)
        state_and_situation, state_only = subjectivity_service.format_for_prompts(subject_context)
        timings["subjectivity"] = (time.perf_counter() - t_subject) * 1000

        # 3. 写入新信号到工作记忆，并获取完整上下文
        t_pfc = time.perf_counter()
        await prefrontal_lobe_memory.append(agent_id, signal, role="user", speaker=speaker or "user")
        wm_records = await prefrontal_lobe_memory.get_records(agent_id)
        working_memory = format_records(wm_records)
        # System1 只给最近几条原文（不含刚写入的本句），控制长度，又能理解"需要""好的"这类省略回答
        recent_dialogue = _recent_dialogue(wm_records[:-1], settings.BRAIN_SYSTEM1_RECENT_TURNS)
        timings["pfc"] = (time.perf_counter() - t_pfc) * 1000

        # === 并行执行记忆召回和情绪分析 ===
        async def _brief_then_system1():
            """Brief召回后接System1情绪分析（System1依赖brief，两者串行）"""
            t0 = time.perf_counter()
            brief = await memory_repository.search_brief(agent_id, signal, top_k=settings.BRAIN_BRIEF_TOP_K)
            timings["brief"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            emotion = await self._system1(agent_id, brief, signal, speaker_info, state_and_situation, recent_dialogue)
            timings["system1"] = (time.perf_counter() - t0) * 1000
            return brief, emotion

        async def _episodic():
            """Episodic记忆召回"""
            t0 = time.perf_counter()
            episodic = await memory_repository.search_episodic(agent_id, signal, top_k=settings.BRAIN_EPISODIC_TOP_K)
            timings["episodic"] = (time.perf_counter() - t0) * 1000
            return episodic

        async def _knowledge_prefetch():
            """技能召回（预取，无论是否需要都先召回）"""
            t0 = time.perf_counter()
            knowledge = await self._knowledge_recall(agent_id, signal)
            timings["knowledge_prefetch"] = (time.perf_counter() - t0) * 1000
            return knowledge

        total_start = time.perf_counter()

        # 先算一次查询向量写入缓存，brief / episodic / knowledge 三路召回共用，不再各自远程 embedding
        t0 = time.perf_counter()
        try:
            await asyncio.to_thread(embedding_client.embed_query, signal)
        except Exception as e:
            # 预热失败不阻断，三路召回会各自重试 embedding
            logger.warning(f"Query embedding prefetch failed: {e}")
        timings["embedding"] = (time.perf_counter() - t0) * 1000

        # episodic 单独起任务：deliberate/memory 路径才等它，fast 路径直接取消
        episodic_task = asyncio.create_task(_episodic())

        # brief→system1、knowledge预取并行执行（working_memory已在前置步骤完成）
        (brief, emotion), knowledge_prefetched = await asyncio.gather(
            _brief_then_system1(),
            _knowledge_prefetch(),
        )

        # 技能是否可用：System1 认为需要，或者预取结果本身足够相关（System1 对"非技术类技能"如古诗、养生常判断为不需要）
        top_score = knowledge_prefetched[0]["score"] if knowledge_prefetched else 0.0
        requires_knowledge = emotion.get("requires_knowledge", False) or top_score >= settings.BRAIN_KNOWLEDGE_SCORE_THRESHOLD
        knowledge = knowledge_prefetched[:settings.BRAIN_KNOWLEDGE_INJECT_TOP_K] if requires_knowledge else []
        timings["knowledge"] = timings.get("knowledge_prefetch", 0.0) if requires_knowledge else 0.0

        route, route_reason = _choose_route(emotion, requires_knowledge, speaker_known)
        decision: Optional[Dict[str, Any]] = None
        if route == ROUTE_FAST:
            episodic_task.cancel()
            episodic = []
            plan = _quick_plan(emotion)
        else:
            t0 = time.perf_counter()
            episodic = await episodic_task
            timings["episodic_wait"] = (time.perf_counter() - t0) * 1000
            if route == ROUTE_MEMORY:
                plan = _quick_plan(emotion, episodic, knowledge)
            else:
                t0 = time.perf_counter()
                logger.info(f"[WorkingMemoryDebug] agent={agent_id}, working_memory length={len(working_memory)}, content={working_memory[:200]}")
                decision = await self._system2(
                    agent_id, brief, emotion, episodic, knowledge, working_memory, signal, speaker_info, state_and_situation
                )
                timings["system2"] = (time.perf_counter() - t0) * 1000
                # System2失败时退化为直觉计划，保证总能说出话
                plan = _format_decision(decision) if decision else _quick_plan(emotion, episodic, knowledge)

        silent = bool(decision and decision.get("silent"))
        if silent:
            raw_response = ""
            timings["system3"] = 0.0
        else:
            t0 = time.perf_counter()
            raw_response = await self._system3(agent_id, emotion, plan, working_memory, signal, speaker_info, state_only)
            timings["system3"] = (time.perf_counter() - t0) * 1000
            # System3 只写了动作/神态旁白（如"（转身走开）"），去掉旁白后没有台词，按没接话处理
            silent = not raw_response

        # 对外只给说出口的话；谎言句子单独记录，"我"自己记得撒过谎
        response, lies = parse_lies(raw_response)
        if not lies and decision and decision.get("lies"):
            # System3 漏标时，用System2决定的假话兜底，保证"我"记得自己撒过谎
            lies = [str(x) for x in decision["lies"]]

        # "我"的回应也是正在经历的对话，写入工作记忆，下一轮才能接得上；沉默也是回应
        await prefrontal_lobe_memory.append(
            agent_id, response or SILENCE_RECORD, role="assistant", speaker=SELF_SPEAKER, lies=lies,
        )

        # 主体性观测：处境进了哪些系统（System3 不给处境；System2 只在 deliberate 路径调用）
        has_situation = "## 客观处境" in state_and_situation
        situation_to = []
        if has_situation:
            situation_to.append("system1")
            if decision is not None:
                situation_to.append("system2")
        subjectivity_trace = {
            "situation": has_situation,
            "situation_to": situation_to,
            "overall_state": subject_context is not None,
        }
        logger.info(
            f"[Subjectivity] agent={agent_id} speaker={speaker} route={route} "
            f"situation_to={situation_to or '-'} overall_state={subjectivity_trace['overall_state']}"
        )

        timings["total"] = (time.perf_counter() - total_start) * 1000
        logger.info(f"react {agent_id}: route={route}({route_reason}) emotion={emotion['emotion']}({emotion['intensity']:.1f}) "
                    f"action={emotion.get('quick_action')} requires_knowledge={requires_knowledge} brief={len(brief)} "
                    f"episodic={len(episodic)} knowledge={len(knowledge)} timings={ {k: round(v) for k, v in timings.items()} }")

        # 🔥 评估并持久化重要记忆（同步等待完成）
        await memory_gate_service.evaluate_and_persist(agent_id, signal, speaker or "user")

        # 判断是否生成待办：要调两次模型（System One + lite），放后台执行，不阻塞本轮返回
        self._spawn(subjectivity_service.add_todo(
            agent_id=agent_id,
            content=signal,
            source="dialogue",
            context={
                "signal": signal,
                "response": response,
                "speaker": speaker or "用户"
            }
        ))

        # 🆕 记录 react 输出事件（说过的话），本轮完整中间状态放在顶层 trace 字段，耗时放在顶层 rt 字段
        timings["react_total"] = (time.perf_counter() - react_start) * 1000
        trace_id = await agent_life_event_service.record_react_output(
            agent_id=agent_id,
            response=response,
            signal=signal,
            speaker=speaker or "user",
            metadata={
                "route": route,
                "emotion": emotion.get("emotion"),
                "silent": silent
            },
            rt=timings["react_total"],
            trace=_build_react_trace(
                speaker_info=speaker_info, lies=lies, silent=silent, route=route, route_reason=route_reason,
                emotion=emotion, decision=decision, requires_knowledge=requires_knowledge,
                brief=brief, episodic=episodic, knowledge=knowledge,
                overall_state=subject_context.overall_state if subject_context else None,
                working_memory_chars=len(working_memory or ""),
                timings=timings, llm_usage=llm_usage,
            ),
        )

        # 🆕 对话后更新 Subjectivity
        self._spawn(self._update_subjectivity_after_dialogue(
            agent_id=agent_id,
            signal=signal,
            response=response,
            speaker=speaker,
            emotion=emotion,
            trace_id=trace_id,
        ))

        return {
            "response": response,
            "lies": lies,
            "silent": silent,
            "route": route,
            "route_reason": route_reason,
            "emotion": emotion,
            "decision": decision,
            "brief": brief,
            "episodic": episodic,
            "knowledge": knowledge,
            "requires_knowledge": requires_knowledge,
            "subjectivity": subjectivity_trace,
            "timings_ms": {k: round(v, 1) for k, v in timings.items()},
        }

    # ---------------- System1 ----------------

    async def _system1(
        self,
        agent_id: str,
        brief: List[Dict[str, Any]],
        signal: str,
        speaker_info: str,
        situation_context: str = "",
        recent_dialogue: str = "",
    ) -> Dict[str, Any]:
        """
        潜意识情绪反应：使用tev1快速分析情绪、冲动、投入意愿和认知门控

        返回包含 emotion, intensity, impulse, engagement, user_intent, quick_action,
        needs_memory, needs_reasoning 等字段；失败时返回中性情绪
        """
        try:
            # 格式化brief
            brief_text = _format_memories(brief, "（暂无）")

            # 获取persona（可选）
            persona = await persona_manager.get(agent_id)
            persona_text = persona.get("system1", "")

            # 调用 System One API（传入 situation_context）
            emotion = await system1_tev1.analyze(
                brief_text, signal, persona_text, speaker_info, situation_context, recent_dialogue
            )

            return emotion

        except Exception as e:
            logger.error(f"System1(tev1) failed, fallback to neutral emotion: {e}", exc_info=True)
            return dict(NEUTRAL_EMOTION)

    async def _knowledge_recall(self, agent_id: str, signal: str) -> List[Dict[str, Any]]:
        """
        技能记忆召回：当System1判断需要技能时调用

        Args:
            agent_id: Agent ID
            signal: 外部信号

        Returns:
            技能列表
        """
        try:
            # search_knowledge 内部是同步的远程 embedding + qdrant 查询，放线程池避免阻塞事件循环
            knowledges = await asyncio.to_thread(
                knowledge_memory_service.search_knowledge,
                agent_id=agent_id,
                query=signal,
                limit=self.knowledge_top_k,
                score_threshold=0.5,
            )

            logger.debug(f"Knowledge recall: query='{signal[:50]}...', found={len(knowledges)} items")

            return knowledges

        except Exception as e:
            logger.error(f"Knowledge recall failed: {e}", exc_info=True)
            return []

    async def _validate_fact_sources(
        self,
        agent_id: str,
        source_text: str,
        episodic: List[Dict[str, Any]],
        brief: List[Dict[str, Any]]
    ) -> str:
        """
        校验事实来源：检查标注为"记忆ID"的ID是否真实存在

        Args:
            agent_id: Agent ID
            source_text: System2输出的事实来源字段，格式如"[事件]=memory_123"
            episodic: 召回的情景记忆列表
            brief: 召回的简要记忆列表

        Returns:
            校验后的事实来源文本（不存在的记忆ID自动改为"推测"）
        """
        # 解析事实来源：[事件]=记忆ID 或 [事件]=推测 或 [事件]=计划
        fact_pattern = re.compile(r'\[([^\]]+)\]\s*=\s*(\S+)')
        matches = fact_pattern.findall(source_text)

        if not matches:
            return source_text

        # 收集所有真实存在的记忆ID
        valid_ids = set()
        for mem in episodic + brief:
            mem_id = mem.get("id")
            if mem_id:
                valid_ids.add(str(mem_id))

        # 校验每个标注
        validated_parts = []
        for event, source in matches:
            # 如果标注为记忆ID，但该ID不在召回的记忆中，自动改为"推测"
            if source.startswith("memory_") or source.startswith("mem_") or source.isdigit():
                if source not in valid_ids:
                    logger.warning(f"Invalid memory ID in fact source: [{event}]={source}, changed to '推测'")
                    validated_parts.append(f"[{event}]=推测")
                else:
                    validated_parts.append(f"[{event}]={source}")
            else:
                # 推测、计划等非记忆ID的标注，原样保留
                validated_parts.append(f"[{event}]={source}")

        return "；".join(validated_parts) if validated_parts else source_text


    # ---------------- System2 ----------------

    async def _system2(
        self,
        agent_id: str,
        brief: List[Dict[str, Any]],
        emotion: Dict[str, Any],
        episodic: List[Dict[str, Any]],
        knowledge: List[Dict[str, Any]],
        working_memory: str,
        signal: str,
        speaker_info: str,
        situation_summary: str
    ) -> Optional[Dict[str, Any]]:
        """
        社交决策：persona + 信条 + 本能反应 + 情景记忆 + 技能 + 工作记忆 + 主体性上下文 -> 此刻采取的社交行为
        不生成说出口的话。两段persona都给：system1段的价值观/触发点是决定压不压冲动的依据，
        system2段的身份卡/元认知决定能力边界和诚信策略。
        输出是带标签的陈述（心里想/怎么做/核心意思/…），比JSON少写很多结构字符；
        失败（调用异常或一个标签都解析不出）返回None，由调用方退化为直觉计划。

        ❌ 已移除：
        - thought_context：Thoughts 只驱动主动行为，不在被动回应中使用
        """
        persona = await persona_manager.get(agent_id)
        engagement_state = _format_engagement_state(emotion.get('engagement', 0.5))
        conflict_state = _format_conflict_state(emotion.get('conflict', 0.0))
        situation = emotion.get('situation', '正常交流')
        impulse_display = emotion.get('impulse', '无')

        brief_formatted = _format_memories(brief, "（暂无）")
        episodic_formatted = _format_memories(episodic, "（没有想起相关的经历）")

        # Debug: 打印召回的记忆内容（帮助排查事实来源问题）
        logger.info(f"[System2 Input Debug] agent={agent_id}, brief count={len(brief)}, episodic count={len(episodic)}")
        if brief:
            logger.info(f"[Brief Memories] {[m.get('memory', '')[:100] for m in brief[:3]]}")

        user_content = SYSTEM2_USER_TEMPLATE.format(
            situation_summary=situation_summary,
            brief=brief_formatted,
            instinct=_format_instinct(emotion),
            episodic=episodic_formatted,
            knowledge=_format_knowledge(knowledge, "（无相关知识）"),
            working_memory=working_memory or "（无）",
            signal=signal,
        )

        messages = [
            {"role": "system", "content": SYSTEM2_PROMPT.format(
                speaker_info=speaker_info,
                situation=situation,
                impulse=impulse_display,
                engagement_state=engagement_state,
                conflict_state=conflict_state,
                persona=f"{persona['system1']}\n{persona['system2']}")},
            {"role": "user", "content": user_content},
        ]
        try:
            content = await doubao_client.chat(
                messages, temperature=settings.BRAIN_SYSTEM2_TEMPERATURE,
                max_tokens=settings.BRAIN_SYSTEM2_MAX_TOKENS,
                model=settings.DOUBAO_MODEL_PRO,
            )
        except Exception as e:
            logger.error(f"System2 failed: {e}", exc_info=True)
            return None
        decision = _parse_decision(content or "")
        if decision is None:
            logger.warning(f"System2 output has no decision labels, fallback to quick plan.\nRaw output:\n{content}")
            return None

        # 事实来源校验：检查标注为"记忆ID"的ID是否真实存在
        source_text = decision.get("source", "")
        if source_text and source_text not in NONE_VALUES:
            decision["source"] = await self._validate_fact_sources(agent_id, source_text, episodic, brief)

        return decision

    # ---------------- System3 ----------------

    async def _system3(
        self,
        agent_id: str,
        emotion: Dict[str, Any],
        plan: str,
        working_memory: str,
        signal: str,
        speaker_info: str,
        overall_state: str,
    ) -> str:
        """
        语言行为实现：把已决定的行为转成"我"的口语。只给speaking_manner（说话习惯）和整体状态（影响语气），
        不给记忆、信条和处境——要说的事实都已经在plan里，多给只会让它补内容。
        返回带[谎言]标记的原始输出，由调用方解析。
        """
        persona = await persona_manager.get(agent_id)
        messages = [
            {"role": "system", "content": SYSTEM3_PROMPT.format(speaking_manner=persona["speaking_manner"])},
            {"role": "user", "content": SYSTEM3_USER_TEMPLATE.format(
                speaker_info=speaker_info,
                overall_state=overall_state or "【当前整体状态】\n（无）",
                working_memory=working_memory or "（无）",
                signal=signal,
                instinct=_format_instinct(emotion),
                plan=plan,
            ) + _language_hint(signal)},
        ]
        try:
            content = (await doubao_client.chat(
                messages, temperature=settings.BRAIN_SYSTEM3_TEMPERATURE,
                max_tokens=settings.BRAIN_SYSTEM3_MAX_TOKENS,
                model=settings.DOUBAO_MODEL_CHARACTER,
            )).strip()
            return self._only_spoken(content)
        except Exception as e:
            logger.error(f"System3 failed: {e}", exc_info=True)
            return ""

    @staticmethod
    def _only_spoken(text: str) -> str:
        """
        兜底：去掉模型偶尔夹带的括号旁白（如"（叹气）""（语气带着火气）"）和包裹整句的引号。
        说出口的话里几乎不会出现括号，提示词约束不住时由这里保证只剩原话。
        整段都是旁白（如"（沉默地转身走开）"）时返回空串，即"我"没有说话。
        """
        cleaned = STAGE_DIRECTION.sub("", text).strip()
        if len(cleaned) >= 2 and cleaned[0] in "“\"" and cleaned[-1] in "”\"":
            cleaned = cleaned[1:-1].strip()
        if cleaned != text:
            logger.info(f"System3 output sanitized: {text[:80]!r} -> {cleaned[:80]!r}")
        return cleaned

    # ---------------- Memory Gate Integration ----------------

    async def _update_subjectivity_after_dialogue(
        self,
        agent_id: str,
        signal: str,
        response: str,
        speaker: Optional[str],
        emotion: Dict[str, Any],
        trace_id: Optional[str] = None,
    ):
        """
        对话后更新 Subjectivity：生成念头、更新情绪状态

        异步执行，不阻塞主流程。overall_state 的变化写入 emotion 事件，metadata.trace_id 关联本轮 react_output
        """
        try:
            # 1. 根据情绪更新 overall_state
            emotion_name = emotion.get("emotion", "平静")
            intensity = emotion.get("intensity", 0.0)

            # 情绪映射到 mood 和 stress 变化
            mood_delta = 0.0
            stress_delta = 0.0

            if emotion_name in ["喜悦", "高兴", "兴奋", "感激"]:
                mood_delta = intensity * 0.3  # 正面情绪提升 mood
                stress_delta = -intensity * 0.1  # 降低 stress
            elif emotion_name in ["焦虑", "愤怒", "沮丧", "恐惧"]:
                mood_delta = -intensity * 0.3  # 负面情绪降低 mood
                stress_delta = intensity * 0.2  # 提升 stress
            elif emotion_name in ["平静", "放松"]:
                stress_delta = -intensity * 0.15  # 平静降低 stress

            if abs(mood_delta) > 0.01 or abs(stress_delta) > 0.01:
                reason = f"对话情绪: {emotion_name} (强度: {intensity:.2f})"
                change = await subjectivity_service.adjust_overall_state(
                    agent_id=agent_id,
                    mood_delta=mood_delta,
                    stress_delta=stress_delta,
                    reason=reason
                )
                logger.info(f"Updated overall_state for {agent_id}: mood_delta={mood_delta:.3f}, stress_delta={stress_delta:.3f}")
                await agent_life_event_service.record_emotion(
                    agent_id=agent_id,
                    emotion_state={
                        "before": change["before"],
                        "after": change["after"],
                        "delta": {"mood": round(mood_delta, 4), "stress": round(stress_delta, 4)},
                        "emotion": emotion_name,
                        "intensity": intensity,
                    },
                    trigger=reason,
                    metadata={"trace_id": trace_id, "speaker": speaker},
                )

            # 2. 根据对话内容判断是否生成念头
            # 降低阈值：情绪强度 > 0.1 或参与度 > 0.5 时生成念头
            engagement = emotion.get("engagement", 0.0)

            if intensity > 0.1 or engagement > 0.5:
                # 生成念头（计时：生成耗时记为 thought 事件的 rt）
                t_thought = time.perf_counter()
                thought_data = await subjectivity_service.generate_thought(
                    agent_id=agent_id,
                    trigger="dialogue"
                )
                if thought_data:
                    # 保存念头到 SubjectContext 和 MongoDB
                    await subjectivity_service.add_thought(
                        agent_id=agent_id,
                        thought=thought_data.get("thought", ""),
                        related_to=thought_data.get("related_to"),
                        target_person=thought_data.get("target_person"),
                        urgency=thought_data.get("urgency", 0.0),
                        rt=(time.perf_counter() - t_thought) * 1000,
                    )
                    logger.info(f"Generated and saved thought for {agent_id}: {thought_data.get('thought', '')[:50]}")

        except Exception as e:
            logger.error(f"Failed to update subjectivity after dialogue for {agent_id}: {e}", exc_info=True)



cyber_brain = CyberBrain()
