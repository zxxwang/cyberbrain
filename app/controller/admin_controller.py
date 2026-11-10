"""
Admin Controller: 管理接口

提供：
- GET /admin/agents: 获取所有agent列表
- POST /admin/initialize_agent: 批量初始化agent（记忆+persona+关系+subjectivity，一次性）
- GET /admin/agent/{agent_id}: 获取agent的完整信息
- GET /admin/timeline/{agent_id}: 获取agent的时间线事件
- PUT /admin/agent/{agent_id}: 覆盖式更新Redis中的persona/subject context/relationships
- DELETE /admin/agent/{agent_id}/pfc: 清空前额叶短时记忆（测试用）
- POST /admin/agent/{agent_id}/memories: 新增brief/episodic/knowledge记忆
- GET /admin/config: 查询运行时可调配置
- PUT /admin/config: 修改运行时可调配置（下一次读取即生效，并写回 .env）
"""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field

from app.common.response import APIResponse
from app.config.runtime_config import RuntimeConfigError, list_runtime_configs, update_runtime_configs
from app.repository.memory_repository import memory_repository
from app.repository.relationship_repository import relationship_repository
from app.service.persona_manager import persona_manager
from app.service.knowledge_memory_service import knowledge_memory_service
from app.service.subjectivity_service import subjectivity_service
from app.service.agent_life_event_service import agent_life_event_service
from app.client.redis_client import redis_client
from app.service.persona_manager import PERSONA_KEY_PREFIX
from app.service.prefrontal_lobe_memory import stm_key, compress_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


class BriefMemory(BaseModel):
    """Brief记忆数据结构"""
    content: str
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class EpisodicMemory(BaseModel):
    """Episodic记忆数据结构"""
    content: str
    emotion_charge: Optional[float] = Field(default=0.0, ge=-1.0, le=1.0)
    tags: Optional[List[str]] = None
    occurred_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class KnowledgeMemory(BaseModel):
    """技能记忆数据结构"""
    title: str = Field(..., description="技能标题")
    content: str = Field(..., description="技能详细内容")
    knowledge_type: str = Field(default="concept", description="技能类型: concept/howto/best_practice/troubleshooting/example")
    domain: str = Field(..., description="大领域，如'前端开发'")
    sub_domain: Optional[str] = Field(default=None, description="子领域，如'CSS框架'")
    proficiency: float = Field(default=0.5, ge=0.0, le=1.0, description="熟练度 0-1")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="可信度 0-1")
    source: str = Field(default="experience", description="来源: experience/learning/project/documentation")
    tags: Optional[List[str]] = Field(default=None, description="标签列表")
    related_knowledge: Optional[List[str]] = Field(default=None, description="关联知识ID列表")
    prerequisite_knowledge: Optional[List[str]] = Field(default=None, description="前置知识ID列表")


class Relationship(BaseModel):
    """人际关系数据结构"""
    person_name: str = Field(..., description="人名")
    relationship: str = Field(..., description="关系类型，如'同事'/'上级'/'朋友'/'家人'")
    description: str = Field(default="", description="关系描述，如'关系一般，比较严格'")


class SituationInit(BaseModel):
    """Situation初始化"""
    current_activity: Optional[str] = Field(default=None, description="当前主要活动，如'准备游戏上线'")
    time_sensitive_events: List[Dict[str, str]] = Field(default_factory=list, description="时间敏感事件，格式：[{description, deadline, status}]")
    ongoing_states: List[str] = Field(default_factory=list, description="持续状态（无日期），如'最近失恋了'")
    attention_focus: List[str] = Field(default_factory=list, description="当前关注点，如['游戏稳定性', '玩家反馈']")


class OverallStateInit(BaseModel):
    """整体状态初始化"""
    mood: float = Field(default=0.5, ge=0.0, le=1.0, description="心情 0-1")
    stress: float = Field(default=0.3, ge=0.0, le=1.0, description="压力 0-1")
    energy: float = Field(default=0.7, ge=0.0, le=1.0, description="精力 0-1")


class ThoughtInit(BaseModel):
    """念头初始化"""
    content: str = Field(..., description="念头内容")
    trigger: str = Field(default="manual", description="触发来源")
    intensity: float = Field(default=0.5, ge=0.0, le=1.0, description="强度 0-1")


class PendingTaskInit(BaseModel):
    """待办任务初始化"""
    content: str = Field(..., description="任务内容")
    source: str = Field(default="manual", description="来源")
    priority: float = Field(default=0.6, ge=0.0, le=1.0, description="优先级 0-1")
    deadline: Optional[str] = Field(default=None, description="截止时间 YYYY-MM-DD")
    related_person: Optional[str] = Field(default=None, description="相关人员")
    interaction_type: Optional[str] = Field(default=None, description="交互类型")
    status: str = Field(default="pending", description="状态：只有 pending 会写入；其他值（如 completed）会被忽略")


class InitializeAgentRequest(BaseModel):
    """初始化Agent请求 - 一次性设置所有记忆和persona"""
    agent_id: str
    brief_memories: List[BriefMemory] = Field(default_factory=list)
    episodic_memories: List[EpisodicMemory] = Field(default_factory=list)
    knowledge_memories: List[KnowledgeMemory] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list, description="人际关系列表")
    situation: Optional[SituationInit] = Field(default=None, description="当前情境信息")
    overall_state: Optional[OverallStateInit] = Field(default=None, description="整体状态（心情/压力/精力）")
    thoughts: List[ThoughtInit] = Field(default_factory=list, description="初始念头列表")
    pending_tasks: List[PendingTaskInit] = Field(default_factory=list, description="待办任务列表")
    persona_system1: Optional[str] = Field(default=None, description="System1人格画像（情绪参数）")
    persona_system2: Optional[str] = Field(default=None, description="System2人格画像（认知与表达参数）")
    speaking_manner: Optional[str] = Field(default=None, description="说话习惯（用词、句式、语气）")


@router.get("/agents", response_model=APIResponse[List[str]])
async def list_agents():
    """
    获取所有agent列表

    以 Redis 中的 persona 和 subject_context 取并集：
    - persona:{id}:system1  initialize_agent 传了 persona 就会写
    - subject_context:{id}  initialize_agent 传了 situation/overall_state/thoughts/tasks，或该 agent 发生过对话就会写
    不用 Qdrant episodic：没有情景记忆、mem0 没抽出内容或记忆已被衰减删除的 agent 都会漏掉
    """
    try:
        context_prefix = f"{subjectivity_service.redis_key_prefix}:"
        persona_keys = await redis_client.scan_keys(f"{PERSONA_KEY_PREFIX}*:system1")
        context_keys = await redis_client.scan_keys(f"{context_prefix}*")
        agent_ids = {k[len(PERSONA_KEY_PREFIX):-len(":system1")] for k in persona_keys}
        agent_ids |= {k[len(context_prefix):] for k in context_keys}
        return APIResponse(data=sorted(agent_ids))
    except Exception as e:
        logger.error(f"Failed to list agents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _build_pending_tasks(tasks: List["PendingTaskInit"]) -> List[Dict[str, Any]]:
    """
    手动传入的待办 -> pending_tasks 记录。pending_tasks 只存待执行的任务，
    status 不是 pending 的（如已完成）直接忽略；task_id 用 uuid，同一秒批量写入也不会重复
    """
    now = datetime.now().isoformat()
    result = []
    for task in tasks:
        if task.status != "pending":
            logger.info(f"Skip non-pending task on write: {task.content} (status={task.status})")
            continue
        result.append({
            "task_id": f"task_{uuid.uuid4().hex}",
            "content": task.content,
            "source": task.source,
            "priority": task.priority,
            "deadline": task.deadline,
            "related_person": task.related_person,
            "interaction_type": task.interaction_type,
            "created_at": now,
            "status": "pending",
        })
    return result


async def _agent_exists(agent_id: str) -> Optional[str]:
    """
    agent 是否已有任何数据，有则返回第一个命中的数据来源，没有返回 None

    initialize_agent 的每项入参都是可选的，所以没有单一必有的 key，逐个查它可能写过的地方：
    先查 Redis（便宜），再查 Qdrant 的 brief / episodic / knowledge
    """
    persona = PERSONA_KEY_PREFIX + agent_id
    redis_keys = {
        "persona": [f"{persona}:system1", f"{persona}:system2", f"{persona}:speaking_manner"],
        "subject_context": [f"{subjectivity_service.redis_key_prefix}:{agent_id}"],
        "relationships": [f"agent:{agent_id}:relationships"],
    }
    for source, keys in redis_keys.items():
        if await redis_client.client.exists(*keys):
            return source

    for memory_type in ("brief", "episodic"):
        if await memory_repository.get_all(agent_id, memory_type=memory_type):
            return memory_type
    if await asyncio.to_thread(knowledge_memory_service.get_all_knowledge, agent_id, 1):
        return "knowledge"
    return None


@router.post("/initialize_agent", response_model=APIResponse[dict])
async def initialize_agent(request: InitializeAgentRequest):
    """批量初始化agent - 一次性设置记忆、技能和persona。agent_id 已存在时返回 409"""
    agent_id = request.agent_id
    # 锁住检查到写完这段，防止同一 agent_id 的两个并发请求都通过检查
    lock = f"lock:initialize_agent:{agent_id}"
    if not await redis_client.acquire_lock(lock, ttl=600):
        raise HTTPException(status_code=409, detail=f"Agent {agent_id} 正在初始化中")
    try:
        existing = await _agent_exists(agent_id)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Agent {agent_id} 已存在（已有 {existing} 数据），不允许重复初始化",
            )
        return await _initialize_agent(request)
    finally:
        await redis_client.release_lock(lock)


async def _add_brief(agent_id: str, mem: BriefMemory) -> bool:
    """经 mem0 新增一条 brief 记忆（infer=False，不做合并/去重），成功返回 True"""
    try:
        metadata = dict(mem.metadata or {})
        if mem.tags:
            metadata["tags"] = mem.tags
        ids = await memory_repository.add(
            agent_id=agent_id,
            messages=[{"role": "user", "content": mem.content}],
            memory_type="brief",
            metadata=metadata,
        )
        return bool(ids)
    except Exception as e:
        logger.warning(f"Failed to add brief memory: {e}")
        return False


async def _add_episodic(agent_id: str, mem: EpisodicMemory) -> bool:
    """经 mem0 新增一条 episodic 记忆（infer=False，不做合并/去重），成功返回 True"""
    try:
        metadata = dict(mem.metadata or {})
        if mem.tags:
            metadata["tags"] = mem.tags
        if mem.emotion_charge is not None:
            metadata["emotion_charge"] = mem.emotion_charge
        if mem.occurred_at:
            metadata["occurred_at"] = mem.occurred_at
        ids = await memory_repository.add(
            agent_id=agent_id,
            messages=[{"role": "user", "content": mem.content}],
            memory_type="episodic",
            metadata=metadata,
        )
        return bool(ids)
    except Exception as e:
        logger.warning(f"Failed to add episodic memory: {e}")
        return False


async def _add_knowledge(agent_id: str, knowledge: KnowledgeMemory) -> bool:
    """直接写 Qdrant 新增一条 knowledge 记忆（每次生成新 uuid），成功返回 True"""
    try:
        knowledge_id = await asyncio.to_thread(
            knowledge_memory_service.add_knowledge,
            agent_id=agent_id,
            title=knowledge.title,
            content=knowledge.content,
            knowledge_type=knowledge.knowledge_type,
            domain=knowledge.domain,
            sub_domain=knowledge.sub_domain,
            proficiency=knowledge.proficiency,
            confidence=knowledge.confidence,
            source=knowledge.source,
            tags=knowledge.tags,
            related_knowledge=knowledge.related_knowledge,
            prerequisite_knowledge=knowledge.prerequisite_knowledge,
        )
        return bool(knowledge_id)
    except Exception as e:
        logger.warning(f"Failed to add knowledge memory: {e}")
        return False


async def _initialize_agent(request: InitializeAgentRequest):
    try:
        agent_id = request.agent_id
        brief_count = len(request.brief_memories)
        episodic_count = len(request.episodic_memories)
        knowledge_count = len(request.knowledge_memories)
        relationship_count = len(request.relationships)

        logger.info(f"Initializing agent {agent_id}: {brief_count} brief, {episodic_count} episodic, "
                    f"{knowledge_count} knowledge, {relationship_count} relationships, "
                    f"persona={'yes' if request.persona_system1 and request.persona_system2 else 'no'}, "
                    f"speaking_manner={'yes' if request.speaking_manner else 'no'}")

        # 1. 设置persona（如果提供）
        if request.persona_system1 and request.persona_system2:
            await persona_manager.set(
                agent_id,
                request.persona_system1,
                request.persona_system2,
                request.speaking_manner
            )

        brief_success = 0
        episodic_success = 0
        knowledge_success = 0
        relationship_success = 0

        # 2-4. 批量添加 brief / episodic / knowledge 记忆
        for mem in request.brief_memories:
            brief_success += await _add_brief(agent_id, mem)
        for mem in request.episodic_memories:
            episodic_success += await _add_episodic(agent_id, mem)
        for knowledge in request.knowledge_memories:
            knowledge_success += await _add_knowledge(agent_id, knowledge)

        # 5. 批量添加人际关系
        for rel in request.relationships:
            try:
                await relationship_repository.add_relationship(
                    agent_id=agent_id,
                    person_name=rel.person_name,
                    relationship=rel.relationship,
                    description=rel.description,
                )
                relationship_success += 1
            except Exception as e:
                logger.warning(f"Failed to add relationship {rel.person_name}: {e}")

        # 6. 初始化Situation（如果提供）
        situation_set = False
        if request.situation:
            try:
                await subjectivity_service.set_situation(
                    agent_id=agent_id,
                    current_activity=request.situation.current_activity,
                    time_sensitive_events=request.situation.time_sensitive_events,
                    ongoing_states=request.situation.ongoing_states,
                    attention_focus=request.situation.attention_focus
                )
                situation_set = True
                logger.info(f"Initialized situation for {agent_id}")
            except Exception as e:
                logger.warning(f"Failed to set situation: {e}")

        # 7. 初始化Overall State（如果提供）
        overall_state_set = False
        if request.overall_state:
            try:
                await subjectivity_service.set_overall_state(
                    agent_id=agent_id,
                    mood=request.overall_state.mood,
                    stress=request.overall_state.stress,
                    energy=request.overall_state.energy
                )
                overall_state_set = True
                logger.info(f"Initialized overall_state for {agent_id}: mood={request.overall_state.mood:.2f}, "
                           f"stress={request.overall_state.stress:.2f}, energy={request.overall_state.energy:.2f}")
            except Exception as e:
                logger.warning(f"Failed to set overall_state: {e}")

        # 8. 初始化Thoughts（如果提供）
        thoughts_added = 0
        for thought in request.thoughts:
            try:
                await subjectivity_service.add_thought(
                    agent_id=agent_id,
                    thought=thought.content,  # content -> thought
                    related_to=None,
                    target_person=None,
                    urgency=thought.intensity  # intensity -> urgency
                )
                thoughts_added += 1
            except Exception as e:
                logger.warning(f"Failed to add thought: {e}")

        # 9. 初始化 Pending Tasks（如果提供）
        tasks_added = 0
        if request.pending_tasks:
            try:
                subject_ctx = await subjectivity_service.get_context(agent_id)
                if not subject_ctx:
                    from app.service.subjectivity_service import SubjectContext
                    subject_ctx = SubjectContext(agent_id=agent_id)
                new_tasks = _build_pending_tasks(request.pending_tasks)
                subject_ctx.pending_tasks.extend(new_tasks)
                await subjectivity_service.save_context(subject_ctx)
                tasks_added = len(new_tasks)
            except Exception as e:
                logger.warning(f"Failed to add pending tasks: {e}")

        result = {
            "agent_id": agent_id,
            "brief": {
                "total": brief_count,
                "success": brief_success,
                "failed": brief_count - brief_success
            },
            "episodic": {
                "total": episodic_count,
                "success": episodic_success,
                "failed": episodic_count - episodic_success
            },
            "knowledge": {
                "total": knowledge_count,
                "success": knowledge_success,
                "failed": knowledge_count - knowledge_success
            },
            "relationships": {
                "total": relationship_count,
                "success": relationship_success,
                "failed": relationship_count - relationship_success
            },
            "situation_set": situation_set,
            "overall_state_set": overall_state_set,
            "thoughts_added": thoughts_added,
            "tasks_added": tasks_added,
            "persona_set": bool(request.persona_system1 and request.persona_system2)
        }

        logger.info(f"Agent {agent_id} initialization completed: {result}")
        return APIResponse(data=result)

    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timeline/{agent_id}")
async def get_agent_timeline(
    agent_id: str,
    event_types: Optional[str] = Query(None, description="事件类型过滤，逗号分隔，如: react_input,react_output,todo"),
    page: int = Query(1, ge=1, description="页码，从1开始"),
    page_size: int = Query(50, ge=1, le=200, description="每页数量，最大200")
):
    """
    获取 Agent 的生命事件时间线（分页）

    事件类型：
    - react_input: 听到的话
    - react_output: 说过的话
    - thought: 产生的念头
    - todo: 待办任务（created/completed/cancelled）
    - situation: Situation 变化
    - memory: 记忆写入
    - emotion: 情绪变化
    - task_execution: 任务执行
    """
    try:
        # 解析事件类型过滤
        event_type_list = None
        if event_types:
            event_type_list = [t.strip() for t in event_types.split(",") if t.strip()]

        # 获取时间线
        result = await agent_life_event_service.get_agent_timeline(
            agent_id=agent_id,
            event_types=event_type_list,
            page=page,
            page_size=page_size
        )

        return APIResponse(data=result)

    except Exception as e:
        logger.error(f"Failed to get agent timeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# mem0 写入时把 metadata 平铺进 Qdrant payload，get_all 也平铺返回；这些是 mem0 / 本项目的内部字段，不算用户 metadata
_MEM0_INTERNAL_FIELDS = {"id", "memory", "data", "hash", "user_id", "role", "created_at", "updated_at",
                         "text_lemmatized", "decay_weight", "last_decay_update",
                         "last_reactivated_at"}


def _memory_metadata(mem: Dict[str, Any]) -> Dict[str, Any]:
    """从 get_all 的平铺结果里还原写入时的 metadata（tags / emotion_charge / occurred_at 等）"""
    return {k: v for k, v in mem.items() if k not in _MEM0_INTERNAL_FIELDS}


@router.get("/agent/{agent_id}", response_model=APIResponse[dict])
async def get_agent_info(agent_id: str):
    """
    获取 Agent 的完整信息（按照 /initialize_agent 的入参结构返回）

    返回：
    - persona: {system1, system2, speaking_manner}
    - relationships: [{person_name, relationship, description}]
    - situation: {current_activity, time_sensitive_events, ongoing_states, attention_focus}
    - overall_state: {mood, stress, energy}
    - thoughts: [{content, trigger, intensity}]
    - pending_tasks: [{content, source, priority, deadline, related_person, interaction_type, status}]
    - brief_memories: [{content, tags, metadata}]
    - episodic_memories: [{content, tags, emotion_charge, occurred_at, metadata}]
    - knowledge_memories: [{title, content, knowledge_type, domain, ...}]
    """
    try:
        result = {
            "agent_id": agent_id,
            "persona": {},
            "relationships": [],
            "situation": None,
            "overall_state": None,
            "thoughts": [],
            "pending_tasks": [],
            "brief_memories": [],
            "episodic_memories": [],
            "knowledge_memories": []
        }

        # 1. 获取 Persona
        try:
            persona = await persona_manager.get(agent_id)
            result["persona"] = {
                "system1": persona.get("system1", ""),
                "system2": persona.get("system2", ""),
                "speaking_manner": persona.get("speaking_manner", "")
            }
        except Exception as e:
            logger.warning(f"Failed to get persona for {agent_id}: {e}")

        # 2. 获取 Relationships
        try:
            relationships_dict = await relationship_repository.load_all_relationships(agent_id)
            for person_name, relation_desc in relationships_dict.items():
                # relation_desc 格式: "关系类型，描述"
                parts = relation_desc.split("，", 1)
                relationship = parts[0] if parts else relation_desc
                description = parts[1] if len(parts) > 1 else ""

                result["relationships"].append({
                    "person_name": person_name,
                    "relationship": relationship,
                    "description": description
                })
        except Exception as e:
            logger.warning(f"Failed to get relationships for {agent_id}: {e}")

        # 3. 获取 Subjectivity (Situation + Overall State + Thoughts)
        try:
            context = await subjectivity_service.get_context(agent_id)
            if context:
                # Situation
                if context.situation:
                    result["situation"] = {
                        "current_activity": context.situation.get("current_activity"),
                        "time_sensitive_events": context.situation.get("time_sensitive_events", []),
                        "ongoing_states": context.situation.get("ongoing_states", []),
                        "attention_focus": context.situation.get("attention_focus", [])
                    }

                # Overall State
                if context.overall_state:
                    result["overall_state"] = {
                        "mood": context.overall_state.get("mood", 0.5),
                        "stress": context.overall_state.get("stress", 0.3),
                        "energy": context.overall_state.get("energy", 0.7)
                    }

                # Thoughts
                if context.thoughts:
                    for thought in context.thoughts:
                        result["thoughts"].append({
                            "content": thought.get("thought", ""),
                            "trigger": "manual",  # 无法获取原始 trigger
                            "intensity": thought.get("urgency", 0.5)
                        })

                # Pending Tasks
                if context.pending_tasks:
                    # 完成/取消的任务已从 pending_tasks 移除（历史见 timeline 的 todo 事件）
                    for task in context.pending_tasks:
                        result["pending_tasks"].append({
                            "content": task.get("content", ""),
                            "source": task.get("source", "manual"),
                            "priority": task.get("priority", 0.6),
                            "deadline": task.get("deadline"),
                            "related_person": task.get("related_person"),
                            "interaction_type": task.get("interaction_type"),
                            "status": task.get("status", "pending")
                        })
        except Exception as e:
            logger.warning(f"Failed to get subjectivity for {agent_id}: {e}")

        # 4. 获取记忆数据
        try:
            # Brief Memories
            brief_memories = await memory_repository.get_all(agent_id, memory_type="brief")
            for mem in brief_memories:
                metadata = _memory_metadata(mem)
                result["brief_memories"].append({
                    "content": mem.get("memory", ""),
                    "tags": metadata.get("tags", []),
                    "metadata": metadata
                })

            # Episodic Memories
            episodic_memories = await memory_repository.get_all(agent_id, memory_type="episodic")
            for mem in episodic_memories:
                metadata = _memory_metadata(mem)
                result["episodic_memories"].append({
                    "content": mem.get("memory", ""),
                    "tags": metadata.get("tags", []),
                    "emotion_charge": metadata.get("emotion_charge"),
                    "occurred_at": metadata.get("occurred_at"),
                    "metadata": metadata
                })

            # Knowledge Memories
            knowledge_list = knowledge_memory_service.get_all_knowledge(agent_id)
            for knowledge in knowledge_list:
                result["knowledge_memories"].append({
                    "title": knowledge.get("title", ""),
                    "content": knowledge.get("content", ""),
                    "knowledge_type": knowledge.get("knowledge_type", ""),
                    "domain": knowledge.get("domain", ""),
                    "sub_domain": knowledge.get("sub_domain", ""),
                    "proficiency": knowledge.get("proficiency", 0.5),
                    "confidence": knowledge.get("confidence", 0.5),
                    "source": knowledge.get("source", ""),
                    "tags": knowledge.get("tags", []),
                    "related_knowledge": knowledge.get("related_knowledge", []),
                    "prerequisite_knowledge": knowledge.get("prerequisite_knowledge", [])
                })
        except Exception as e:
            logger.warning(f"Failed to get memories for {agent_id}: {e}")

        return APIResponse(data=result)

    except Exception as e:
        logger.error(f"Failed to get agent info for {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class UpdateAgentRequest(BaseModel):
    """更新Agent请求 - 覆盖式更新Redis中的初始化数据"""
    persona_system1: Optional[str] = Field(default=None, description="System1人格画像（情绪参数）")
    persona_system2: Optional[str] = Field(default=None, description="System2人格画像（认知与表达参数）")
    speaking_manner: Optional[str] = Field(default=None, description="说话习惯（System3用词、句式、语气）")
    situation: Optional[SituationInit] = Field(default=None, description="当前情境信息")
    overall_state: Optional[OverallStateInit] = Field(default=None, description="整体状态（心情/压力/精力）")
    thoughts: Optional[List[ThoughtInit]] = Field(default=None, description="念头列表（覆盖现有）")
    pending_tasks: Optional[List[PendingTaskInit]] = Field(default=None, description="待办任务列表（覆盖现有）")
    relationships: Optional[List[Relationship]] = Field(default=None, description="人际关系列表（覆盖现有）")


@router.put("/agent/{agent_id}", response_model=APIResponse[dict])
async def update_agent(agent_id: str, request: UpdateAgentRequest):
    """
    覆盖式更新Agent的Redis初始化数据

    只更新提供的字段，未提供的字段保持不变。
    - persona_system1/system2/speaking_manner: 覆盖对应的persona字段
    - situation: 覆盖整个situation对象
    - overall_state: 覆盖整个overall_state对象
    - thoughts: 覆盖整个thoughts列表
    - pending_tasks: 覆盖整个pending_tasks列表
    - relationships: 覆盖所有关系（先清空再添加）

    注意：system3对应的是speaking_manner字段
    """
    try:
        # 1. 检查agent是否存在
        exists = await _agent_exists(agent_id)
        if not exists:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        updated_fields = []

        # 2. 更新Persona字段（system1/system2/speaking_manner）
        persona_updates = {}
        if request.persona_system1 is not None:
            persona_key = f"{PERSONA_KEY_PREFIX}{agent_id}:system1"
            persona_updates[persona_key] = request.persona_system1.strip()
            updated_fields.append("persona_system1")

        if request.persona_system2 is not None:
            persona_key = f"{PERSONA_KEY_PREFIX}{agent_id}:system2"
            persona_updates[persona_key] = request.persona_system2.strip()
            updated_fields.append("persona_system2")

        if request.speaking_manner is not None:
            persona_key = f"{PERSONA_KEY_PREFIX}{agent_id}:speaking_manner"
            persona_updates[persona_key] = request.speaking_manner.strip()
            updated_fields.append("speaking_manner")

        if persona_updates:
            await redis_client.set_many(persona_updates)
            logger.info(f"Updated persona fields for {agent_id}: {list(persona_updates.keys())}")

        # 3. 更新Subject Context字段
        context_updated = False
        if any([
            request.situation is not None,
            request.overall_state is not None,
            request.thoughts is not None,
            request.pending_tasks is not None
        ]):
            from app.service.subjectivity_service import SubjectContext

            # 获取现有context或创建新的
            context = await subjectivity_service.get_context(agent_id)
            if not context:
                context = SubjectContext(agent_id=agent_id)

            # 覆盖提供的字段
            if request.situation is not None:
                context.situation = {
                    "current_activity": request.situation.current_activity,
                    "time_sensitive_events": request.situation.time_sensitive_events,
                    "ongoing_states": request.situation.ongoing_states,
                    "attention_focus": request.situation.attention_focus
                }
                updated_fields.append("situation")

            if request.overall_state is not None:
                context.overall_state = {
                    "mood": request.overall_state.mood,
                    "stress": request.overall_state.stress,
                    "energy": request.overall_state.energy
                }
                updated_fields.append("overall_state")

            if request.thoughts is not None:
                # 覆盖thoughts列表
                context.thoughts = [
                    {
                        "time": datetime.now().isoformat(),
                        "thought": thought.content,
                        "related_to": thought.trigger,
                        "target_person": None,
                        "urgency": thought.intensity,
                        "expressed": False
                    }
                    for thought in request.thoughts
                ]
                updated_fields.append("thoughts")

            if request.pending_tasks is not None:
                # 覆盖pending_tasks列表
                context.pending_tasks = _build_pending_tasks(request.pending_tasks)
                updated_fields.append("pending_tasks")

            await subjectivity_service.save_context(context)
            context_updated = True
            logger.info(f"Updated subject context for {agent_id}")

        # 4. 更新Relationships（覆盖式：先清空再添加）
        if request.relationships is not None:
            # 清空现有关系
            await relationship_repository.clear_all_relationships(agent_id)

            # 添加新关系
            for rel in request.relationships:
                await relationship_repository.add_relationship(
                    agent_id=agent_id,
                    person_name=rel.person_name,
                    relationship=rel.relationship,
                    description=rel.description
                )

            updated_fields.append("relationships")
            logger.info(f"Updated {len(request.relationships)} relationships for {agent_id}")

        result = {
            "agent_id": agent_id,
            "updated_fields": updated_fields,
            "persona_updated": bool(persona_updates),
            "context_updated": context_updated,
            "relationships_updated": request.relationships is not None
        }

        logger.info(f"Agent {agent_id} update completed: {updated_fields}")
        return APIResponse(data=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




@router.delete("/agent/{agent_id}/pfc", response_model=APIResponse[dict])
async def clear_pfc(agent_id: str):
    """
    清空 Agent 的前额叶短时记忆（PFC），方便测试

    删除 wm:{agent_id}:stm（原始对话记录）和 wm:{agent_id}:stm:compress（压缩摘要）。
    两者一起删：只删 stm 的话，压缩摘要仍会被拼进 System2 上下文。
    """
    try:
        keys = [stm_key(agent_id), compress_key(agent_id)]
        deleted = await redis_client.delete(*keys)
        logger.info(f"Cleared PFC for {agent_id}: {deleted} keys deleted")
        return APIResponse(data={"agent_id": agent_id, "deleted_keys": deleted})
    except Exception as e:
        logger.error(f"Failed to clear PFC for {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class AddMemoriesRequest(BaseModel):
    """新增记忆请求 - 传进来的每一条都作为新记忆写入，不覆盖、不合并"""
    brief_memories: List[BriefMemory] = Field(default_factory=list)
    episodic_memories: List[EpisodicMemory] = Field(default_factory=list)
    knowledge_memories: List[KnowledgeMemory] = Field(default_factory=list)


@router.post("/agent/{agent_id}/memories", response_model=APIResponse[dict])
async def add_memories(agent_id: str, request: AddMemoriesRequest):
    """
    给已存在的 Agent 新增 brief / episodic / knowledge 记忆

    - brief / episodic：经 mem0 写入 Qdrant（infer=False，原文直接入库，不做抽取和去重合并）
    - knowledge：直接写 Qdrant，每条生成新 uuid
    agent 不存在返回 404（新 agent 请走 /initialize_agent）
    """
    if not await _agent_exists(agent_id):
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    if not (request.brief_memories or request.episodic_memories or request.knowledge_memories):
        raise HTTPException(status_code=400, detail="No memories provided")

    try:
        brief_ok = sum([await _add_brief(agent_id, m) for m in request.brief_memories])
        episodic_ok = sum([await _add_episodic(agent_id, m) for m in request.episodic_memories])
        knowledge_ok = sum([await _add_knowledge(agent_id, k) for k in request.knowledge_memories])

        def stat(total: int, ok: int) -> dict:
            return {"total": total, "success": ok, "failed": total - ok}

        result = {
            "agent_id": agent_id,
            "brief": stat(len(request.brief_memories), brief_ok),
            "episodic": stat(len(request.episodic_memories), episodic_ok),
            "knowledge": stat(len(request.knowledge_memories), knowledge_ok),
        }
        logger.info(f"Added memories for {agent_id}: {result}")
        return APIResponse(data=result)
    except Exception as e:
        logger.error(f"Failed to add memories for {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/config", response_model=APIResponse[List[dict]])
async def get_runtime_configs():
    """
    查询运行时可调配置（白名单见 app/config/runtime_config.py）

    每项返回：name、value（当前生效值）、default（settings 默认值）、type、description、min、max
    """
    return APIResponse(data=list_runtime_configs())


@router.put("/config", response_model=APIResponse[dict])
async def update_runtime_config(
    updates: Dict[str, Any] = Body(..., examples=[{"BRAIN_SYSTEM2_TEMPERATURE": 0.4, "BRAIN_EPISODIC_TOP_K": 8}]),
):
    """
    修改运行时可调配置，代码下一次读取对应配置时即为新值

    - body 为 {配置名: 新值}，只传要改的项
    - 先整体校验（名字在白名单内、类型正确、在取值范围内），任意一项不合法返回 400 且一项都不改
    - 校验通过后先写回 .env（重启后延续），再更新内存中的 settings（立即生效）
    """
    if not updates:
        raise HTTPException(status_code=400, detail="No configs provided")
    try:
        changed = update_runtime_configs(updates)
    except RuntimeConfigError as e:
        raise HTTPException(status_code=400, detail=e.errors)
    except OSError as e:
        logger.error(f"Failed to write .env: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"写入 .env 失败，配置未修改: {e}")
    return APIResponse(data={"changed": changed, "unchanged": [k for k in updates if k not in changed]})
