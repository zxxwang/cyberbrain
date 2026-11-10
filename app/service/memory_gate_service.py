"""
Memory Gate Service: 记忆价值评估服务

职责：
- 使用 Doubao System One 评估输入信息的记忆价值（importance）
- 使用 Doubao System One 分类记忆类型（memory_type）
- 评估并持久化重要记忆到相应脑区
- 根据对话内容更新 Situation 和 Overall State
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List

from app.client.doubao_client import doubao_client
from app.repository.memory_repository import memory_repository
from app.service.subjectivity_service import subjectivity_service
from app.service.agent_life_event_service import agent_life_event_service

logger = logging.getLogger(__name__)


class MemoryGateService:
    """记忆门控服务"""

    def __init__(self):
        self.enabled = True

    async def evaluate(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict:
        """
        评估记忆价值和类型

        Args:
            messages: 对话消息列表 [{"role": "user", "content": "..."}]

        Returns:
            {
                "importance": 0-4,  # 价值评分
                "memory_type": "EPISODIC/SEMANTIC",  # 记忆类型
                "should_save": bool,  # 是否值得保存
                "storage_recommendation": "DISCARD/WORKING/LONG_TERM",  # 存储建议
                "ttl": int,  # 如果是工作记忆，建议的过期时间（秒）
            }
        """
        if not self.enabled:
            return self._default_result()

        try:
            # 提取消息文本
            text = self._extract_text(messages)

            # 使用 Doubao System One 批量分类（一次请求，两个问题）
            response = await doubao_client.decide(
                state=text,
                questions={
                    "importance": {
                        "type": "score",
                        "instructions": """判断这条消息告诉了 AI 关于用户的什么信息，值得记住多久？
                                            核心原则：
                                            - 提问（用户想知道什么）≠ 信息（用户是什么样的人）
                                            - 意向（打算/想）< 事实（是/做了）
                                            - 极短语句按字面最常见含义理解
                                            - 忽略礼貌词，提取主体事实""",
                        "criteria": [
                            "纯社交/语气，无关用户本身",
                            "用户在问问题，或仅是此刻瞬时状态",
                            "近期确定的事（明天做X），或不确定的意向（打算/想）",
                            "用户的稳定特征（我是/我喜欢），或已完成的事件（昨天做了），或通用经验规律（X比Y好，X适合Y）",
                            "安全/健康约束（过敏/禁忌）",
                        ],
                    },
                    "memory_type": {
                        "type": "choice",
                        "instructions": "判断这条信息最适合的记忆类型。注意区分：对话/事件/经历(EPISODIC) vs 稳定的事实/属性(SEMANTIC)。",
                        "criteria": {
                            "EPISODIC": "对话内容、已发生的具体事件、个人经历、时间性事实、约定、承诺、请求。关键词：说、跟我、告诉、昨天、刚才、完成了、去了、见了、记得、明天。示例：小王说记得汇报、昨天见了客户、完成了项目、约定明天开会。",
                            "SEMANTIC": "相对稳定的身份、偏好、长期目标、用户资料、持续有效的事实或约束（不是对话）。关键词：我是、我喜欢、我住在、我的。示例：我是工程师、我住在北京、我对花生过敏。",
                        },
                    },
                },
            )

            # 解析结果（System One返回格式：{"answers": {"question_key": {...}}）
            answers = response.get("answers", {})
            importance_raw = answers.get("importance", {}).get("score", 0.0)
            importance = round(importance_raw)  # 转为 0-4 整数

            memory_type = answers.get("memory_type", {}).get("choice", "SEMANTIC")

            # 决策逻辑
            should_save = importance > 0
            storage_recommendation = self._determine_storage(importance)
            ttl = self._determine_ttl(importance)

            result = {
                "importance": importance,
                "importance_raw": importance_raw,  # 保留原始分数用于调试
                "memory_type": memory_type,
                "should_save": should_save,
                "storage_recommendation": storage_recommendation,
                "ttl": ttl,
                "input_text": text,  # 用于调试
            }

            logger.info(
                f"Memory gate evaluation: importance={importance} ({importance_raw:.2f}), "
                f"type={memory_type}, storage={storage_recommendation}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to evaluate memory gate: {e}", exc_info=True)
            return self._default_result()

    def _extract_text(self, messages: List[Dict[str, str]]) -> str:
        """提取消息文本"""
        texts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            if content:
                # 只保留用户和助手的消息
                if role in ["user", "assistant"]:
                    texts.append(content)
        return " ".join(texts)

    def _determine_storage(self, importance: int) -> str:
        """
        决定存储建议

        Args:
            importance: 0-4

        Returns:
            "DISCARD" | "WORKING" | "LONG_TERM"
        """
        if importance == 0:
            return "DISCARD"
        elif importance <= 2:
            return "WORKING"  # Redis 工作记忆
        else:
            return "LONG_TERM"  # Qdrant 长期记忆

    def _determine_ttl(self, importance: int) -> int:
        """
        决定工作记忆的 TTL（秒）

        Args:
            importance: 0-4

        Returns:
            TTL 秒数（仅对 WORKING 有意义）
        """
        ttl_map = {
            0: 0,          # 不保存
            1: 86400,      # 1天
            2: 259200,     # 3天
            3: 0,          # 长期记忆，不需要 TTL
            4: 0,          # 长期记忆，不需要 TTL
        }
        return ttl_map.get(importance, 0)

    def _default_result(self) -> Dict:
        """默认结果（服务不可用或出错时）"""
        return {
            "importance": 2,  # 保守：默认中等价值
            "importance_raw": 2.0,
            "memory_type": "SEMANTIC",
            "should_save": True,
            "storage_recommendation": "WORKING",
            "ttl": 259200,  # 3天
            "input_text": "",
        }

    async def batch_evaluate(
        self,
        batch: List[List[Dict[str, str]]],
    ) -> List[Dict]:
        """
        批量评估（用于测试或批量处理）

        Args:
            batch: 多条消息列表

        Returns:
            评估结果列表
        """
        results = []
        for messages in batch:
            result = await self.evaluate(messages)
            results.append(result)
        return results

    async def evaluate_and_persist(
        self,
        agent_id: str,
        signal: str,
        speaker: str,
    ) -> None:
        """
        评估输入信息的记忆价值，并持久化到相应脑区

        此方法适合异步调用，不阻塞主流程

        Args:
            agent_id: Agent ID
            signal: 用户输入信号
            speaker: 说话人
        """
        try:
            messages = [{"role": "user", "content": signal}]
            # 1. 评估价值，同时做重激活：对方再次提到某件事时，相似的情景记忆权重恢复（不论这句话本身是否值得记）
            #    重激活必须在本次写入之前做，否则会命中刚写入的这条
            gate_result, _ = await asyncio.gather(
                self.evaluate(messages=messages),
                memory_repository.reactivate_related(agent_id, messages, memory_type="episodic"),
            )

            importance = gate_result["importance"]
            memory_type = gate_result["memory_type"]

            # 2. 分层持久化策略
            # - importance 0: 丢弃，不记录
            # - importance 1-2: 仅 PFC (工作记忆)，不写 Episodic
            # - importance 3-4: Episodic (情景记忆)，重要事件
            # - SEMANTIC: Brief (核心信条)，高度抽象的事实

            if importance < 1:
                logger.debug(
                    f"Memory gate: signal importance={importance} (type={memory_type}), "
                    f"discarded (threshold=1)"
                )
                return

            # importance 1-2: 只记录到 PFC，不写入长期记忆
            if importance < 3 and memory_type == "EPISODIC":
                logger.info(
                    f"Memory gate: signal importance={importance}, "
                    f"recorded to PFC only (not persistent)"
                )
                # 注：PFC 记录已在 cyber_brain.react() 中完成
                return

            logger.info(
                f"Memory gate: signal importance={importance} type={memory_type}, "
                f"persisting to long-term memory..."
            )

            # 3. 根据类型持久化到相应脑区
            if memory_type == "EPISODIC":
                # 🔧 将 speaker 信息嵌入到内容中，避免 Mem0 抽取时丢失
                if speaker and speaker != "user" and speaker != "self":
                    # Agent 间对话：标注说话人
                    enriched_content = f"{speaker}对我说：{signal}"
                else:
                    # 用户对话或自言自语
                    enriched_content = signal

                # 持久化到情景记忆
                await memory_repository.add(
                    agent_id=agent_id,
                    messages=[{"role": "user", "content": enriched_content}],
                    memory_type="episodic",
                    metadata={
                        "speaker": speaker,
                        "importance": importance,
                        "source": "memory_gate",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                logger.info(f"Persisted to EPISODIC: {enriched_content[:50]}...")

            elif memory_type == "SEMANTIC":
                # 🔧 将 speaker 信息嵌入到内容中
                if speaker and speaker != "user" and speaker != "self":
                    enriched_content = f"{speaker}对我说：{signal}"
                else:
                    enriched_content = signal

                # 持久化到语义记忆（Brief）
                await memory_repository.add(
                    agent_id=agent_id,
                    messages=[{"role": "user", "content": enriched_content}],
                    memory_type="brief",
                    metadata={
                        "speaker": speaker,
                        "importance": importance,
                        "category": "user_profile",
                        "source": "memory_gate",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                logger.info(f"Persisted to SEMANTIC(Brief): {enriched_content[:50]}...")

            # 注意：Knowledge 类型已从 Memory Gate 移除
            # 知识习得通过 initialize agent 时显式调用 knowledge_memory_service.add_knowledge()

            # 4. 更新 Situation 和 Overall State（基于重要信息）
            await self._update_situation_and_state(agent_id, signal, importance, memory_type)

        except Exception as e:
            logger.error(f"Failed to evaluate and persist memory: {e}", exc_info=True)

    async def _update_situation_and_state(
        self,
        agent_id: str,
        signal: str,
        importance: int,
        memory_type: str
    ):
        """
        基于重要信息更新 Situation 和 Overall State

        策略：
        - 只有 importance >= 3 的信息才可能影响状态
        - EPISODIC 类型：可能包含新的事件或状态变化
        - SEMANTIC 类型：可能包含稳定特征或长期状态

        Args:
            agent_id: Agent ID
            signal: 用户输入信号
            importance: 重要性评分 (0-4)
            memory_type: 记忆类型（EPISODIC/SEMANTIC）
        """
        try:
            # 获取当前 situation
            current_situation = await subjectivity_service.get_situation(agent_id)
            if not current_situation:
                # 如果从未设置过，初始化空状态
                current_situation = {
                    "current_activity": "",
                    "time_sensitive_events": [],
                    "ongoing_states": [],
                    "attention_focus": []
                }

            # 使用 LLM 判断这条信号是否需要更新 situation
            update_decision = await doubao_client.decide(
                state=f"当前状态：{current_situation}\n\n新信息：{signal}",
                questions={
                    "should_update": {
                        "type": "choice",
                        "instructions": "这条新信息是否需要更新当前状态？",
                        "criteria": {
                            "yes": "包含明确的新事件、新计划、状态变化、关注点变化",
                            "no": "只是普通对话，不涉及状态变化"
                        }
                    }
                }
            )

            should_update = update_decision.get("answers", {}).get("should_update", {}).get("choice") == "yes"

            if not should_update:
                logger.debug(f"Skip situation update for signal: {signal[:30]}...")
                # 即使不更新 situation，也可能需要调整 overall_state
                await subjectivity_service.adjust_overall_state(
                    agent_id,
                    mood_delta=0.0,
                    stress_delta=0.0,
                    reason="no_update"
                )
                return

            # 如果需要更新，使用 LLM 生成新的 situation
            from app.config.prompts import SITUATION_UPDATE_PROMPT
            from app.service.prefrontal_lobe_memory import format_records

            prompt = SITUATION_UPDATE_PROMPT.format(
                current_situation=str(current_situation),
                records=format_records([{"type": "raw", "role": "user", "speaker": "user", "content": signal}])
            )

            response = (await doubao_client.generate(prompt, max_tokens=1000)).strip()

            # 去除可能的markdown代码块
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            import json
            updated_situation = json.loads(response)

            # 验证字段
            required_keys = {"current_activity", "time_sensitive_events", "ongoing_states", "attention_focus"}
            if not required_keys.issubset(updated_situation.keys()):
                logger.warning(f"LLM returned incomplete situation: {updated_situation}")
                return

            # 保存更新后的 situation
            await subjectivity_service.replace_situation(agent_id, updated_situation)
            logger.info(f"Updated situation for {agent_id} based on signal: {signal[:30]}...")

            # 根据更新后的 situation 调整 overall_state
            # 简单策略：有 deadline 临近的事件 → stress ↑
            stress_delta = 0.0
            if updated_situation.get("time_sensitive_events"):
                from datetime import datetime, timedelta
                today = datetime.now().date()
                for event in updated_situation["time_sensitive_events"]:
                    deadline_str = event.get("deadline")
                    if deadline_str:
                        deadline = datetime.strptime(deadline_str, "%Y-%m-%d").date()
                        days_left = (deadline - today).days
                        if 0 < days_left <= 3:
                            stress_delta += 0.1  # 3天内的事件，stress +0.1

            change = await subjectivity_service.adjust_overall_state(
                agent_id,
                mood_delta=0.0,
                stress_delta=stress_delta,
                reason="situation_update"
            )
            if stress_delta:
                await agent_life_event_service.record_emotion(
                    agent_id=agent_id,
                    emotion_state={**change, "delta": {"mood": 0.0, "stress": round(stress_delta, 4)}},
                    trigger="situation_update：3天内有时间敏感事件",
                    metadata={"source": "memory_gate"},
                )

        except Exception as e:
            logger.error(f"Failed to update situation and state for {agent_id}: {e}", exc_info=True)


# 单例
memory_gate_service = MemoryGateService()
