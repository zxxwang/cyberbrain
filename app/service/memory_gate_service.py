"""
Memory Gate Service: 记忆价值评估服务

职责（对方的每句话，实时、后台）：
- 评估记忆价值（importance）和类型（memory_type），作为标签写到工作记忆的这条记录上
- 重激活相似的情景记忆
- 把说话人本人的稳定个人信息融合进关系表的 profile
- 根据重要信息更新 Situation 和 Overall State

不写 Episodic / Brief：经历由工作记忆整理时提取（episode_consolidation_service），信念只由反思写入
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

from app.client.doubao_client import doubao_client
from app.repository.memory_repository import memory_repository
from app.service.prefrontal_lobe_memory import prefrontal_lobe_memory
from app.repository.relationship_repository import PROFILE_SLOTS, relationship_repository
from app.service.subjectivity_service import subjectivity_service
from app.service.agent_life_event_service import agent_life_event_service

logger = logging.getLogger(__name__)


class MemoryGateService:
    """记忆门控服务"""

    def __init__(self):
        self.enabled = True

    # 槽位判定规则（抽取提示词用）：写清楚"什么算、什么不算"，避免"在浦东上班"被抽成住址
    PROFILE_SLOT_RULES = {
        "姓名": "说话人的真实姓名（\"我叫王五\"）。称呼（老王、姐）不算",
        "称呼": "别人怎么叫自己（\"大家都叫我老王\"）",
        "年龄": "**说话人本人**的年龄或出生年份。家人的年龄归家庭成员",
        "生日": "**说话人本人**的生日或出生月日",
        "住址": "说话人住在哪里（\"我住在北京\"\"我搬到上海了\"\"我家在杭州\"）。工作地点、孩子学校都不算",
        "籍贯": "老家、家乡、土生土长的地方",
        "职业": "说话人做什么工作（\"我是老师\"\"我做后端\"）",
        "单位": "在哪个公司/机构（\"我在字节跳动\"\"我在市医院上班\"）",
        "职位": "职务级别（\"我是团队负责人\"\"我是主治医师\"）。与职称无关的工作内容不算",
        "家庭成员": "家人及其年龄、上学、职业等信息（\"我女儿五岁\"\"我妈有高血压\"）。一人一条，把信息写在同一 value 里",
        "健康状况": "**说话人本人**的疾病、慢性病、身体状况",
        "过敏与禁忌": "说话人本人的过敏原、忌口、不能碰的东西（\"我对花生过敏\"\"我不吃辣\"）。一项一条",
        "长期偏好": "说话人本人长期稳定的喜好（\"我只喝美式\"\"我喜欢下雨天\"）。一次性的选择不算",
        "忌讳": "说话人明确不希望被做的事、不愿聊的话题",
        "重要日期": "对说话人有意义且每年会到的日期（结婚纪念日、忌日）。一次性的日程不算",
        "习惯": "说话人长期反复的行为（\"我每天六点起床\"\"我习惯熬夜\"）。一次性的行为不算",
    }

    # 个人信息抽取：句子 → 受控槽位里的结构化事实
    PROFILE_EXTRACT_PROMPT = """从这句话里，抽出说话人（{speaker}）主动说出的、关于**他自己**的稳定个人信息。

## 规则

1. 只抽说话人本人的事。他转述别人的事（"我老板他家在……"）、问句（"我住哪你知道吗"）、
   情绪和临时状态（"我今天很累""我快忙死了"）都不抽。
2. 归属要分清楚：
   - "我女儿五岁" → 家庭成员 = 女儿五岁。**不是**说话人 5 岁
   - "我在浦东上班" → 工作地点是浦东，**不是**住址；住址只抽"住 / 搬到 / 家在"这类
   - "我在字节做后端" → 单位 = 字节，职业 = 后端工程师
3. value 必须是可以直接比较的最小事实，不要带"现在""我""了"这类词：
   - "我上个月搬到上海了，现在在浦东上班" → 住址 = 上海；单位 = 浦东（若是单位）；不抽工作地点
   - "我对花生过敏，也不能吃辣" → 过敏与禁忌 = 花生、辣（两条）
4. 时间词（"以前住杭州""现在搬到了上海"）：只抽**当前**有效的事实；"以前""原来"那种旧状态不抽
   （旧状态靠库里已有的记录和覆盖机制保留）。
5. 一句话里没有可抽取的就返回空数组。宁可少抽，不要猜。

## 槽位判定

{slot_rules}

## 输出

只输出 JSON，不要任何其他文字：

{{"facts": [{{"slot": "住址", "value": "上海", "text": "搬到上海了"}}]}}

- slot：只能是上面列出的槽位名
- value：最小事实，用于去重和比较
- text：一句自然的话，用于给提示词展示

句子：{signal}"""

    async def _extract_person_facts(self, speaker: str, signal: str) -> Dict[str, list]:
        """抽个人信息并校验槽位，返回 {slot: [{value, text}]}；解析失败返回空"""
        import json as _json
        try:
            raw = await doubao_client.generate(
                self.PROFILE_EXTRACT_PROMPT.format(
                    speaker=speaker,
                    slot_rules="\n".join(f"- {k}：{v}" for k, v in self.PROFILE_SLOT_RULES.items()),
                    signal=signal,
                ),
                max_tokens=300,
            )
            text = (raw or "").strip()
            if "{" in text:
                text = text[text.index("{"): text.rindex("}") + 1]
            facts = _json.loads(text).get("facts") or []
        except Exception as e:
            logger.warning(f"Person fact extraction failed for {speaker}: {e}")
            return {}

        out: Dict[str, list] = {}
        for f in facts:
            slot = (f.get("slot") or "").strip()
            value = (f.get("value") or "").strip()
            if slot in PROFILE_SLOTS and value:
                out.setdefault(slot, []).append({"value": value, "text": (f.get("text") or value).strip()})
        return out

    async def _update_person_profile(self, agent_id: str, speaker: str, signal: str) -> None:
        """
        把这句话里关于说话人的个人信息融合进关系表的 profile。
        陌生人（不在关系表里）直接忽略：路人一上来就说自己的事，人也会听着、但不会正儿八经记下来。
        """
        if not speaker or speaker in ("user", "self", "用户"):
            return  # 匿名说话人不知道是谁的个人信息
        if not await relationship_repository.get(agent_id, speaker):
            logger.info(f"Person fact ignored for {agent_id}: '{speaker}' is not in relationships (stranger)")
            return
        facts = await self._extract_person_facts(speaker, signal)
        if not facts:
            return
        for slot, items in facts.items():
            try:
                await relationship_repository.merge_profile(agent_id, speaker, slot, items)
                logger.info(f"Profile {agent_id}/{speaker}/{slot}: merged {[i['value'] for i in items]}")
            except Exception as e:
                logger.error(f"Merge profile failed ({slot}): {e}", exc_info=True)

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
                    "person_fact": {
                        "type": "noul",
                        "instructions": f"""这句话里，说话人有没有主动说出关于他自己的、稳定的个人信息？
                                                   只看说话人本人（"我……"），不看他转述别人的事，也不看问句。
                                                   可归入这些属性的算：{'、'.join(PROFILE_SLOTS)}
                                                   （"我是工程师""我住在北京""我对花生过敏"算；"你住在哪""今天好累"不算）""",
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
            # 个人信息：只是"这句话里有没有"，是否需要融合由调用方决定（陌生人忽略）
            person_fact = float(answers.get("person_fact", {}).get("noul", 0.0))

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
                "person_fact": person_fact,
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
        record_id: Optional[str] = None,
    ) -> None:
        """
        对方这句话的实时处理（后台执行，不阻塞 react 返回）。

        不再写 Episodic：经历由工作记忆整理时统一提取（episode_consolidation_service），
        那时看得到上下文和"我"说的话。这里只做必须实时的几件事：
        1. 重激活：对方再次提起旧事，相似的情景记忆权重恢复、updated_at 更新（衰减和反思依赖它）
        2. 打标签：把重要度、记忆类型写到工作记忆里这条记录上，整理经历时作为"第一印象"参考
        3. 个人信息：说话人本人主动说出的稳定信息，融合进关系表的 profile，下一轮就能用上
        4. 处境与状态：重要信息（如"明天要交方案"）马上影响处境和压力

        Args:
            record_id: 这句话在工作记忆里的记录 id，用于打标签
        """
        try:
            messages = [{"role": "user", "content": signal}]
            gate_result, _ = await asyncio.gather(
                self.evaluate(messages=messages),
                memory_repository.reactivate_related(agent_id, messages, memory_type="episodic"),
            )
            importance = gate_result["importance"]
            memory_type = gate_result["memory_type"]

            # 第一印象标签：整理经历时 LLM 看得到"当时觉得这句话有多重要"，再结合上下文重新判断
            if record_id:
                await prefrontal_lobe_memory.tag(agent_id, record_id, {
                    "importance": importance, "gate_type": memory_type,
                })

            # 个人信息：只看这句话是否含有说话人的稳定信息，与重要度无关
            if gate_result.get("person_fact", 0.0) >= 0.6:
                try:
                    await self._update_person_profile(agent_id, speaker, signal)
                except Exception as e:
                    logger.error(f"Update person profile failed: {e}", exc_info=True)

            # 处境与状态：只有重要的信息才值得改变处境
            if importance >= 3:
                await self._update_situation_and_state(agent_id, signal, importance, memory_type)

        except Exception as e:
            logger.error(f"Failed to evaluate memory gate: {e}", exc_info=True)

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
