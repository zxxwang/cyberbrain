"""
Task Executor: 用 LLM 模拟任务执行结果（修正版）

修正：
1. 使用 memory_repository.add() 而不是 episodic_memory_service
2. 不写入 PFC（任务执行不是对话）
3. 学习类任务写入 Knowledge
4. 重要任务可能影响 Situation（通过 LLM 判断）
5. 🆕 社交任务主动发起对话
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict
from app.client.doubao_client import doubao_client
from app.service.subjectivity_service import subjectivity_service
from app.repository.memory_repository import memory_repository
from app.service.knowledge_memory_service import knowledge_memory_service
from app.repository.relationship_repository import relationship_repository
from app.service.agent_life_event_service import agent_life_event_service

logger = logging.getLogger(__name__)


class TaskExecutor:
    """任务执行器：用 LLM 模拟执行待办任务"""

    async def execute_task(self, agent_id: str, task: Dict) -> Dict:
        """执行任务并返回结果"""
        try:
            # 🆕 判断是否需要主动交互
            if task.get("interaction_type") == "dialogue":
                return await self._execute_dialogue_task(agent_id, task)
            
            # 1. LLM 模拟执行
            result = await self._simulate_execution(task)

            # 2. 写入 Episodic Memory
            await memory_repository.add(
                agent_id=agent_id,
                messages=[{"role": "self", "content": f"[自主完成] {result}"}],
                memory_type="episodic",
                metadata={
                    "task_id": task["task_id"],
                    "type": "task_completion",
                    "task_content": task["content"],
                    "speaker": "self",
                    "timestamp": datetime.now().isoformat()
                }
            )

            # 3. 学习类任务额外写入 Knowledge
            if self._is_learning_task(task["content"]):
                await self._save_to_knowledge(agent_id, task["content"], result)

            # 4. 检查是否影响 Situation
            await self._update_situation_if_needed(agent_id, task, result)

            logger.info(f"Task executed for {agent_id}: {task['content']} → {result}")

            # 🆕 记录任务执行事件
            await agent_life_event_service.record_task_execution(
                agent_id=agent_id,
                task_id=task["task_id"],
                task_content=task["content"],
                result={"success": True, "result": result}
            )

            return {"success": True, "result": result}

        except Exception as e:
            logger.error(f"Failed to execute task: {e}", exc_info=True)

            # 🆕 记录任务执行失败事件
            await agent_life_event_service.record_task_execution(
                agent_id=agent_id,
                task_id=task.get("task_id", "unknown"),
                task_content=task.get("content", ""),
                result={"success": False, "error": str(e)}
            )

            return {"success": False, "result": f"执行失败：{str(e)}"}

    async def _simulate_execution(self, task: Dict) -> str:
        """LLM 模拟任务执行"""
        prompt = f"""你是一个 Agent，需要完成一个待办任务。

任务内容：{task['content']}
优先级：{task.get('priority', 0.5):.2f}
截止日期：{task.get('deadline', '无')}
相关人：{task.get('related_person', '无')}

请模拟你完成了这个任务，用第一人称描述结果（20字以内）。

示例：
- "学习 Redis" → "我学习了 Redis 的基本用法，了解了数据结构"
- "给张三发邮件" → "我给张三发了邮件，告知了项目进度"

只输出结果，不要其他内容。"""

        result = await doubao_client.generate(prompt, max_tokens=100)
        return result.strip()

    def _is_learning_task(self, task_content: str) -> bool:
        """判断是否为学习类任务"""
        learning_keywords = ["学习", "了解", "研究", "学", "查", "搜索", "查看", "阅读"]
        return any(kw in task_content for kw in learning_keywords)

    async def _save_to_knowledge(self, agent_id: str, task_content: str, result: str):
        """学习结果保存到 Knowledge"""
        try:
            domain = self._extract_domain(task_content)
            # add_knowledge 是同步的（远程 embedding + qdrant 写入），放线程池避免阻塞事件循环
            knowledge_id = await asyncio.to_thread(
                knowledge_memory_service.add_knowledge,
                agent_id=agent_id,
                title=task_content,
                content=result,
                knowledge_type="concept",
                domain=domain,
                source="learning",
                tags=["task_completion"],
            )
            if knowledge_id:
                logger.info(f"Saved to knowledge: domain={domain}, id={knowledge_id}")
            else:
                logger.warning(f"Failed to save learning result to knowledge: {task_content}")
        except Exception as e:
            logger.error(f"Failed to save to knowledge: {e}", exc_info=True)

    def _extract_domain(self, task_content: str) -> str:
        """提取学习领域"""
        for prefix in ["学习", "了解", "研究", "学", "查", "搜索", "查看"]:
            if prefix in task_content:
                domain = task_content.replace(prefix, "").strip()
                domain = domain.replace("的使用", "").replace("使用", "").strip()
                return domain if domain else "通用知识"
        return "通用知识"

    async def _update_situation_if_needed(self, agent_id: str, task: Dict, result: str):
        """判断任务完成是否影响 Situation"""
        try:
            # 使用 LLM 判断是否需要更新 situation
            decision = await self._should_update_situation(task, result)

            if not decision["should_update"]:
                return

            # 更新 situation
            situation = await subjectivity_service.get_situation(agent_id)
            if not situation:
                situation = {}

            # 根据任务类型更新
            if decision["update_type"] == "current_activity":
                situation["current_activity"] = decision["new_value"]
            elif decision["update_type"] == "time_sensitive_event_done":
                # 标记事件完成
                events = situation.get("time_sensitive_events", [])
                for event in events:
                    if event.get("description") in task["content"]:
                        event["status"] = "completed"
                situation["time_sensitive_events"] = events
            elif decision["update_type"] == "ongoing_state":
                # 更新持续状态
                states = situation.get("ongoing_states", [])
                states.append(decision["new_value"])
                situation["ongoing_states"] = states[-5:]  # 只保留最近5个

            await subjectivity_service.replace_situation(agent_id, situation)
            logger.info(f"Updated situation: {decision['update_type']} = {decision.get('new_value', '')}")

        except Exception as e:
            logger.error(f"Failed to update situation: {e}", exc_info=True)

    async def _should_update_situation(self, task: Dict, result: str) -> Dict:
        """LLM 判断是否需要更新 situation"""
        prompt = f"""判断任务完成是否影响 Agent 的处境。

                任务：{task['content']}
                结果：{result}
                
                判断：
                1. 是否需要更新处境？
                2. 更新什么？
                   - current_activity: 当前主要活动（如"正在开发游戏"）
                   - time_sensitive_event_done: 时间敏感事件完成
                   - ongoing_state: 持续状态（如"刚学会了新技能"）
                
                输出JSON：
                {{
                  "should_update": true/false,
                  "update_type": "current_activity" | "time_sensitive_event_done" | "ongoing_state" | null,
                  "new_value": "新状态描述" | null
                }}
                
                示例：
                - "学习 Redis" → {{"should_update": true, "update_type": "ongoing_state", "new_value": "刚学习了 Redis"}}
                - "给张三发邮件" → {{"should_update": false}}
                - "准备游戏发布" → {{"should_update": true, "update_type": "current_activity", "new_value": "正在准备游戏发布"}}
                
                只输出JSON。"""

        try:
            resp = await doubao_client.generate(prompt, max_tokens=150)
            resp = resp.strip()
            if resp.startswith("```"):
                resp = "\n".join(resp.split("\n")[1:-1])
            return json.loads(resp)
        except:
            return {"should_update": False}


    async def _execute_dialogue_task(self, agent_id: str, task: Dict) -> Dict:
        """
        执行社交任务：主动发起对话
        
        流程：
        1. 解析目标 Agent ID
        2. 生成开场白
        3. 调用目标 Agent 的 react()
        4. 记录对话结果
        """
        # 1. 解析目标 Agent ID
        related_person = task.get("related_person")
        if not related_person:
            return {"success": False, "result": "未指定联系对象"}

        # related_person 在对话场景中已经是 agent_id（speaker）
        target_agent_id = related_person

        logger.info(f"Target agent resolved: {related_person} -> {target_agent_id}")
        
        # 2. 生成开场白
        opening = await self._generate_opening(agent_id, task)
        
        logger.info(f"🔥 Agent {agent_id} 主动联系 {target_agent_id} ({related_person}): {opening}")
        
        # 3. 发起对话
        from app.service.cyber_brain import CyberBrain
        cyber_brain = CyberBrain()
        
        try:
            response_data = await cyber_brain.react(
                agent_id=target_agent_id,
                signal=opening,
                speaker=agent_id  # 发起方作为说话人
            )
            
            response = response_data["response"]
            
        except Exception as e:
            logger.error(f"Failed to initiate dialogue: {e}", exc_info=True)
            return {"success": False, "result": f"对话发起失败：{str(e)}"}

        # 4. 记录到发起方的记忆
        # 注：主动对话的记忆应该经过 memory_gate 评估，而非直接写入
        # 但由于这是任务执行（重要事件），可以直接写入 episodic
        # 🔧 修改：只记录完整的对话事件，不记录碎片
        dialogue_summary = (
            f"我主动联系{related_person or target_agent_id}汇报任务。"
            f"我说：{opening[:50]}... "
            f"对方回复：{response[:50]}..."
        )

        await memory_repository.add(
            agent_id=agent_id,
            messages=[{"role": "self", "content": dialogue_summary}],
            memory_type="episodic",
            metadata={
                "task_id": task["task_id"],
                "type": "dialogue_task",
                "target_agent_id": target_agent_id,
                "opening": opening,
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # 5. 更新 Situation
        situation = await subjectivity_service.get_situation(agent_id)
        if not situation:
            situation = {}
        
        if "recent_contacts" not in situation:
            situation["recent_contacts"] = []
        
        situation["recent_contacts"].append({
            "person": related_person,
            "agent_id": target_agent_id,
            "date": datetime.now().date().isoformat(),
            "topic": task["content"]
        })
        situation["recent_contacts"] = situation["recent_contacts"][-10:]
        
        await subjectivity_service.replace_situation(agent_id, situation)
        
        result = f"我主动联系了{related_person}，对方回复：{response[:30]}..."
        
        logger.info(f"✅ Dialogue task completed: {result}")
        
        return {
            "success": True,
            "result": result,
            "dialogue": {
                "opening": opening,
                "response": response,
                "target_agent_id": target_agent_id
            }
        }

    async def _generate_opening(self, agent_id: str, task: Dict) -> str:
        """生成开场白"""
        # 获取发起方的上下文
        brief = await memory_repository.search_brief(agent_id, task['content'], top_k=5)
        situation = await subjectivity_service.get_situation(agent_id)

        # 获取对方的关系信息
        related_person = task.get('related_person', '对方')
        try:
            relationship = await relationship_repository.get_relationship(agent_id, related_person)
            if relationship:
                target_desc = relationship  # 如："上级（小王）"
            else:
                target_desc = related_person
        except:
            target_desc = related_person

        prompt = f"""你现在要主动联系{target_desc}。

                    原因：{task['content']}
                    
                    你的最近记忆：
                    {chr(10).join([m.get('content', '')[:50] for m in brief[:3]]) if brief else '无'}
                    
                    请生成一句开场白（15字以内），表明你主动联系的目的。
                    只输出开场白，不要其他内容。"""

        opening = await doubao_client.generate(prompt, max_tokens=50)
        return opening.strip()



task_executor = TaskExecutor()
