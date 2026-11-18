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
import re
import logging
from datetime import datetime
from typing import Dict, List
from app.client.doubao_client import doubao_client
from app.client.web_search_client import web_search_client
from app.config.settings import settings
from app.service.subjectivity_service import subjectivity_service
from app.repository.memory_repository import memory_repository
from app.service.knowledge_memory_service import knowledge_memory_service
from app.repository.relationship_repository import relationship_repository
from app.repository.capability_repository import CAP_WEB_SEARCH, capability_repository
from app.service.agent_life_event_service import agent_life_event_service

logger = logging.getLogger(__name__)


class TaskExecutor:
    """任务执行器：用 LLM 模拟执行待办任务"""

    def __init__(self):
        # 正在执行的任务：对话中立即执行的任务和 worker 定时检查可能同时碰到同一个任务
        self._running: set = set()

    @staticmethod
    async def can_execute_immediately(agent_id: str, task: Dict) -> bool:
        """
        对话中判断为"现在就做"的待办，哪些可以立即执行：目前只有联网搜索任务，且 agent 要有联网能力。
        - 普通任务的"执行"是 LLM 编一句完成结果，立即执行会让下一轮对话就出现没做过的事
        - 对话中生成的社交任务，联系对象固定是当前说话人（正在对话的人），立即"主动联系"没有意义，
          且说话人姓名不一定是 agent_id，会以人名创建出游离数据
        """
        return bool(task.get("needs_web_search")) and await capability_repository.has(agent_id, CAP_WEB_SEARCH)

    async def run_task(self, agent_id: str, task: Dict) -> Dict:
        """执行任务，成功后从待办中移除；失败则保留在待办里，由 worker 下次检查时重试"""
        task_id = task["task_id"]
        if task_id in self._running:
            logger.info(f"Task {task_id} is already running, skip")
            return {"success": False, "result": "任务正在执行"}
        self._running.add(task_id)
        try:
            result = await self.execute_task(agent_id, task)
            if result.get("success"):
                await subjectivity_service.complete_task(agent_id, task_id)
            elif result.get("cancel"):
                # 执行器判定这个任务做不了（如联系对象不存在），取消掉，避免 worker 每次检查都重试
                await subjectivity_service.cancel_task(agent_id, task_id)
            return result
        finally:
            self._running.discard(task_id)

    async def execute_task(self, agent_id: str, task: Dict) -> Dict:
        """执行任务并返回结果"""
        try:
            # 需要联网补充信息的任务：真实搜索，不再让 LLM 编造执行结果。
            # 先于"主动交互"判断：决策模型常把"查一下补进知识库"同时判为需要联系人（汇报），
            # 但这类任务的实质是获取信息，结果写进记忆后主人随时可以问
            # 能力在执行时再查一次：开关关闭前生成的联网任务、通过接口直接写入的 needs_web_search 任务都不经过漏斗，
            # 没有联网能力时按普通任务执行
            if task.get("needs_web_search"):
                if await capability_repository.has(agent_id, CAP_WEB_SEARCH):
                    return await self._execute_web_search_task(agent_id, task)
                logger.info(f"Agent {agent_id} has no web_search capability, run as a normal task: {task['content']}")

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

            # 不再把模拟结果写入 Knowledge：模拟执行是 LLM 编的一句话，不是真实获取的知识，
            # 写进去会污染知识库。需要获取知识的任务由 needs_web_search 走真实的联网搜索
            # 同理也不写处境：以前会把"已与某某完成深谈"这类编出来的完成结果追加进持续状态，
            # 处境里堆满已经结束、甚至没发生过的事。处境只由对话（Memory Gate）和接口维护

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

    # ---------------- 联网搜索任务 ----------------

    # 正反两个查询：只搜一次时，结果偏向哪一边，整理出的知识就偏向哪一边（如只搜到软文就会得出"有用"）
    WEB_QUERY_PROMPT = """把下面这个待办改写成两个搜索引擎查询词，每个不超过30字，只写要查的内容，不带"帮我""查一下"：
第一行：直接查这件事的查询词
第二行：查它的风险、局限、反面证据或争议的查询词
待办：{task}
只输出两行查询词。"""

    # 来源类型 → 可信度。只看搜索相关性会把高相关的软文当成可靠来源
    SOURCE_TYPE_CONFIDENCE = {"学术研究": 0.75, "官方或专家共识": 0.75, "新闻媒体": 0.55, "商家或营销": 0.35}
    SOURCE_TYPE_DEFAULT = "新闻媒体"

    WEB_DIGEST_PROMPT = """你在为自己的知识库整理资料。待办：{task}

以下是搜索结果（编号、标题、链接、摘录）：
{results}

第一步，判断每条结果的来源类型，每行一条，格式"编号 类型"，类型只能是：学术研究 / 官方或专家共识 / 新闻媒体 / 商家或营销
（商家或营销包括：卖相关产品的商家、品牌官网、诊所、带货或推广文章、自媒体软文）
然后输出一行 ===

第二步，只根据摘录整理出 1-{max_items} 条知识，每条之间用一行 === 分隔，每条三行：
标题（15字内）
领域（如：抗衰老科学、健康产品、编程）
正文（{max_chars}字内）。要求：待办要求列出清单、步骤或分类时，逐项完整列出，不要用"等""等N项"省略；写明说法出自哪类来源（如"据某专家共识""某商家宣称"）；正反证据都有时两边都写；证据只来自动物实验或小样本时写明；不要把商家或营销的说法写成结论。句末用 [1][2] 标注出自哪条结果

摘录里没有能回答这个待办的内容时，第二步只输出：无"""

    async def _execute_web_search_task(self, agent_id: str, task: Dict) -> Dict:
        """
        联网搜索 → 根据搜索结果整理知识 → 写入 Knowledge 与 Episodic → 完成任务。

        - 一个待办搜两次：直接查询 + 风险/反面证据查询，合并去重
        - 整理时标注每条来源的类型，知识可信度按引用来源的类型给，而不是按搜索相关性
        - 未配置 key 或搜索失败：返回失败，任务保持待执行（下次检查再试），不编造结果
        - 搜到了但没有能回答问题的内容：任务完成，但只记一条"查了没查到"的经历，不写知识
        """
        content = task["content"]
        if not web_search_client.enabled:
            logger.warning(f"Web search task kept pending (TAVILY_API_KEY not configured): {content}")
            return {"success": False, "result": "未配置联网搜索，任务保持待执行"}

        raw = (await doubao_client.generate(self.WEB_QUERY_PROMPT.format(task=content), max_tokens=80)).strip()
        queries = [q.strip().lstrip("12.、:：").strip() for q in raw.splitlines() if q.strip()][:2] or [content]
        try:
            batches = await asyncio.gather(*[web_search_client.search(q) for q in queries])
        except Exception as e:
            logger.error(f"Web search failed for task {task['task_id']}: {e}", exc_info=True)
            await agent_life_event_service.record_task_execution(
                agent_id=agent_id, task_id=task["task_id"], task_content=content,
                result={"success": False, "error": f"web search failed: {e}", "queries": queries})
            return {"success": False, "result": f"联网搜索失败：{e}"}

        # 合并两次搜索的结果，按链接去重，保留搜索顺序（直接查询的结果在前）
        results, seen = [], set()
        for batch in batches:
            for r in batch:
                if r["url"] not in seen:
                    seen.add(r["url"])
                    results.append(r)

        source_types, knowledge = (await self._digest_search_results(content, results)) if results else ({}, [])

        saved = []
        for k in knowledge:
            idx = [i for i in k["refs"] if 0 <= i < len(results)] or [0]
            sources = [results[i] for i in idx]
            types = [source_types.get(i, self.SOURCE_TYPE_DEFAULT) for i in idx]
            # 多个来源时取最可靠的那个：有一条学术或官方来源支撑，就不该被同时引用的软文拉低
            confidence = max(self.SOURCE_TYPE_CONFIDENCE[t] for t in types)
            body = k["content"] + "\n来源：" + "；".join(
                f"{r['title']}（{t}）{r['url']}" for r, t in zip(sources, types))
            kid = await asyncio.to_thread(
                knowledge_memory_service.add_knowledge,
                agent_id=agent_id,
                title=k["title"],
                content=body,
                knowledge_type="concept",
                domain=k["domain"],
                confidence=confidence,
                source="learning",
                tags=["web_search", "task_completion"],
            )
            if kid:
                saved.append(k["title"])

        if saved:
            summary = f"为了「{content}」上网查了资料，整理进知识库：{'、'.join(saved)}"
        elif results:
            summary = f"为了「{content}」上网查了资料，但没找到能回答这个问题的可靠内容"
        else:
            summary = f"为了「{content}」上网查了资料，没有搜到结果"

        source_list = [{"url": r["url"], "type": source_types.get(i, self.SOURCE_TYPE_DEFAULT)}
                       for i, r in enumerate(results)]
        await memory_repository.add(
            agent_id=agent_id,
            messages=[{"role": "user", "content": summary}],
            memory_type="episodic",
            metadata={
                "task_id": task["task_id"],
                "type": "web_search",
                "task_content": content,
                "queries": queries,
                "sources": [r["url"] for r in results],
                "speaker": "self",
                "timestamp": datetime.now().isoformat(),
            },
        )
        await agent_life_event_service.record_task_execution(
            agent_id=agent_id, task_id=task["task_id"], task_content=content,
            result={"success": True, "result": summary, "queries": queries,
                    "knowledge_saved": saved, "sources": source_list},
        )
        logger.info(f"Web search task done for {agent_id}: {summary}")

        # 告诉委托这件事的人查完了（按关系表里的类型：真人写入发件箱，agent 调用 react；陌生人不通知）
        try:
            from app.service.outreach_service import outreach_service
            details = "\n".join(f"- 《{k['title']}》{k['content'][:150]}" for k in knowledge if k["title"] in saved)
            await outreach_service.notify_task_done(agent_id, task, summary, details)
        except Exception as e:
            logger.error(f"Notify task done failed for {agent_id}: {e}", exc_info=True)

        return {"success": True, "result": summary, "knowledge_saved": saved, "queries": queries,
                "sources": source_list}

    async def _digest_search_results(self, task_content: str, results: List[Dict]) -> tuple:
        """
        整理搜索结果。返回 (来源类型 {结果下标: 类型}, 知识条目 [{"title", "domain", "content", "refs"}])，
        摘录里没有可用内容时知识条目为空
        """
        listing = "\n\n".join(
            f"[{i + 1}] {r['title']}\n{r['url']}\n{r['content'][:800]}" for i, r in enumerate(results)
        )
        resp = (await doubao_client.generate(
            self.WEB_DIGEST_PROMPT.format(task=task_content, results=listing,
                                          max_items=settings.WEB_KNOWLEDGE_MAX_ITEMS,
                                          max_chars=settings.WEB_KNOWLEDGE_MAX_CHARS),
            # 每条正文按 1 字约 1.5 token 估算，再加来源分类和格式的余量
            max_tokens=int(settings.WEB_KNOWLEDGE_MAX_ITEMS * settings.WEB_KNOWLEDGE_MAX_CHARS * 1.5) + 600)).strip()
        blocks = [b.strip() for b in resp.split("===")]

        source_types = {}
        for line in (blocks[0].splitlines() if blocks else []):
            m = re.match(r"^\s*\[?(\d+)\]?\s*[.、:：]?\s*(\S+)", line)
            if m and m.group(2) in self.SOURCE_TYPE_CONFIDENCE:
                source_types[int(m.group(1)) - 1] = m.group(2)

        items = []
        for block in blocks[1:]:
            if block.rstrip("。.") in ("无", "没有"):
                continue
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            if len(lines) < 3:
                continue
            # 模型偶尔会带上"标题：""领域：""正文："前缀，去掉
            strip = lambda t: re.sub(r"^\s*(标题|领域|正文)\s*[:：]\s*", "", t).strip()
            title, domain, body = strip(lines[0]), strip(lines[1]), strip(" ".join(lines[2:]))
            refs = sorted({int(n) - 1 for n in re.findall(r"\[(\d+)\]", body)})
            items.append({"title": title[:40], "domain": domain[:20], "content": body, "refs": refs})
        return source_types, items[:settings.WEB_KNOWLEDGE_MAX_ITEMS]

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

    async def _execute_dialogue_task(self, agent_id: str, task: Dict) -> Dict:
        """
        执行社交任务：主动联系任务里的那个人。

        对方是谁、用什么方式联系，由关系表里的 kind 决定（不靠名字猜）：
        真人写入发件箱等前端取走，agent 用关系里记录的 agent_id 调用对方的 react。
        对方不在关系表里时取消任务（不主动联系陌生人，也不会以人名为 agent_id 创建游离数据）
        """
        related_person = task.get("related_person")
        if not related_person:
            return {"success": False, "cancel": True, "result": "未指定联系对象"}
        if not await relationship_repository.get(agent_id, related_person):
            logger.warning(f"Dialogue task cancelled: '{related_person}' is not in relationships of {agent_id}")
            return {"success": False, "cancel": True,
                    "result": f"无法主动联系「{related_person}」：不在关系表里"}

        from app.service.outreach_service import outreach_service
        plan = (f"我主动联系对方，为了这件事：{task['content']}。\n"
                f"直接说明来意，一两句话。这是我主动开口，对方刚才没有说话。")
        message = await outreach_service.deliver(agent_id, related_person, plan, source="task", ref=task["task_id"])
        if not message:
            return {"success": False, "result": "没有说出口"}

        reply = message.get("reply", "")
        summary = (f"我主动联系{related_person}：{message['content'][:60]}"
                   + (f" 对方回复：{reply[:60]}" if reply else "（已留言，等对方查看）"))
        await memory_repository.add(
            agent_id=agent_id,
            messages=[{"role": "user", "content": summary}],
            memory_type="episodic",
            metadata={
                "task_id": task["task_id"],
                "type": "dialogue_task",
                "target": related_person,
                "delivered_via": message.get("delivered_via"),
                "timestamp": datetime.now().isoformat(),
            },
        )
        logger.info(f"✅ Dialogue task completed: {summary}")
        return {"success": True, "result": summary, "dialogue": message}


task_executor = TaskExecutor()
