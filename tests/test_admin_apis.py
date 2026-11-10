"""
admin 新增接口测试（针对运行中的服务，默认 http://localhost:8000）

覆盖：
- PUT    /admin/agent/{agent_id}           覆盖式更新 persona / subject context / relationships
- DELETE /admin/agent/{agent_id}/pfc       清空前额叶短时记忆
- POST   /admin/agent/{agent_id}/memories  新增 brief / episodic / knowledge 记忆

使用随机 agent_id，结束后清理该 agent 在 Redis / Qdrant / MongoDB 中的全部数据，不影响已有 agent。
运行：python3.11 tests/test_admin_apis.py [base_url]
"""
import asyncio
import os
import sys
import uuid

import httpx
from motor.motor_asyncio import AsyncIOMotorClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.client.redis_client import redis_client  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.service.prefrontal_lobe_memory import stm_key, compress_key  # noqa: E402

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/") + "/admin"
AGENT = f"tmp_admin_test_{uuid.uuid4().hex[:8]}"
OTHER = f"tmp_admin_other_{uuid.uuid4().hex[:8]}"  # 只用来验证 PFC 清空不误伤别的 agent

results = []


def check(name: str, cond: bool, detail=""):
    results.append((name, cond))
    print(f"  {'✅' if cond else '❌'} {name}" + ("" if cond else f"  -> {detail}"))


async def get_agent(c: httpx.AsyncClient) -> dict:
    r = await c.get(f"{BASE}/agent/{AGENT}")
    return r.json()["data"]


async def test_update(c: httpx.AsyncClient):
    print("\n[接口1] PUT /admin/agent/{agent_id}")

    r = await c.put(f"{BASE}/agent/no_such_agent_{uuid.uuid4().hex[:6]}", json={"persona_system1": "x"})
    check("不存在的 agent 返回 404", r.status_code == 404, r.text)

    r = await c.put(f"{BASE}/agent/{AGENT}", json={"overall_state": {"mood": 1.5}})
    check("非法参数（mood=1.5）返回 422", r.status_code == 422, r.text)

    before = await get_agent(c)
    r = await c.put(f"{BASE}/agent/{AGENT}", json={})
    after = await get_agent(c)
    check("空 body：updated_fields 为空", r.status_code == 200 and r.json()["data"]["updated_fields"] == [], r.text)
    check("空 body：数据不变", {k: before[k] for k in ("persona", "relationships", "situation", "overall_state", "thoughts")}
          == {k: after[k] for k in ("persona", "relationships", "situation", "overall_state", "thoughts")})

    # persona 部分覆盖
    r = await c.put(f"{BASE}/agent/{AGENT}", json={"persona_system2": "  新的S2  "})
    d = await get_agent(c)
    check("只传 system2：updated_fields 正确", r.json()["data"]["updated_fields"] == ["persona_system2"], r.text)
    check("system2 被覆盖且去首尾空格", d["persona"]["system2"] == "新的S2", d["persona"])
    check("system1 / speaking_manner 保持不变",
          d["persona"]["system1"] == "旧S1" and d["persona"]["speaking_manner"] == "旧说话方式", d["persona"])

    await c.put(f"{BASE}/agent/{AGENT}", json={"persona_system1": "新的S1", "speaking_manner": "新说话方式"})
    d = await get_agent(c)
    check("system1 + speaking_manner(System3) 覆盖",
          d["persona"] == {"system1": "新的S1", "system2": "新的S2", "speaking_manner": "新说话方式"}, d["persona"])

    # subject context：只覆盖 overall_state，其他保持
    await c.put(f"{BASE}/agent/{AGENT}", json={"overall_state": {"mood": 0.1, "stress": 0.9, "energy": 0.2}})
    d = await get_agent(c)
    check("overall_state 覆盖", d["overall_state"] == {"mood": 0.1, "stress": 0.9, "energy": 0.2}, d["overall_state"])
    check("只改 overall_state 时 thoughts 保持", [t["content"] for t in d["thoughts"]] != [] and
          set(t["content"] for t in d["thoughts"]) == {"旧念头A", "旧念头B"}, d["thoughts"])
    check("只改 overall_state 时 situation 保持", d["situation"]["current_activity"] == "旧活动", d["situation"])

    # situation 整体替换：旧的 attention_focus 不应残留
    await c.put(f"{BASE}/agent/{AGENT}", json={"situation": {"current_activity": "新活动", "ongoing_states": ["感冒中"]}})
    d = await get_agent(c)
    s = d["situation"]
    check("situation 整体替换", s["current_activity"] == "新活动" and s["ongoing_states"] == ["感冒中"], s)
    check("situation 旧字段不残留（attention_focus / events 被清空）",
          not s.get("attention_focus") and not s.get("time_sensitive_events"), s)

    # thoughts / pending_tasks 覆盖
    await c.put(f"{BASE}/agent/{AGENT}", json={
        "thoughts": [{"content": "新念头", "intensity": 0.8}],
        "pending_tasks": [{"content": "新任务", "priority": 0.9, "deadline": "2026-12-01"}],
    })
    d = await get_agent(c)
    check("thoughts 覆盖（旧的 2 条消失，只剩新 1 条）",
          [(t["content"], t["intensity"]) for t in d["thoughts"]] == [("新念头", 0.8)], d["thoughts"])
    check("pending_tasks 覆盖（旧的 2 条消失，只剩新 1 条）",
          [(t["content"], t["priority"], t["deadline"]) for t in d["pending_tasks"]] == [("新任务", 0.9, "2026-12-01")],
          d["pending_tasks"])

    # relationships 覆盖
    await c.put(f"{BASE}/agent/{AGENT}", json={"relationships": [
        {"person_name": "王五", "relationship": "朋友", "description": "认识十年"}]})
    d = await get_agent(c)
    check("relationships 覆盖（张三/李四消失，只剩王五）",
          d["relationships"] == [{"person_name": "王五", "relationship": "朋友", "description": "认识十年"}],
          d["relationships"])

    await c.put(f"{BASE}/agent/{AGENT}", json={"relationships": []})
    d = await get_agent(c)
    check("relationships 传 [] 清空全部关系", d["relationships"] == [], d["relationships"])


async def test_clear_pfc(c: httpx.AsyncClient):
    print("\n[接口2] DELETE /admin/agent/{agent_id}/pfc")

    await redis_client.rpush(stm_key(AGENT), '{"speaker":"张三","content":"你好"}', '{"speaker":"我","content":"嗨"}')
    await redis_client.rpush(compress_key(AGENT), '{"speaker":"summary","content":"之前聊过天气"}')
    await redis_client.rpush(stm_key(OTHER), '{"speaker":"x","content":"别的agent"}')
    check("预置 PFC 数据（stm + compress）",
          await redis_client.client.exists(stm_key(AGENT), compress_key(AGENT)) == 2)

    r = await c.delete(f"{BASE}/agent/{AGENT}/pfc")
    check("返回 deleted_keys=2", r.status_code == 200 and r.json()["data"]["deleted_keys"] == 2, r.text)
    check("stm 和 compress 都已删除",
          await redis_client.client.exists(stm_key(AGENT), compress_key(AGENT)) == 0)
    check("其他 agent 的 PFC 不受影响", await redis_client.llen(stm_key(OTHER)) == 1)

    r = await c.delete(f"{BASE}/agent/{AGENT}/pfc")
    check("重复清空幂等（deleted_keys=0）", r.status_code == 200 and r.json()["data"]["deleted_keys"] == 0, r.text)

    d = await get_agent(c)
    check("清空 PFC 不影响 persona", d["persona"]["system1"] == "新的S1", d["persona"])


async def test_add_memories(c: httpx.AsyncClient):
    print("\n[接口3] POST /admin/agent/{agent_id}/memories")

    r = await c.post(f"{BASE}/agent/no_such_agent_{uuid.uuid4().hex[:6]}/memories",
                     json={"brief_memories": [{"content": "x"}]})
    check("不存在的 agent 返回 404", r.status_code == 404, r.text)

    r = await c.post(f"{BASE}/agent/{AGENT}/memories", json={})
    check("空请求返回 400", r.status_code == 400, r.text)

    r = await c.post(f"{BASE}/agent/{AGENT}/memories", json={"knowledge_memories": [{"title": "t", "content": "c"}]})
    check("knowledge 缺 domain 返回 422", r.status_code == 422, r.text)

    before = await get_agent(c)
    payload = {
        "brief_memories": [
            {"content": "我喜欢喝美式咖啡，不加糖", "tags": ["偏好"]},
            {"content": "我住在杭州西湖区", "tags": ["基本信息"]},
        ],
        "episodic_memories": [
            {"content": "上周六和王五去爬了泰山，看到了日出", "emotion_charge": 0.7,
             "tags": ["旅行"], "occurred_at": "2026-11-01"},
        ],
        "knowledge_memories": [
            {"title": "Python装饰器", "content": "装饰器是接收函数并返回新函数的高阶函数",
             "domain": "编程", "sub_domain": "Python", "proficiency": 0.8, "tags": ["python"]},
        ],
    }
    r = await c.post(f"{BASE}/agent/{AGENT}/memories", json=payload)
    data = r.json().get("data", {})
    check("返回统计 brief 2/2、episodic 1/1、knowledge 1/1",
          r.status_code == 200 and data["brief"]["success"] == 2 and data["episodic"]["success"] == 1
          and data["knowledge"]["success"] == 1, r.text)

    d = await get_agent(c)
    check("brief 新增 2 条", len(d["brief_memories"]) - len(before["brief_memories"]) == 2, d["brief_memories"])
    check("brief 原文入库（infer=False 不改写）",
          {"我喜欢喝美式咖啡，不加糖", "我住在杭州西湖区"} <= {m["content"] for m in d["brief_memories"]},
          [m["content"] for m in d["brief_memories"]])
    epi = [m for m in d["episodic_memories"] if "泰山" in m["content"]]
    check("episodic 写入且 metadata 带 emotion_charge / occurred_at / tags",
          len(epi) == 1 and epi[0]["emotion_charge"] == 0.7 and epi[0]["occurred_at"] == "2026-11-01"
          and epi[0]["tags"] == ["旅行"], epi)
    kn = [k for k in d["knowledge_memories"] if k["title"] == "Python装饰器"]
    check("knowledge 写入且字段正确",
          len(kn) == 1 and kn[0]["domain"] == "编程" and kn[0]["sub_domain"] == "Python"
          and kn[0]["proficiency"] == 0.8, kn)

    # 相同内容再传一次，应作为新记录追加，不去重不覆盖
    r = await c.post(f"{BASE}/agent/{AGENT}/memories", json=payload)
    d2 = await get_agent(c)
    check("重复提交相同内容：brief 再 +2（不去重）", len(d2["brief_memories"]) == len(d["brief_memories"]) + 2,
          len(d2["brief_memories"]))
    check("重复提交相同内容：episodic 再 +1", len(d2["episodic_memories"]) == len(d["episodic_memories"]) + 1,
          len(d2["episodic_memories"]))
    check("重复提交相同内容：knowledge 再 +1", len(d2["knowledge_memories"]) == len(d["knowledge_memories"]) + 1,
          len(d2["knowledge_memories"]))
    check("新增记忆不影响 Redis 数据（persona / overall_state）",
          d2["persona"] == d["persona"] and d2["overall_state"] == d["overall_state"])


async def cleanup():
    keys = await redis_client.scan_keys(f"*{AGENT}*") + await redis_client.scan_keys(f"*{OTHER}*")
    if keys:
        await redis_client.delete(*keys)

    async with httpx.AsyncClient(base_url=settings.QDRANT_URL, trust_env=False, timeout=30) as q:
        for coll, field in (("brief_memories", "user_id"), ("episodic_memories", "user_id"),
                            ("knowledge_memories", "agent_id")):
            await q.post(f"/collections/{coll}/points/delete?wait=true",
                         json={"filter": {"must": [{"key": field, "match": {"value": AGENT}}]}})

    mongo = AsyncIOMotorClient(settings.MONGODB_URI)
    res = await mongo[settings.MONGODB_DB_NAME][settings.MONGODB_COLLECTION_LIFE_EVENT].delete_many(
        {"agent_id": {"$in": [AGENT, OTHER]}})
    mongo.close()
    print(f"\n清理完成：Redis {len(keys)} 个 key，Qdrant 3 个集合，MongoDB {res.deleted_count} 条事件")


async def main():
    print(f"服务: {BASE}\n测试 agent: {AGENT}")
    async with httpx.AsyncClient(timeout=120, trust_env=False) as c:
        try:
            r = await c.post(f"{BASE}/initialize_agent", json={
                "agent_id": AGENT,
                "persona_system1": "旧S1", "persona_system2": "旧S2", "speaking_manner": "旧说话方式",
                "relationships": [{"person_name": "张三", "relationship": "同事"},
                                  {"person_name": "李四", "relationship": "上级", "description": "比较严格"}],
                "situation": {"current_activity": "旧活动", "attention_focus": ["旧关注点"],
                              "time_sensitive_events": [{"event": "旧事件", "deadline": "2026-12-31"}]},
                "overall_state": {"mood": 0.5, "stress": 0.3, "energy": 0.7},
                "thoughts": [{"content": "旧念头A"}, {"content": "旧念头B"}],
                "pending_tasks": [{"content": "旧任务A"}, {"content": "旧任务B"}],
            })
            assert r.status_code == 200, f"初始化测试 agent 失败: {r.text}"

            await test_update(c)
            await test_clear_pfc(c)
            await test_add_memories(c)
        finally:
            await cleanup()
            await redis_client.close()

    passed = sum(ok for _, ok in results)
    print(f"\n结果：{passed}/{len(results)} 通过")
    for name, ok in results:
        if not ok:
            print(f"  失败: {name}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
