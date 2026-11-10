#!/usr/bin/env python3.11
"""
林诗雨对话测试

每轮记录：
- 回复、路径
- System1 情绪决策全字段
- overall_state（对话前/后，通过 /admin/agent 读取真实值）
- 记忆召回（brief / episodic / knowledge）
- 各阶段耗时（服务端 timings_ms + 客户端墙钟时间）

输出：
- tests/test_report_<ts>.json   原始数据
- tests/test_report_<ts>.md     完整对话 + 性能统计
"""
import json
import statistics
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional

BASE_URL = "http://localhost:8000"
AGENT_ID = "lin_shiyu_001"
DISPLAY_NAME = "林诗雨"  # 报告里角色的显示名；其他用例通过 run() 覆盖
# _update_subjectivity_after_dialogue 是 create_task 异步执行的，等它写完再读 overall_state
STATE_SETTLE_SECONDS = 3

TEST_SCENARIOS = [
    {
        "name": "场景1：《局外人》的连续深入讨论",
        "speaker": "文学青年小陈",
        "context": "一个真心热爱文学的年轻人，想深入理解加缪",
        "dialogues": [
            "诗雨姐，我最近在读加缪的《局外人》，但总觉得读不懂默尔索为什么那么冷漠。你怎么理解这个人物的？",
            "你说默尔索是拒绝虚伪，但他在母亲葬礼上不哭，这不是太过分了吗？难道爱也是虚伪的吗？",
            "那你第一次读的时候也不理解吗？是什么让你后来突然懂了？",
            "你提到'荒诞'，这个词我不太理解。什么是荒诞？",
        ],
    },
    {
        "name": "场景2：创作心得的深入交流",
        "speaker": "青年作家阿雨",
        "context": "30岁的青年作家，也在经历创作瓶颈",
        "dialogues": [
            "诗雨，我最近也写不出东西了。每天坐在电脑前，脑子里一片空白。你说我是不是已经失去写作的才能了？",
            "你说'还没找到真正想说的东西'，但我怎么知道什么是真正想说的？",
            "你的散文《雨天书店》我读了好几遍，特别喜欢那篇写沙发的。那些细节是真实的还是虚构的？你是怎么写得那么有质感的？",
        ],
    },
    {
        "name": "场景3：《看不见的城市》读书会准备",
        "speaker": "沈清文",
        "context": "与沈教授讨论周五读书会",
        "dialogues": [
            "小林，周五的读书会你准备得怎么样了？《看不见的城市》可不好讲，很多人读不进去。",
            "你说55座城市都是威尼斯，这个理解很有意思。你第一次读的时候就懂了吗？",
            "如果有读者问'这本书到底在说什么'，你会怎么回答？",
        ],
    },
    {
        "name": "场景4：关于书店意义的深度对谈",
        "speaker": "徐安然",
        "context": "深夜对话，徐安然的困惑也触发了林诗雨的思考",
        "dialogues": [
            "诗雨姐，我一直很好奇，你为什么要开这家书店？明明知道会亏本，为什么还要坚持？",
            "你说书店是精神栖息地，但这个时代还有多少人需要这种东西？大家不都在刷手机、追剧吗？",
            "我有时候觉得，你守着这家书店，会不会只是一种自我感动？",
        ],
    },
    {
        "name": "场景5：推荐书籍的连续对话",
        "speaker": "迷茫的读者",
        "context": "30岁，人生遇到困境，想通过阅读寻找答案",
        "dialogues": [
            "我最近很迷茫，工作和生活都找不到意义。朋友推荐我读书，但我不知道该读什么。你能推荐一些书吗？",
            "你为什么推荐《活着》？这本书名听起来好沉重。",
            "那《挪威的森林》呢？我听说这是爱情小说？",
            "你说的这些书，它们之间有什么共同的东西吗？",
        ],
    },
]

EMOTION_FIELDS = [
    "emotion", "intensity", "situation", "impulse", "engagement", "conflict",
    "user_intent", "quick_action", "action_confidence",
    "think_prob", "needs_memory", "needs_reasoning", "requires_knowledge",
]

# 服务端阶段（brief→system1 串行；episodic、knowledge_prefetch 与之并行）
STAGES = ["embedding", "brief", "system1", "knowledge_prefetch", "episodic", "episodic_wait",
          "system2", "system3", "total"]


# ============================================================
# HTTP（用 curl，httpx 在本机会被系统代理劫持返回 503）
# ============================================================

def _curl(args: List[str], timeout: int) -> Optional[Dict]:
    try:
        r = subprocess.run(["curl", "-s", "--max-time", str(timeout)] + args,
                           capture_output=True, text=True, timeout=timeout + 5)
        if r.returncode == 0 and r.stdout:
            return json.loads(r.stdout)
    except Exception as e:
        print(f"    ⚠️ curl 异常: {e}")
    return None


def call_react(signal: str, speaker: str) -> Optional[Dict]:
    body = json.dumps({"agent_id": AGENT_ID, "signal": signal, "speaker": speaker}, ensure_ascii=False)
    return _curl(["-X", "POST", f"{BASE_URL}/brain/react",
                  "-H", "Content-Type: application/json", "-d", body], timeout=180)


def get_overall_state() -> Dict:
    resp = _curl([f"{BASE_URL}/admin/agent/{AGENT_ID}"], timeout=60)
    if not resp:
        return {}
    data = resp.get("data") or {}
    return {
        "overall_state": data.get("overall_state") or {},
        "thoughts": len(data.get("thoughts") or []),
        "pending_tasks": len(data.get("pending_tasks") or []),
    }


# ============================================================
# 格式化
# ============================================================

def _mem_text(m: Dict) -> str:
    return m.get("memory") or m.get("title") or m.get("content") or ""


def _mem_score(m: Dict) -> float:
    return m.get("rerank_score") or m.get("score") or 0.0


def fmt_memories(rec: Dict) -> List[str]:
    lines = []
    for label, key in [("Brief", "brief"), ("Episodic", "episodic"), ("Knowledge", "knowledge")]:
        items = rec["memories"][key]
        if not items:
            lines.append(f"    {label}: 无")
            continue
        lines.append(f"    {label} ({len(items)}条):")
        for i, m in enumerate(items, 1):
            lines.append(f"      {i}. [{m['score']:.2f}] {m['text']}")
    return lines


def fmt_emotion(e: Dict) -> List[str]:
    def f(k, nd=3):
        v = e.get(k)
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
    return [
        f"    情绪: {e.get('emotion')}  强度: {f('intensity')}  冲动: {e.get('impulse')}",
        f"    情境: {e.get('situation')}  用户意图: {e.get('user_intent')}",
        f"    投入度: {f('engagement')}  冲突度: {f('conflict')}",
        f"    快速行动: {e.get('quick_action')}  置信度: {f('action_confidence')}",
        f"    门控: think_prob={f('think_prob')}  needs_memory={f('needs_memory')}  "
        f"needs_reasoning={f('needs_reasoning')}  requires_knowledge={e.get('requires_knowledge')}",
    ]


def fmt_state(before: Dict, after: Dict) -> List[str]:
    lines = []
    b, a = before.get("overall_state", {}), after.get("overall_state", {})
    for k in ["mood", "stress", "energy"]:
        bv, av = b.get(k), a.get(k)
        if isinstance(bv, (int, float)) and isinstance(av, (int, float)):
            d = av - bv
            arrow = "↑" if d > 0.001 else ("↓" if d < -0.001 else "→")
            lines.append(f"    {k:<6}: {bv:.3f} → {av:.3f}  ({arrow}{d:+.3f})")
        else:
            lines.append(f"    {k:<6}: {bv} → {av}")
    lines.append(f"    thoughts: {before.get('thoughts')} → {after.get('thoughts')}   "
                 f"pending_tasks: {before.get('pending_tasks')} → {after.get('pending_tasks')}")
    return lines


def fmt_timings(rec: Dict) -> List[str]:
    t = rec["timings_ms"]
    g = lambda k: t.get(k, 0.0)
    chain = g("brief") + g("system1")
    lines = [
        f"    [串行] 查询 embedding: {g('embedding'):.0f}",
        f"    [并行段] brief→system1 链: {chain:.0f}  (brief {g('brief'):.0f} + system1 {g('system1'):.0f})",
        f"             knowledge_prefetch: {g('knowledge_prefetch'):.0f}   episodic: {g('episodic'):.0f}",
        f"    [串行段] episodic_wait: {g('episodic_wait'):.0f}   system2: {g('system2'):.0f}   system3: {g('system3'):.0f}",
        f"    服务端 total: {g('total'):.0f}   客户端墙钟: {rec['wall_ms']:.0f}   "
        f"未计时开销: {rec['untracked_ms']:.0f}",
    ]
    return lines


def render_turn(rec: Dict) -> List[str]:
    lines = [
        f"第 {rec['turn']} 轮",
        f"  👤 {rec['speaker']}: {rec['user_input']}",
        f"  💬 {DISPLAY_NAME}: {rec['agent_response']}",
        f"  路径: {rec['route']}   长度: {len(rec['agent_response'])}字",
        "  🎭 System1:",
        *fmt_emotion(rec["emotion"]),
        "  💭 overall_state（对话前 → 对话后）:",
        *fmt_state(rec["state_before"], rec["state_after"]),
        "  🧠 记忆召回:",
        *fmt_memories(rec),
        "  ⏱️ 耗时(ms):",
        *fmt_timings(rec),
    ]
    return lines


# ============================================================
# 执行
# ============================================================

def run_turn(scenario: Dict, turn: int, signal: str, state_before: Dict) -> Optional[Dict]:
    t0 = time.perf_counter()
    resp = call_react(signal, scenario["speaker"])
    wall_ms = (time.perf_counter() - t0) * 1000
    if not resp or "response" not in resp:
        print(f"  ❌ 第{turn}轮请求失败: {str(resp)[:200]}")
        return None

    time.sleep(STATE_SETTLE_SECONDS)
    state_after = get_overall_state()

    timings = resp.get("timings_ms") or {}
    mems = {}
    for key in ["brief", "episodic", "knowledge"]:
        mems[key] = [{"score": _mem_score(m), "text": _mem_text(m)} for m in (resp.get(key) or [])]

    return {
        "scenario": scenario["name"],
        "turn": turn,
        "speaker": scenario["speaker"],
        "user_input": signal,
        "agent_response": resp.get("response", ""),
        "silent": resp.get("silent"),
        "route": resp.get("route"),
        "emotion": {k: (resp.get("emotion") or {}).get(k) for k in EMOTION_FIELDS},
        "decision": resp.get("decision"),
        "state_before": state_before,
        "state_after": state_after,
        "memories": mems,
        "timings_ms": timings,
        "wall_ms": wall_ms,
        # 服务端 total 只覆盖召回→System3；speaker_info/subjectivity/pfc/memory_gate/add_todo/life_event 都在外面
        "untracked_ms": wall_ms - timings.get("total", 0.0),
    }


def perf_summary(records: List[Dict]) -> List[str]:
    def stat(values: List[float]) -> str:
        if not values:
            return "n/a"
        vs = sorted(values)
        p90 = vs[min(len(vs) - 1, int(round(0.9 * (len(vs) - 1))))]
        return f"avg {statistics.mean(vs):7.0f}  p50 {statistics.median(vs):7.0f}  p90 {p90:7.0f}  max {vs[-1]:7.0f}  (n={len(vs)})"

    lines = ["## 性能统计（ms）", "", "```"]
    for st in STAGES:
        vals = [r["timings_ms"][st] for r in records if st in r["timings_ms"] and r["timings_ms"][st] > 0]
        lines.append(f"{st:<20} {stat(vals)}")
    lines.append(f"{'brief+system1 链':<18} {stat([r['timings_ms'].get('brief', 0) + r['timings_ms'].get('system1', 0) for r in records])}")
    lines.append(f"{'客户端墙钟':<16} {stat([r['wall_ms'] for r in records])}")
    lines.append(f"{'未计时开销':<16} {stat([r['untracked_ms'] for r in records])}")
    lines.append("```")

    routes: Dict[str, List[Dict]] = {}
    for r in records:
        routes.setdefault(r["route"], []).append(r)
    lines += ["", "### 按路径", "", "| 路径 | 次数 | 墙钟 avg | 服务端 total avg | system2 avg | system3 avg |",
              "|---|---|---|---|---|---|"]
    for route, rs in routes.items():
        avg = lambda k: statistics.mean([x["timings_ms"].get(k, 0) for x in rs])
        lines.append(f"| {route} | {len(rs)} | {statistics.mean([x['wall_ms'] for x in rs]):.0f} | "
                     f"{avg('total'):.0f} | {avg('system2'):.0f} | {avg('system3'):.0f} |")

    # 墙钟占比（服务端阶段按关键路径拆：brief→system1 链 vs knowledge_prefetch 取较大者，再加串行段）
    lines += ["", "### 墙钟构成（平均）", ""]
    def m(k):
        return statistics.mean([r["timings_ms"].get(k, 0) for r in records])
    wall = statistics.mean([r["wall_ms"] for r in records])
    parallel = statistics.mean([max(r["timings_ms"].get("brief", 0) + r["timings_ms"].get("system1", 0),
                                    r["timings_ms"].get("knowledge_prefetch", 0)) for r in records])
    parts = [("查询 embedding（串行，召回前）", m("embedding")),
             ("并行段（brief→system1 / knowledge 取大）", parallel),
             ("episodic_wait", m("episodic_wait")),
             ("system2", m("system2")),
             ("system3", m("system3")),
             ("未计时（前置+后置串行步骤）", statistics.mean([r["untracked_ms"] for r in records]))]
    for name, v in parts:
        lines.append(f"- {name}: {v:.0f}ms（{v / wall * 100:.1f}%）")
    lines.append(f"- 墙钟合计: {wall:.0f}ms")
    return lines


def run(agent_id: str, display_name: str, scenarios: List[Dict], report_prefix: str) -> None:
    """通用执行入口：其他角色的测试用例 import 本模块后调用 run() 即可复用全部记录与统计"""
    global AGENT_ID, DISPLAY_NAME
    AGENT_ID, DISPLAY_NAME = agent_id, display_name

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"{display_name}对话测试 {ts}")
    initial_state = get_overall_state()
    print(f"初始 overall_state: {initial_state}")

    records: List[Dict] = []
    md: List[str] = [f"# {display_name}对话测试 {ts}", "", f"初始状态: `{json.dumps(initial_state, ensure_ascii=False)}`", ""]
    state = initial_state

    for sc in scenarios:
        print(f"\n{'=' * 90}\n{sc['name']}  ({sc['context']})\n{'=' * 90}")
        md += [f"## {sc['name']}", "", f"背景：{sc['context']}", ""]
        for i, signal in enumerate(sc["dialogues"], 1):
            rec = run_turn(sc, i, signal, state)
            if not rec:
                continue
            state = rec["state_after"]
            records.append(rec)
            block = render_turn(rec)
            print("\n" + "\n".join(block))
            md += ["```", *block, "```", ""]

    md += perf_summary(records)
    json_path = f"tests/{report_prefix}_{ts}.json"
    md_path = f"tests/{report_prefix}_{ts}.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"test_time": ts, "initial_state": initial_state, "records": records}, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("\n" + "\n".join(perf_summary(records)))
    print(f"\n报告: {md_path}\n数据: {json_path}")


def main():
    run(AGENT_ID, DISPLAY_NAME, TEST_SCENARIOS, "test_report")


if __name__ == "__main__":
    main()
