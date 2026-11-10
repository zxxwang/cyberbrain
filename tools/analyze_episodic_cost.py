"""
分析现有测试日志中的 Episodic 改写成本
基于 react_20261026_175919.md 的数据
"""

# 从测试报告提取数据
EPISODIC_AVG_TIME = 3175  # ms
EPISODIC_P50_TIME = 3524  # ms
EPISODIC_P90_TIME = 4401  # ms

TOTAL_AVG_TIME = 11067  # ms
SYSTEM1_AVG_TIME = 3976  # ms
SYSTEM2_AVG_TIME = 5692  # ms
SYSTEM3_AVG_TIME = 1912  # ms

# Episodic 组成估算
QUERY_REWRITE_NUM = 4  # 当前改写数量
VECTOR_SEARCH_TIME_PER_QUERY = 200  # ms（估算，Qdrant很快）
LLM_REWRITE_TIME_ESTIMATE = 2500  # ms（4个query的改写时间）

print("=" * 80)
print("Episodic 改写成本分析")
print("=" * 80)
print()

print("【当前数据】（来自 react_20261026_175919.md）")
print(f"  Episodic 平均耗时: {EPISODIC_AVG_TIME}ms")
print(f"  Episodic P50: {EPISODIC_P50_TIME}ms")
print(f"  Episodic P90: {EPISODIC_P90_TIME}ms")
print(f"  改写数量: {QUERY_REWRITE_NUM}")
print()

print("【成本拆解估算】")
# 假设：1个原始query + 3个改写query = 4次向量检索
vector_search_total = VECTOR_SEARCH_TIME_PER_QUERY * QUERY_REWRITE_NUM
rewrite_overhead = EPISODIC_AVG_TIME - vector_search_total

print(f"  向量检索耗时: {QUERY_REWRITE_NUM} queries × {VECTOR_SEARCH_TIME_PER_QUERY}ms = {vector_search_total}ms")
print(f"  改写+其他耗时: {rewrite_overhead}ms")
print(f"  改写占比: {rewrite_overhead / EPISODIC_AVG_TIME * 100:.1f}%")
print()

print("【优化方案对比】")
scenarios = [
    {"name": "当前（4 queries改写）", "queries": 4, "rewrite": True},
    {"name": "降低改写（2 queries）", "queries": 2, "rewrite": True},
    {"name": "不改写（1 query）", "queries": 1, "rewrite": False},
]

for scenario in scenarios:
    if scenario["rewrite"]:
        # 改写耗时与query数量成正比
        rewrite_time = LLM_REWRITE_TIME_ESTIMATE * (scenario["queries"] - 1) / (QUERY_REWRITE_NUM - 1)
    else:
        rewrite_time = 0
    
    vector_time = VECTOR_SEARCH_TIME_PER_QUERY * scenario["queries"]
    total_time = rewrite_time + vector_time
    time_saved = EPISODIC_AVG_TIME - total_time
    
    print(f"\n  方案：{scenario['name']}")
    print(f"    改写耗时: {rewrite_time:.0f}ms")
    print(f"    向量检索: {vector_time}ms")
    print(f"    总耗时: {total_time:.0f}ms")
    print(f"    节省: {time_saved:.0f}ms ({time_saved / EPISODIC_AVG_TIME * 100:.1f}%)")
    
    # 计算整体RT影响
    new_total_rt = TOTAL_AVG_TIME - EPISODIC_AVG_TIME + total_time
    rt_saved = TOTAL_AVG_TIME - new_total_rt
    print(f"    整体RT: {new_total_rt:.0f}ms (节省{rt_saved:.0f}ms, {rt_saved / TOTAL_AVG_TIME * 100:.1f}%)")

print()
print("=" * 80)
print("【关键问题】")
print("=" * 80)
print()
print("1. 改写是否真的提升召回质量？")
print("   - 需要实验验证：4 queries vs 1 query 的召回覆盖率差异")
print("   - 如果差异 < 5%，改写就是纯浪费")
print()
print("2. 改写为何这么慢（2500ms）？")
print("   - LLM生成4个变体query")
print("   - 使用的是哪个模型？Doubao-lite还是Doubao-pro？")
print("   - 有没有缓存？高频query重复改写浪费")
print()
print("3. 立即可做的优化（无需实验）：")
print("   - ✅ 加缓存：高频query改写结果缓存1小时")
print("   - ✅ 降低改写数量：4 → 2（节省50%改写时间）")
print("   - ⚠️  完全关闭改写：需要先验证召回质量损失")
print()

print("【推荐优化路径】")
print()
print("第1步：立即优化（不影响召回质量）")
print("  - 给 query_rewrite_service 加缓存（已有_cache但TTL=1h）")
print("  - 检查缓存命中率")
print("  预期收益：高频场景节省 2000ms+")
print()
print("第2步：降低改写数量（小实验）")
print("  - QUERY_REWRITE_NUM: 4 → 2")
print("  - 快速对比几个场景的召回质量")
print("  预期收益：节省 1200ms（38%）")
print()
print("第3步：完全关闭改写（大实验）")
print("  - QUERY_REWRITE_NUM: 2 → 1（或直接关闭改写调用）")
print("  - 跑完整72场景测试对比召回质量")
print("  预期收益：节省 2300ms（73%）")
print()
print("=" * 80)
