"""
实验：对比不同改写数量对召回质量的影响
方法：直接调用向量检索API，对比召回的记忆内容
"""
import asyncio
import os
import sys
import time
from typing import List, Dict, Any
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.repository.memory_repository import memory_repository
from app.client.doubao_client import doubao_client


# 测试场景
TEST_QUERIES = [
    {
        "id": "Q1",
        "query": "年轻人就该去大厂或者考公求稳定，搞什么创业，纯属瞎折腾",
        "category": "价值观冲突",
        "keywords": ["创业", "独立", "大厂", "稳定", "选择"]
    },
    {
        "id": "Q2", 
        "query": "人活着到底图个啥",
        "category": "人生观",
        "keywords": ["意义", "目标", "人生", "理想"]
    },
    {
        "id": "Q3",
        "query": "最近咋样兄弟，游戏卖得还行不",
        "category": "工作近况",
        "keywords": ["游戏", "销量", "收入", "项目"]
    },
    {
        "id": "Q4",
        "query": "还记得你第一款游戏刚上线那周吗",
        "category": "往事回忆",
        "keywords": ["第一款", "上线", "迷雾之城", "首周"]
    },
    {
        "id": "Q5",
        "query": "Unity项目在手机上卡得要死，DrawCall两千多，咋优化",
        "category": "技术求助",
        "keywords": ["Unity", "优化", "性能", "DrawCall"]
    },
    {
        "id": "Q6",
        "query": "我的游戏上线一周才卖了50份，是不是我不适合干这行",
        "category": "同行安慰",
        "keywords": ["销量", "首周", "137份", "坚持"]
    },
    {
        "id": "Q7",
        "query": "你和李想是怎么认识的",
        "category": "人际关系",
        "keywords": ["李想", "认识", "死党", "大学"]
    },
    {
        "id": "Q8",
        "query": "做独立游戏能赚钱吗",
        "category": "职业咨询",
        "keywords": ["独立游戏", "收入", "赚钱", "盈利"]
    },
]


async def manual_rewrite(query: str, num: int) -> List[str]:
    """手动改写query（模拟query_rewrite_service）"""
    if num == 1:
        return [query]
    
    prompt = f"""请对以下查询进行改写，生成{num-1}个语义相似但表达不同的变体。
改写角度：同义、反义、上下位、隐含意图

原查询：{query}

请输出{num-1}个改写版本，每行一个，不要编号、不要解释。"""
    
    try:
        response = await doubao_client.generate(prompt, max_tokens=500)
        variants = [line.strip() for line in response.strip().split("\n") if line.strip()][:num-1]
        return [query] + variants
    except Exception as e:
        print(f"  改写失败: {e}，使用原query")
        return [query]


async def search_with_rewrite(
    agent_id: str,
    query: str, 
    num_rewrites: int,
    top_k: int = 10
) -> Dict[str, Any]:
    """使用指定数量的改写进行召回"""
    start_time = time.time()
    
    # 改写
    rewrite_start = time.time()
    queries = await manual_rewrite(query, num_rewrites)
    rewrite_time = (time.time() - rewrite_start) * 1000
    
    # 召回
    retrieve_start = time.time()
    all_memories = []
    
    for q in queries:
        try:
            memories = await memory_repository.search_episodic(
                agent_id=agent_id,
                query=q,
                top_k=5
            )
            all_memories.extend(memories)
        except Exception as e:
            print(f"  检索失败 [{q[:20]}...]: {e}")
    
    # 去重
    seen_ids = set()
    unique_memories = []
    for mem in all_memories:
        mem_id = mem.get("id") or mem.get("memory_id") or mem.get("content", "")[:50]
        if mem_id not in seen_ids:
            seen_ids.add(mem_id)
            unique_memories.append(mem)
    
    retrieve_time = (time.time() - retrieve_start) * 1000
    total_time = (time.time() - start_time) * 1000
    
    return {
        "queries": queries,
        "memories": unique_memories[:top_k],
        "num_memories": len(unique_memories[:top_k]),
        "rewrite_time_ms": rewrite_time,
        "retrieve_time_ms": retrieve_time,
        "total_time_ms": total_time
    }


async def compare_rewrite_strategies(agent_id: str = "chen_mo_001"):
    """对比不同改写策略"""
    
    print("=" * 80)
    print("Episodic 改写质量实验")
    print("=" * 80)
    print(f"Agent: {agent_id}")
    print(f"测试场景: {len(TEST_QUERIES)}个")
    print()
    
    results = []
    
    for scenario in TEST_QUERIES:
        print(f"\n[{scenario['id']}] {scenario['category']}")
        print(f"Query: {scenario['query']}")
        print()
        
        # 测试三种策略
        strategies = [
            ("不改写(1)", 1),
            ("少改写(2)", 2),
            ("多改写(4)", 4)
        ]
        
        scenario_results = {"scenario": scenario, "strategies": {}}
        
        for name, num in strategies:
            print(f"  {name}:")
            result = await search_with_rewrite(agent_id, scenario['query'], num)
            
            # 关键词覆盖率
            recalled_texts = " ".join([m.get("content", "") for m in result['memories']])
            keyword_hits = sum(1 for kw in scenario['keywords'] if kw in recalled_texts)
            coverage = keyword_hits / len(scenario['keywords']) if scenario['keywords'] else 0
            
            print(f"    改写: {', '.join([q[:30]+'...' if len(q)>30 else q for q in result['queries']])}")
            print(f"    召回: {result['num_memories']}条")
            print(f"    关键词覆盖: {keyword_hits}/{len(scenario['keywords'])} = {coverage:.1%}")
            print(f"    耗时: {result['total_time_ms']:.0f}ms (改写{result['rewrite_time_ms']:.0f}ms)")
            
            # 显示召回的记忆片段
            if result['memories']:
                print(f"    记忆片段:")
                for i, mem in enumerate(result['memories'][:3], 1):
                    content = mem.get('content', '')[:60]
                    print(f"      {i}. {content}...")
            
            scenario_results['strategies'][name] = {
                "num_rewrites": num,
                "num_memories": result['num_memories'],
                "coverage": coverage,
                "keyword_hits": keyword_hits,
                "total_time_ms": result['total_time_ms'],
                "rewrite_time_ms": result['rewrite_time_ms']
            }
        
        results.append(scenario_results)
        print()
    
    # 汇总统计
    print("\n" + "=" * 80)
    print("汇总统计")
    print("=" * 80)
    
    strategies_stats = defaultdict(lambda: {
        "coverages": [],
        "num_memories": [],
        "times": [],
        "rewrite_times": []
    })
    
    for r in results:
        for strategy_name, data in r['strategies'].items():
            strategies_stats[strategy_name]["coverages"].append(data["coverage"])
            strategies_stats[strategy_name]["num_memories"].append(data["num_memories"])
            strategies_stats[strategy_name]["times"].append(data["total_time_ms"])
            strategies_stats[strategy_name]["rewrite_times"].append(data["rewrite_time_ms"])
    
    print()
    for strategy_name in ["不改写(1)", "少改写(2)", "多改写(4)"]:
        stats = strategies_stats[strategy_name]
        avg_coverage = sum(stats["coverages"]) / len(stats["coverages"])
        avg_memories = sum(stats["num_memories"]) / len(stats["num_memories"])
        avg_time = sum(stats["times"]) / len(stats["times"])
        avg_rewrite_time = sum(stats["rewrite_times"]) / len(stats["rewrite_times"])
        
        print(f"{strategy_name}:")
        print(f"  平均关键词覆盖率: {avg_coverage:.1%}")
        print(f"  平均召回数量: {avg_memories:.1f}条")
        print(f"  平均总耗时: {avg_time:.0f}ms (改写{avg_rewrite_time:.0f}ms)")
        print()
    
    # 对比分析
    print("=" * 80)
    print("结论")
    print("=" * 80)
    print()
    
    baseline_coverage = sum(strategies_stats["不改写(1)"]["coverages"]) / len(strategies_stats["不改写(1)"]["coverages"])
    rewrite2_coverage = sum(strategies_stats["少改写(2)"]["coverages"]) / len(strategies_stats["少改写(2)"]["coverages"])
    rewrite4_coverage = sum(strategies_stats["多改写(4)"]["coverages"]) / len(strategies_stats["多改写(4)"]["coverages"])
    
    baseline_time = sum(strategies_stats["不改写(1)"]["times"]) / len(strategies_stats["不改写(1)"]["times"])
    rewrite2_time = sum(strategies_stats["少改写(2)"]["times"]) / len(strategies_stats["少改写(2)"]["times"])
    rewrite4_time = sum(strategies_stats["多改写(4)"]["times"]) / len(strategies_stats["多改写(4)"]["times"])
    
    coverage_gain_2 = (rewrite2_coverage - baseline_coverage) * 100
    coverage_gain_4 = (rewrite4_coverage - baseline_coverage) * 100
    
    time_cost_2 = rewrite2_time - baseline_time
    time_cost_4 = rewrite4_time - baseline_time
    
    print(f"覆盖率对比:")
    print(f"  不改写 → 2改写: {coverage_gain_2:+.1f}个百分点, 耗时+{time_cost_2:.0f}ms")
    print(f"  不改写 → 4改写: {coverage_gain_4:+.1f}个百分点, 耗时+{time_cost_4:.0f}ms")
    print()
    
    if coverage_gain_4 < 5:
        print("❌ 4改写覆盖率提升<5%，ROI极低，建议关闭改写")
    elif coverage_gain_4 < 10:
        if coverage_gain_2 >= coverage_gain_4 * 0.7:
            print("⚠️  2改写能达到4改写70%+的效果，且快很多，建议降到2改写")
        else:
            print("⚠️  改写有一定价值，但ROI不高，建议降到2改写")
    else:
        print("✅ 改写显著提升召回质量，建议保留")
        if coverage_gain_2 >= coverage_gain_4 * 0.8:
            print("   但2改写已能达到4改写80%+的效果，可考虑降到2")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(compare_rewrite_strategies())
