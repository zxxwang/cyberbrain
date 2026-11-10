"""
Prompt 问题诊断工具

基于测试结果快速分析 Prompt 的薄弱点
"""
import json
from collections import defaultdict
from typing import List, Dict


def analyze_failures(failed_cases: List[Dict]) -> Dict:
    """分析失败模式"""
    analysis = {
        "over_estimation": [],  # 评分过高
        "under_estimation": [],  # 评分过低
        "boundary_cases": [],  # 边界情况（差1分）
        "severe_cases": [],  # 严重偏差（差2分以上）
    }

    for case in failed_cases:
        diff = case["actual"] - case["expected"]

        # 分类
        if diff > 0:
            analysis["over_estimation"].append(case)
        else:
            analysis["under_estimation"].append(case)

        if abs(diff) == 1:
            analysis["boundary_cases"].append(case)
        elif abs(diff) >= 2:
            analysis["severe_cases"].append(case)

    return analysis


def suggest_prompt_fixes(analysis: Dict) -> List[str]:
    """基于失败模式建议 Prompt 修复"""
    suggestions = []

    # 过度评分
    if analysis["over_estimation"]:
        over_texts = [c["text"] for c in analysis["over_estimation"]]
        suggestions.append("🔴 过度评分问题:")
        suggestions.append(f"   涉及 {len(over_texts)} 个案例")
        suggestions.append(f"   示例: {over_texts[:3]}")

        # 检查是否有提问被高估
        questions = [t for t in over_texts if "？" in t or "怎么" in t or "什么" in t]
        if questions:
            suggestions.append("   ⚠️  提问被高估 -> 在低价值中强调'纯提问'")

        # 检查是否有情感被高估
        emotions = [t for t in over_texts if any(w in t for w in ["心情", "累", "开心", "压力"])]
        if emotions:
            suggestions.append("   ⚠️  情感表达被高估 -> 在低价值中加入'情感状态'")

        # 检查是否有临时任务被高估
        tasks = [t for t in over_texts if any(w in t for w in ["明天", "今天", "这周"])]
        if tasks:
            suggestions.append("   ⚠️  近期任务被高估 -> 在中等价值中强调'临时'和'近期'")

    # 低估
    if analysis["under_estimation"]:
        under_texts = [c["text"] for c in analysis["under_estimation"]]
        suggestions.append("\n🔵 低估问题:")
        suggestions.append(f"   涉及 {len(under_texts)} 个案例")
        suggestions.append(f"   示例: {under_texts[:3]}")

        # 检查是否有否定句被低估
        negations = [t for t in under_texts if "不" in t]
        if negations:
            suggestions.append("   ⚠️  否定句被低估 -> 在高价值中明确'不喜欢、不擅长也是重要偏好'")

        # 检查是否有事件被低估
        events = [t for t in under_texts if any(w in t for w in ["昨天", "完成", "去了"])]
        if events:
            suggestions.append("   ⚠️  已完成事件被低估 -> 在高价值中强调'已经做完的工作'")

        # 检查是否有技能被低估
        skills = [t for t in under_texts if any(w in t for w in ["更", "可以", "适合", "最佳"])]
        if skills:
            suggestions.append("   ⚠️  技能知识被低估 -> 在高价值中加入更多技能关键词")

    # 边界情况
    if analysis["boundary_cases"]:
        suggestions.append(f"\n🟡 边界情况 ({len(analysis['boundary_cases'])}个):")
        suggestions.append("   这些案例差1分，可能是合理的分歧")
        for case in analysis["boundary_cases"][:3]:
            suggestions.append(f"   - {case['text'][:40]} (预期{case['expected']} 实际{case['actual']})")

    # 严重偏差
    if analysis["severe_cases"]:
        suggestions.append(f"\n🔴 严重偏差 ({len(analysis['severe_cases'])}个):")
        suggestions.append("   这些案例差2分以上，需要重点修复")
        for case in analysis["severe_cases"]:
            suggestions.append(f"   - {case['text'][:40]} (预期{case['expected']} 实际{case['actual']})")

    return suggestions


def main():
    print("Prompt 诊断工具\n")
    print("等待测试结果...\n")

    # 假设测试结果会写入JSON
    import time
    import os

    result_file = "gate_extended_test_results.json"

    # 等待结果文件
    for i in range(60):
        if os.path.exists(result_file):
            break
        time.sleep(1)
    else:
        print("超时：未找到测试结果文件")
        return

    # 读取结果
    with open(result_file, "r") as f:
        results = json.load(f)

    # 收集所有失败案例
    all_failed = []
    for group in results:
        if "failed" in group and group["failed"]:
            for case in group["failed"]:
                if "error" not in case:
                    all_failed.append({
                        "group": group["group"],
                        "text": case["text"],
                        "expected": case.get("expected_imp", group.get("expected", 0)),
                        "actual": case["actual"],
                        "raw": case["raw"],
                    })

    print(f"收集到 {len(all_failed)} 个失败案例\n")

    if not all_failed:
        print("🎉 没有失败案例！Prompt 效果极佳！")
        return

    # 分析
    analysis = analyze_failures(all_failed)

    print("="*80)
    print("失败模式分析")
    print("="*80)
    print(f"过度评分: {len(analysis['over_estimation'])} 个")
    print(f"低估评分: {len(analysis['under_estimation'])} 个")
    print(f"边界情况: {len(analysis['boundary_cases'])} 个")
    print(f"严重偏差: {len(analysis['severe_cases'])} 个\n")

    # 建议
    suggestions = suggest_prompt_fixes(analysis)

    print("="*80)
    print("Prompt 修复建议")
    print("="*80)
    for s in suggestions:
        print(s)


if __name__ == "__main__":
    main()
