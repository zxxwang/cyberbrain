"""
测试结果可视化分析

生成准确率热力图和问题分布图
"""


def print_heatmap(group_stats):
    """打印准确率热力图"""
    print("\n准确率热力图:")
    print("=" * 70)

    for group in group_stats:
        name = group["group"][:40]
        accuracy = group["accuracy"]
        total = group["total"]
        correct = group["correct"]

        # 绘制条形图
        bar_length = int(accuracy / 2)  # 50格 = 100%
        bar = "█" * bar_length + "░" * (50 - bar_length)

        # 颜色标记
        if accuracy >= 90:
            mark = "🟢"
        elif accuracy >= 70:
            mark = "🟡"
        else:
            mark = "🔴"

        print(f"{mark} {name:40s} [{bar}] {accuracy:5.1f}% ({correct}/{total})")


def print_importance_distribution(all_failed):
    """打印失败案例的 importance 分布"""
    from collections import defaultdict

    print("\n失败案例的 importance 分布:")
    print("=" * 70)

    # 按预期分组
    by_expected = defaultdict(list)
    for case in all_failed:
        by_expected[case["expected"]].append(case)

    for exp in sorted(by_expected.keys()):
        cases = by_expected[exp]
        print(f"\n预期 importance = {exp} ({len(cases)}个失败):")

        # 按实际分组统计
        by_actual = defaultdict(int)
        for case in cases:
            by_actual[case["actual"]] += 1

        for act in sorted(by_actual.keys()):
            count = by_actual[act]
            diff = act - exp
            arrow = "→" if diff == 0 else ("↑" if diff > 0 else "↓")
            print(f"  实际 {act}: {count}个 {arrow}{abs(diff)}")


def print_raw_score_analysis(all_failed):
    """分析原始分数分布"""
    print("\n原始分数 (raw) 分布分析:")
    print("=" * 70)

    # 按差距分组
    boundary_cases = []  # raw 接近边界的
    clear_errors = []  # raw 明显错误的

    for case in all_failed:
        expected = case["expected"]
        actual = case["actual"]
        raw = case["raw"]

        # 判断是否接近边界
        # 边界: 0.5, 1.5, 2.5, 3.5
        boundaries = [0.5, 1.5, 2.5, 3.5]
        near_boundary = any(abs(raw - b) < 0.3 for b in boundaries)

        if near_boundary:
            boundary_cases.append(case)
        else:
            clear_errors.append(case)

    print(f"接近边界的失败 ({len(boundary_cases)}个):")
    for case in boundary_cases[:5]:
        print(f"  - {case['text'][:40]:40s} raw={case['raw']:.2f} (预期{case['expected']}, 实际{case['actual']})")

    print(f"\n明显错误的失败 ({len(clear_errors)}个):")
    for case in clear_errors[:5]:
        print(f"  - {case['text'][:40]:40s} raw={case['raw']:.2f} (预期{case['expected']}, 实际{case['actual']})")


if __name__ == "__main__":
    # 这个脚本会被其他脚本调用
    pass
