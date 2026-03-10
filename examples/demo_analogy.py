"""
Demo: Analogical reasoning and strategy transfer
==================================================

Runs two tasks in sequence:
1. A resource allocation problem — the agent solves it and stores the experience
2. A structurally similar but surface-different scheduling problem

Observes how the agent finds the analogy and transfers the strategy.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seca.agent import SECAAgent


TASK_1 = """资源分配问题：

一家公司有3个部门（A、B、C），总预算1000万。
- 部门A需要至少400万来维持运营
- 部门B的项目回报率最高（预期ROI 200%），但需要至少300万启动
- 部门C正在进行关键的基础设施升级，需要200万
- 剩余预算可以自由分配

约束：任何部门分配不能超过总预算的50%。

请给出最优的预算分配方案。"""


TASK_2 = """时间安排问题：

一个学生有一周（7天）的考试复习时间。需要复习3门课程。
- 数学必须至少花3天才能通过
- 物理的分数权重最高（占总成绩40%），但至少需要2天系统复习
- 英语有一个基础作业必须完成，需要1天
- 剩余时间可以自由分配

约束：每门课的复习时间不能超过总时间的50%。

请给出最优的时间分配方案。"""


async def main() -> None:
    agent = SECAAgent(verbose=True, data_dir=str(Path(__file__).parent.parent / "data"))

    print("=" * 60)
    print("SECA Demo: Analogical Reasoning")
    print("=" * 60)

    # Task 1: Resource allocation
    print("\n🔵 TASK 1: Resource Allocation")
    print("-" * 40)
    result1 = await agent.run(TASK_1)
    print(f"\nAnswer: {result1.answer[:300]}")
    print(f"Episode stored: {result1.episode_stored}")

    print("\n\n")

    # Task 2: Time scheduling (structurally similar)
    print("🟢 TASK 2: Time Scheduling (structurally similar)")
    print("-" * 40)
    result2 = await agent.run(TASK_2)
    print(f"\nAnswer: {result2.answer[:300]}")
    print(f"Episodes referenced: {result2.episodes_referenced}")
    print(f"Rules used: {result2.rules_used}")

    # Show if analogy was found
    if result2.reasoning_trace and result2.reasoning_trace.mode.value == "analogy":
        print("\n✅ Agent used ANALOGY mode — it recognized the structural similarity!")
    else:
        print(f"\nℹ️  Agent used {result2.reasoning_trace.mode.value if result2.reasoning_trace else 'unknown'} mode")
        if result2.episodes_referenced:
            print("   But it DID reference past episodes, so some transfer may have occurred.")


if __name__ == "__main__":
    asyncio.run(main())
