"""
Demo: Handling contradictory information
=========================================

Gives the SECA agent a text with contradictions and observes how it:
1. Detects the contradictions
2. Applies the Contradiction Resolution rule
3. Produces a reasoned judgment
4. Potentially evolves its rules based on the experience
"""

import asyncio
import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seca.agent import SECAAgent


TASK = """请分析以下矛盾信息并确定最可能的真实情况：

报告A（来源：财务部门，2024年1月15日）：
"项目X的总预算为500万人民币，已支出200万，剩余300万。"

报告B（来源：项目经理，2024年3月20日）：
"项目X的总预算为800万人民币，已支出350万，剩余450万。"

报告C（来源：CEO邮件，2024年2月10日）：
"董事会批准了项目X 300万的追加预算。"

请确定项目X的真实预算情况。"""


async def main() -> None:
    agent = SECAAgent(verbose=True, data_dir=str(Path(__file__).parent.parent / "data"))

    print("=" * 60)
    print("SECA Demo: Contradiction Resolution")
    print("=" * 60)

    result = await agent.run(TASK)

    print("\n" + "=" * 60)
    print("RESULT SUMMARY")
    print("=" * 60)
    print(f"Answer: {result.answer[:500]}")
    print(f"Rules used: {result.rules_used}")
    print(f"Rules evolved: {result.rules_evolved}")
    print(f"Monitor confidence: {result.monitor_report.confidence_assessment:.0%}" if result.monitor_report else "N/A")
    print(f"Episode stored: {result.episode_stored}")


if __name__ == "__main__":
    asyncio.run(main())
