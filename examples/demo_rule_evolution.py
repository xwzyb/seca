"""
Demo: Rule self-evolution
==========================

Runs 5 sequential tasks that progressively expose gaps and weaknesses in
the initial rule set. Observes how rules evolve over time.

After all tasks, prints the full evolution log and the final rule set.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from seca.agent import SECAAgent

console = Console()


TASKS = [
    # Task 1: Standard — should match existing rules fine
    """请分析以下数据并给出结论：
2023年Q1销售额：500万
2023年Q2销售额：600万
2023年Q3销售额：450万
2023年Q4销售额：800万
总结年度销售趋势。""",

    # Task 2: Emotional/ethical — no existing rule for this
    """一个AI系统在医疗诊断中给出了一个技术上正确但会引起患者极大焦虑的结果表述。
技术准确的表述："你有78%的概率患有X疾病。"
替代表述："检测显示有一些需要进一步确认的指标，我们建议做更详细的检查。"
哪个更合适？为什么？请从准确性和人文关怀两个维度分析。""",

    # Task 3: Meta-reasoning — requires thinking about thinking
    """请评估以下推理过程是否合理：

前提1：所有成功的创业公司都有好的产品。
前提2：公司X有好的产品。
结论：公司X是成功的创业公司。

这个推理有什么问题？请识别逻辑谬误并提供正确的推理方式。""",

    # Task 4: Creative synthesis — combining concepts
    """请将以下两个看似不相关的领域进行创造性融合：

领域A：中国传统园林设计原则（借景、曲径通幽、虚实相生）
领域B：现代用户界面(UI)设计

如何将园林设计的理念应用到UI设计中？给出至少3个具体的设计建议。""",

    # Task 5: Self-referential — deliberately triggers meta-cognition
    """请反思你在处理前面几个任务时的表现：
1. 你使用了什么策略？
2. 哪些策略有效，哪些无效？
3. 如果重新处理这些任务，你会做什么不同的事情？
4. 你认为自己需要什么新的能力或规则？

（注意：这是一个关于你自身认知过程的元认知任务。）""",
]


async def main() -> None:
    # Use a fresh data directory for this demo
    data_dir = Path(__file__).parent.parent / "data"
    
    # Clean previous demo rules to start fresh
    rules_file = data_dir / "rules" / "cognitive_rules.json"
    evolution_log = data_dir / "rules" / "evolution_log.json"
    meta_rules_file = data_dir / "rules" / "meta_rules.json"
    for f in [rules_file, evolution_log, meta_rules_file]:
        if f.exists():
            f.unlink()

    agent = SECAAgent(verbose=True, data_dir=str(data_dir))

    console.print(Panel(
        "[bold]This demo runs 5 tasks that progressively challenge the agent's rules.\n"
        "Watch how the rule set evolves after each task.[/bold]",
        title="🧬 Rule Evolution Demo",
        border_style="magenta",
    ))

    initial_rules = agent.rule_engine.all_rules()
    console.print(f"\n[bold]Initial rules: {len(initial_rules)}[/bold]")
    for r in initial_rules:
        console.print(f"  • {r.name} (confidence: {r.confidence:.0%})")

    # Run tasks sequentially
    results = []
    for i, task in enumerate(TASKS, 1):
        console.print(f"\n\n{'='*60}")
        console.print(f"[bold cyan]TASK {i}/{len(TASKS)}[/bold cyan]")
        console.print(f"{'='*60}")

        result = await agent.run(task)
        results.append(result)

        # Show rule state after each task
        current_rules = agent.rule_engine.all_rules()
        console.print(f"\n[dim]Rules after task {i}: {len(current_rules)} active[/dim]")

    # Final summary
    console.print(f"\n\n{'='*60}")
    console.print("[bold magenta]EVOLUTION SUMMARY[/bold magenta]")
    console.print(f"{'='*60}")

    # Final rule set
    final_rules = agent.rule_engine.all_rules(include_deprecated=True)
    table = Table(title="Final Rule Set")
    table.add_column("ID", style="dim", max_width=15)
    table.add_column("Name", max_width=25)
    table.add_column("Created By", max_width=10)
    table.add_column("Confidence", max_width=10)
    table.add_column("Usage", max_width=8)
    table.add_column("Success%", max_width=10)
    table.add_column("Status", max_width=12)

    for r in final_rules:
        status = "[red]deprecated[/red]" if r.deprecated else "[green]active[/green]"
        sr = f"{r.success_rate:.0%}" if r.usage_count > 0 else "N/A"
        table.add_row(
            r.id, r.name, r.created_by, f"{r.confidence:.0%}",
            str(r.usage_count), sr, status,
        )
    console.print(table)

    # Evolution log
    log = agent.rule_evolver.evolution_log
    if log:
        console.print(f"\n[bold]Evolution Log ({len(log)} entries):[/bold]")
        for entry in log:
            console.print(
                f"  [{entry.timestamp[:19]}] "
                f"[bold]{entry.action.upper()}[/bold] → {entry.reason[:80]}"
            )

    # Meta-rules
    meta_rules = agent.rule_evolver.meta_rules
    console.print(f"\n[bold]Final Meta-Rules ({len(meta_rules)}):[/bold]")
    for mr in meta_rules:
        console.print(f"  • {mr.name}: {mr.condition[:60]}... (threshold: {mr.threshold})")


if __name__ == "__main__":
    asyncio.run(main())
