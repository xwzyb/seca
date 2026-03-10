# SECA — Self-Evolving Cognitive Agent

> 一个能自我修改规则的认知智能体实验框架

## 核心理念

当前AI Agent框架（ReAct、AutoGPT、CrewAI等）的共同问题：**规则是人写死的**。SECA的核心创新是让智能体能够**观察自己的推理过程、评估策略效果、并自动修改和创建规则**——实现Hofstadter所说的"奇异循环"(Strange Loop)。

## 架构

```
┌─────────────────────────────────────────────────┐
│         Meta-Cognitive Layer (元认知层)           │
│   MetaCognitiveMonitor → RuleEvolver            │
│   监控推理 → 评估策略 → 修改规则 → 修改元规则    │
│                  🌀 奇异循环                     │
├─────────────────────────────────────────────────┤
│         Cognitive Layer (认知层)                  │
│   RuleEngine → ReasoningEngine → AnalogyEngine  │
│   规则匹配 → 多模式推理 → 跨情境类比             │
├─────────────────────────────────────────────────┤
│         Foundation Layer (基础层)                 │
│   LLM (OpenAI/Anthropic) + Tools + Memory       │
│   工作记忆 / 经验记忆 / 语义记忆                  │
└─────────────────────────────────────────────────┘
```

## 文件结构

```
seca/
├── seca/
│   ├── __init__.py                      # 包入口
│   ├── agent.py                         # 🎯 主智能体 (SECAAgent)
│   ├── foundation/
│   │   ├── llm.py                       # LLM统一接口 (OpenAI/Anthropic)
│   │   ├── memory.py                    # 三层记忆系统
│   │   └── tools.py                     # 工具注册与调用
│   ├── cognitive/
│   │   ├── rules.py                     # 💡 认知规则系统 (CognitiveRule + RuleEngine)
│   │   ├── reasoning.py                 # 多模式推理引擎 (CoT/分解/类比)
│   │   └── analogy.py                   # 类比引擎 (跨情境策略迁移)
│   └── metacognitive/
│       ├── monitor.py                   # 🔍 元认知监控器
│       └── rule_evolver.py              # 🌀 规则演化器 (奇异循环核心)
├── examples/
│   ├── demo_contradiction.py            # 演示：矛盾信息处理
│   ├── demo_analogy.py                  # 演示：类比推理与策略迁移
│   └── demo_rule_evolution.py           # 演示：规则自我演化
├── data/
│   ├── memory/                          # 记忆持久化 (JSON)
│   └── rules/                           # 规则与演化日志 (JSON)
├── pyproject.toml
└── README.md
```

## 核心组件说明

### 认知规则 (`cognitive/rules.py`)
- 规则是**可读的JSON**，不是黑箱
- 每条规则包含：触发条件(condition)、策略(strategy)、置信度(confidence)、使用统计
- 预置5条初始规则（矛盾处理、任务澄清、复杂分解、自我验证、类比迁移）

### 推理引擎 (`cognitive/reasoning.py`)
- 三种推理模式：Chain-of-Thought / 任务分解 / 类比推理
- 自动选择最适合的推理模式
- 每步推理记录完整trace，供元认知层分析

### 类比引擎 (`cognitive/analogy.py`)
- 从经验记忆中检索结构相似的情境
- 用LLM进行深层结构映射（不只是表面相似）
- 将旧策略适配到新情境

### 元认知监控 (`metacognitive/monitor.py`)
- 检测推理循环、矛盾、策略失效
- 评估整体推理质量（置信度）
- 生成建议动作（修改规则/创建规则/废弃规则）

### 规则演化器 (`metacognitive/rule_evolver.py`) — 🌀 奇异循环
- `refine_rule()`: 根据反馈微调规则
- `create_rule()`: 从经验中创造新规则
- `merge_rules()`: 合并相似规则为更通用的规则
- `deprecate_rule()`: 废弃低效规则
- `evolve_meta_rules()`: **修改元规则本身** — 这就是奇异循环！

### 记忆系统 (`foundation/memory.py`)
- **WorkingMemory**: 当前任务的短期记忆
- **EpisodicMemory**: 经验记忆 {情境, 策略, 结果, 反思}
- **SemanticMemory**: 长期知识记忆

## 快速开始

### 1. 安装依赖

```bash
pip install pydantic rich boto3
```

### 2. 配置AWS凭证（Bedrock，默认方式）

```bash
# 方式一：环境变量
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_REGION="us-east-1"

# 方式二：AWS CLI配置（推荐）
aws configure

# 方式三：EC2/ECS实例自动使用IAM角色，无需配置
```

默认使用 `global.anthropic.claude-opus-4-6-v1`，可通过代码切换：

```python
from seca.foundation.llm import configure

# 切换模型
configure(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")

# 切换region
configure(region="us-west-2")

# 回退到直连Anthropic API
configure(provider="anthropic")
```

#### 备选方式：直连API

```bash
# Anthropic直连
pip install anthropic
export ANTHROPIC_API_KEY="your-key"

# OpenAI
pip install openai
export OPENAI_API_KEY="your-key"
```

Provider自动检测优先级：**Bedrock > Anthropic > OpenAI**

### 3. 运行示例

```bash
# 矛盾信息处理
python examples/demo_contradiction.py

# 类比推理
python examples/demo_analogy.py

# 规则自我演化（最精彩！）
python examples/demo_rule_evolution.py
```

## 智能原则映射

| 智能原则 | SECA实现 |
|---------|---------|
| 灵活反应 | RuleEngine动态匹配 + 多模式推理 |
| 处理矛盾 | Contradiction Resolution规则 + Monitor检测 |
| 识别重要/次要 | LLM驱动的规则匹配与置信度评估 |
| 跨情境类比 | AnalogyEngine结构映射 |
| 相似中找差异 | 类比引擎的差异识别 |
| 旧概念生新概念 | RuleEvolver.create_rule() |
| 提出新概念 | RuleEvolver.merge_rules() + evolve_meta_rules() |
| 改变规则的规则 | 🌀 RuleEvolver.evolve_meta_rules() |

## 设计哲学

1. **规则可读** — 所有规则存储为人类可读的JSON，随时可以检查
2. **演化有日志** — 每次规则变更都有before/after记录
3. **保守演化** — 元规则的自我修改是渐进的，不会一次性推翻所有规则
4. **经验驱动** — 规则的创建和修改基于实际经验，不是随机变异

## 下一步扩展方向

- [ ] 用embedding替换SequenceMatcher做记忆检索
- [ ] 添加真正的工具调用（web搜索、代码执行）
- [ ] 多Agent协作（多个SECA智能体交换规则）
- [ ] 可视化规则演化树
- [ ] 基准测试（ARC、Winograd等）
