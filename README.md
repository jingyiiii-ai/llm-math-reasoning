# LLM Math Reasoning

基于 Qwen2.5-7B-Instruct 的数学推理微调项目，使用 LlamaFactory 实现 SFT + LoRA 微调，并在 GSM8K 与 SVAMP 数据集上进行评测与分析。

---

## 项目简介

本项目主要研究大语言模型在数学应用题推理任务中的表现。项目围绕 Chain-of-Thought（CoT）推理与代码辅助推理展开，对不同训练策略进行了对比实验。

项目完整实现了：

- 数学数据清洗与格式统一
- SFT 指令数据构造
- 基于 LoRA 的参数高效微调
- GSM8K / SVAMP 自动化评测
- 错误样本与推理行为分析

---

## 技术栈

- Python
- PyTorch
- Transformers
- PEFT
- LlamaFactory
- LoRA / SFT
- ModelScope

---

## 项目结构

```text
llm-math-reasoning/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── build_math_datasets.py
│   ├── clean_full_codeonly_v2.py
│   ├── ablated_check.py
│   ├── full_check.py
│   └── check_ablated_residue.py
├── results/
│   ├── clean_report.json
│   ├── ablated_residue_report.json
│   └── train_full_codeonly_v2.report.json
└── data/
    ├── sample_raw.json
    ├── sample_full_sft.jsonl
    └── sample_ablated_sft.jsonl
```

---

## 实验内容

项目基于 **Qwen2.5-7B-Instruct** 构建三类对照模型：

- **Base**：原始基座模型
- **Ablated**：仅使用 CoT 推理数据微调
- **Full v2**：加入代码辅助推理数据微调

并在 **GSM8K** 与 **SVAMP** 数据集上进行准确率评测与对比分析。

---

## 数据处理

训练数据来源于公开的 **Math-Solver 数学习题综合数据集**。

数据处理流程包括：

- 异常样本过滤
- 字段格式统一
- 最终答案抽取
- 推理过程标准化
- CoT 与代码辅助推理格式构造

清洗后的数据进一步转换为 **SFT 指令微调格式**，用于 **LoRA 微调训练**。

---

## 运行流程

### 1. 构建训练数据

```bash
python scripts/build_math_datasets.py
```

### 2. 清洗 Full v2 代码辅助数据

```bash
python scripts/clean_full_codeonly_v2.py
```

### 3. 检查 Ablated 数据残留

```bash
python scripts/check_ablated_residue.py
```

### 4. 检查训练数据格式

```bash
python scripts/full_check.py
python scripts/ablated_check.py
```

---

## 实验结果

实验结果表明，经过 **SFT + LoRA** 微调后，模型在数学推理任务上的准确率相比基座模型有明显提升。

同时，跨数据集评测结果显示，微调模型在 **SVAMP** 数据集上仍具备一定泛化能力。

实验分析发现：

- CoT 推理是模型性能提升的主要来源
- 代码辅助推理在部分计算型题目中具有补充作用
- 不同推理方式在不同题型上存在一定互补性



## 说明

由于完整训练数据与模型权重体积较大，本仓库仅提供：

- 核心训练与评测脚本
- 配置文件
- 少量样例数据
- 实验结果摘要

不上传完整训练数据集与模型权重。
