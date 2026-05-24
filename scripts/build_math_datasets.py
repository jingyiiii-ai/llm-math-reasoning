#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_math_datasets.py

强消融版数学训练数据清洗脚本
用于毕业设计《基于代码辅助推理的大模型数学解题能力优化研究》

功能：
1. 以 train_data.json 为主输入，统一压成 SFT 单轮样本：instruction / input / output
2. 支持 3 / 5 轮样本解析，过滤异常轮次
3. 生成 full / ablated 两套平行数据
4. 额外生成 codeonly / gsm8kstyle / mathstyle 子集
5. 对 ablated 使用“强消融”策略：
   - 删除 Python 代码块
   - 删除代码执行结果
   - 删除 Python 相关提示语
   - 删除代码辅助推理痕迹
   - 若仍残留代码痕迹，则整条样本丢弃
6. 自动去重、异常过滤、统计报告
7. 自动校验 ablated 是否仍残留代码痕迹

输出文件：
- train_full_clean.jsonl
- train_ablated_clean.jsonl
- train_full_codeonly.jsonl
- train_ablated_codeonly.jsonl
- train_full_gsm8kstyle.jsonl
- train_ablated_gsm8kstyle.jsonl
- train_full_mathstyle.jsonl
- train_ablated_mathstyle.jsonl
- clean_report.json

如强消融校验失败，还会输出：
- strong_ablation_residue_examples.json
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_SYSTEM = (
    "你是一个数学解题大师，请解决以下数学题，务必详细说明解题思路，"
    "并在必要时提供Python代码来支持你的推理。答案中的数值应使用\\boxed{}包围，"
    "最后的答案以“因此”开头并直接给出结论，不要添加任何多余的内容。"
)

ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
    "user": "user",
    "assistant": "assistant",
}

QUESTION_PREFIX_RE = re.compile(r"^\s*(题目[:：]\s*)", re.IGNORECASE)
MULTI_BLANK_RE = re.compile(r"\n\s*\n+")
MULTI_SPACE_RE = re.compile(r"[ \t]+")

# 删除代码块
FENCED_CODE_RE = re.compile(r"```(?:python)?\s*[\s\S]*?```", re.IGNORECASE)
ORPHAN_CODE_START_RE = re.compile(r"```(?:python)?[\s\S]*$", re.IGNORECASE)

# 提取 ```...``` 中内容（用于 full 中保留执行结果）
TRIPLE_BACKTICK_RE = re.compile(r"```([\s\S]*?)```")

ADVANCED_MATH_PATTERNS = [
    r"曲线积分", r"微分方程", r"二阶线性", r"特征方程", r"定义域", r"函数\s*f\(x\)",
    r"\by''\b", r"\bdx\b", r"\bdy\b", r"\\oint", r"\\int", r"\\frac", r"\\sqrt",
    r"sin", r"cos", r"tan", r"cot", r"sec", r"csc", r"极限", r"导数", r"积分",
    r"正三角形", r"圆\(", r"弧", r"pi", r"\\pi", r"lg\(", r"log\(", r"证明",
    r"多边形", r"内角", r"外角", r"三角函数", r"集合", r"方程为", r"已知曲线",
]

GSM8K_STYLE_HINTS = [
    "how many", "how much", "total", "average", "more than", "less than",
    "每", "一共", "总共", "多少", "剩下", "平均", "比", "多", "少", "用了",
    "买了", "跑", "箱子", "盒子", "苹果", "水果", "厘米", "千克", "毫升", "美元",
    "英寸", "速度", "时间", "比例", "人数", "台阶", "面积", "体积",
]

# 强消融：代码痕迹关键词（发现就删）
CODE_TRACE_KEYWORDS = [
    "python代码",
    "python 验证",
    "python验证",
    "根据python代码",
    "代码执行结果",
    "运行上述代码",
    "执行上述python代码",
    "执行上述代码",
    "复制到python环境中运行",
    "现在将上面的代码复制到python环境中运行",
    "上面的python代码执行结果为",
    "生成的python代码如下",
    "我们可以使用python代码来验证",
    "我们可以使用python代码来验证这个计算过程",
    "我们可以使用python代码来验证这个解",
    "下面给出python代码",
    "python code",
]

# 强消融：检测残留真实代码/代码痕迹
CODE_RESIDUE_PATTERNS = [
    re.compile(r"```python", re.I),
    re.compile(r"```"),
    re.compile(r"\bdef\s+\w+\s*\("),
    re.compile(r"\bprint\s*\("),
    re.compile(r"\bimport\s+\w+"),
    re.compile(r"\bfor\s+\w+\s+in\s+"),
    re.compile(r"\bwhile\s+"),
    re.compile(r"\breturn\b"),
    re.compile(r"\bif\s+.*:"),
    re.compile(r"\bclass\s+\w+"),
]

# 行级删除：这些行即使没有代码块，也视为代码痕迹
NOISE_LINE_HINTS = [
    "python代码",
    "python 验证",
    "python验证",
    "根据python代码",
    "根据您提供的python代码",
    "代码执行结果",
    "运行上述代码",
    "执行上述代码",
    "执行上述python代码",
    "复制到python环境中运行",
    "上面的python代码执行结果为",
    "输出结果为",
    "输出会是",
    "生成的python代码如下",
    "我们可以使用python代码来验证",
    "我们可以使用来验证这个计算过程",
    "我们可以使用来验证这个解",
    "下面给出python代码",
    "python code",
]


@dataclass
class CleanSample:
    instruction: str
    input: str
    full_output: str
    ablated_output: str
    style: str
    source: str
    turns: int
    has_code: bool
    quality_score: int


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ")
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def ensure_question_prefix(question: str) -> str:
    question = normalize_text(question)
    if not question.startswith("题目："):
        question = f"题目：{question}"
    return question


def normalize_question_for_key(text: str) -> str:
    text = normalize_text(text)
    text = QUESTION_PREFIX_RE.sub("", text)
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。、“”‘’!！?？:：;；,.\-—_（）()\[\]{}]", "", text)
    return text


def classify_style(question: str) -> str:
    q = question.lower()

    for pat in ADVANCED_MATH_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            return "mathstyle"

    hit_count = 0
    for hint in GSM8K_STYLE_HINTS:
        if hint.lower() in q:
            hit_count += 1

    if hit_count >= 2:
        return "gsm8kstyle"

    if any(unit in q for unit in ["千克", "厘米", "毫升", "美元", "英寸", "小时", "分钟", "公斤", "kg", "cm", "ml"]):
        return "gsm8kstyle"

    return "general"


def compute_quality_score(
    question: str,
    full_output: str,
    ablated_output: str,
    turns: int,
    has_code: bool,
    had_system: bool,
) -> int:
    score = 0

    if had_system:
        score += 2
    if turns in (3, 5):
        score += 2
    if has_code:
        score += 1
    if "\\boxed" in full_output:
        score += 2
    if len(question) >= 8:
        score += 1
    if len(full_output) >= 80:
        score += 2
    if len(ablated_output) >= 50:
        score += 1
    if len(full_output) > 5000:
        score -= 2

    return score


def extract_execution_result_from_user_turn(text: str) -> str:
    """
    从 5 轮样本中的 user 执行反馈里提取执行结果，供 full 版本保留。
    """
    text = normalize_text(text)
    if not text:
        return ""

    blocks = TRIPLE_BACKTICK_RE.findall(text)
    if blocks:
        result = "\n".join(x.strip() for x in blocks if x.strip())
    else:
        result = text

    # 清掉前缀提示
    prefixes = [
        "现在将上面的代码复制到Python环境中运行，运行结果为：",
        "现在将上面的代码复制到Python环境中运行，运行结果将是：",
        "运行上述代码，我们可以得到题目要求的答案。输出结果将是：",
        "运行上述代码，输出会是：",
        "执行上述Python代码，运行结果为：",
        "上面的Python代码执行结果为：",
        "运行以上代码，输出会是：",
        "运行上述代码，我们可以得到题目要求的答案。输出结果为：",
    ]

    for p in prefixes:
        result = result.replace(p, "")

    result = result.strip("` \n")
    return normalize_text(result)


def remove_code_blocks_aggressive(text: str) -> str:
    """
    激进删除所有 fenced 代码块和孤立代码块开头。
    """
    text = normalize_text(text)
    if not text:
        return ""

    text = FENCED_CODE_RE.sub("", text)
    text = ORPHAN_CODE_START_RE.sub("", text)
    return normalize_text(text)


def strong_remove_code_and_noise(text: str) -> str:
    """
    强消融：
    1. 删除 Python 代码块
    2. 删除代码执行结果提示
    3. 删除 Python 相关说明行
    4. 删除残留反引号代码块
    """
    text = normalize_text(text)
    if not text:
        return ""

    # 第一步：先删完整/孤立代码块
    text = remove_code_blocks_aggressive(text)

    # 第二步：删掉常见代码提示短语
    replace_patterns = [
        r"###\s*Python代码验证[:：]?",
        r"###\s*Python\s*验证代码[:：]?",
        r"###\s*Python代码[:：]?",
        r"###\s*Python\s*验证[:：]?",
        r"Python代码验证[:：]?",
        r"Python\s*验证代码[:：]?",
        r"Python\s*验证[:：]?",
        r"Python代码[:：]?",
        r"生成的Python代码如下[:：]?",
        r"下面给出Python代码[:：]?",
        r"我们可以使用Python代码来验证这个计算过程[:：]?",
        r"我们可以使用Python代码来验证这个解[:：]?",
        r"我们可以使用Python代码来验证[:：]?",
        r"根据Python代码的验证结果[，,]?",
        r"根据计算和Python代码的验证[，,]?",
        r"根据您提供的Python代码执行结果[，,]?",
        r"根据您在Python环境中运行的结果[，,]?",
        r"现在将上面的代码复制到Python环境中运行[，,]?[运行结果为将是:：]*",
        r"运行上述代码[，,]?(我们可以得到题目要求的答案。)?[输出结果为将是:：]*",
        r"执行上述Python代码[，,]?[运行结果为将是:：]*",
        r"上面的Python代码执行结果为[：:]?",
        r"运行以上代码[，,]?[输出会是为将是:：]*",
    ]

    for pat in replace_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # 第三步：按行过滤明显代码痕迹
    kept_lines = []
    for line in text.split("\n"):
        line_strip = line.strip()
        lower_line = line_strip.lower()

        if not line_strip:
            kept_lines.append("")
            continue

        # 含明显代码痕迹的整行直接删
        if any(hint in lower_line for hint in NOISE_LINE_HINTS):
            continue

        # 剩余的代码痕迹行也删
        if "```" in line_strip:
            continue

        # 过于像代码的行删掉
        if re.search(r"\bdef\s+\w+\s*\(", line_strip):
            continue
        if re.search(r"\bprint\s*\(", line_strip):
            continue
        if re.search(r"\bimport\s+\w+", line_strip):
            continue
        if re.search(r"\breturn\b", line_strip):
            continue
        if re.search(r"\bfor\s+\w+\s+in\s+", line_strip):
            continue
        if re.search(r"\bwhile\s+", line_strip):
            continue
        if re.search(r"\bif\s+.*:", line_strip):
            continue
        if re.search(r"^\s*#", line_strip):
            continue

        kept_lines.append(line)

    text = "\n".join(kept_lines)

    # 第四步：清掉多余空白
    text = normalize_text(text)

    # 第五步：把一些清洗后留下的“我们可以使用来验证……”这类残句删掉
    cleanup_lines = []
    for line in text.split("\n"):
        s = line.strip()
        lower_s = s.lower()

        if not s:
            cleanup_lines.append("")
            continue

        bad_fragments = [
            "我们可以使用来验证",
            "根据您提供的",
            "根据计算和的验证",
            "代码验证结果",
            "python环境中运行",
        ]
        if any(x in lower_s for x in bad_fragments):
            continue

        cleanup_lines.append(line)

    text = "\n".join(cleanup_lines)
    text = normalize_text(text)

    return text


def detect_code_residue(text: str) -> List[str]:
    """
    强消融后的最终校验：
    如果仍命中这些规则，说明样本不够干净。
    """
    hits = []
    lower_text = text.lower()

    for p in CODE_RESIDUE_PATTERNS:
        if p.search(text):
            hits.append(p.pattern)

    for kw in CODE_TRACE_KEYWORDS:
        if kw in lower_text:
            hits.append(f"KW:{kw}")

    return hits


def parse_conversations(item: Dict[str, Any], source_name: str) -> Tuple[Optional[CleanSample], Optional[str]]:
    conversations = item.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return None, "missing_or_empty_conversations"

    msgs = []
    for conv in conversations:
        role = ROLE_MAP.get(conv.get("from", ""), conv.get("from", ""))
        content = normalize_text(conv.get("value", ""))
        if role and content:
            msgs.append({"role": role, "content": content})

    if len(msgs) < 2:
        return None, "too_few_valid_messages"

    had_system = msgs[0]["role"] == "system"
    turns = len(msgs)

    # 强约束：只要 3 轮或 5 轮
    if turns not in (3, 5):
        return None, f"unsupported_turns_{turns}"

    system_msg = msgs[0]["content"] if had_system else DEFAULT_SYSTEM
    start_idx = 1 if had_system else 0

    user_idx = None
    for i in range(start_idx, len(msgs)):
        if msgs[i]["role"] == "user":
            user_idx = i
            break

    if user_idx is None:
        return None, "missing_user_question"

    question = ensure_question_prefix(msgs[user_idx]["content"])
    tail = msgs[user_idx + 1:]

    if not tail:
        return None, "missing_response_after_question"

    assistant_msgs = [m["content"] for m in tail if m["role"] == "assistant"]
    user_msgs_after_question = [m["content"] for m in tail if m["role"] == "user"]

    if not assistant_msgs:
        return None, "missing_assistant_answer"

    first_assistant = assistant_msgs[0]
    last_assistant = assistant_msgs[-1]

    has_code = bool(re.search(r"```python", first_assistant, flags=re.I)) or bool(FENCED_CODE_RE.search(first_assistant))

    # ===== full 版本 =====
    if len(assistant_msgs) == 1:
        full_output = first_assistant
    else:
        exec_results = []
        for msg in user_msgs_after_question:
            result = extract_execution_result_from_user_turn(msg)
            if result:
                exec_results.append(result)

        full_parts = [first_assistant]
        if exec_results:
            full_parts.append("代码执行结果：\n" + "\n".join(exec_results))
        if last_assistant != first_assistant:
            full_parts.append(last_assistant)

        full_output = "\n\n".join(x.strip() for x in full_parts if x and x.strip())

    full_output = normalize_text(full_output)

    # ===== 强消融 ablated 版本 =====
    if len(assistant_msgs) == 1:
        ablated_output = strong_remove_code_and_noise(first_assistant)
    else:
        ablated_parts = [strong_remove_code_and_noise(first_assistant)]
        if last_assistant != first_assistant:
            ablated_parts.append(strong_remove_code_and_noise(last_assistant))
        ablated_output = "\n\n".join(x.strip() for x in ablated_parts if x and x.strip())

    ablated_output = normalize_text(ablated_output)

    # 基础长度检查
    if len(full_output) < 30:
        return None, "full_output_too_short"
    if len(ablated_output) < 20:
        return None, "ablated_output_too_short"

    # 强消融检查：若还有代码痕迹，整条丢弃
    residue_hits = detect_code_residue(ablated_output)
    if residue_hits:
        return None, "ablated_code_residue"

    style = classify_style(question)
    quality_score = compute_quality_score(
        question=question,
        full_output=full_output,
        ablated_output=ablated_output,
        turns=turns,
        has_code=has_code,
        had_system=had_system,
    )

    sample = CleanSample(
        instruction=system_msg,
        input=question,
        full_output=full_output,
        ablated_output=ablated_output,
        style=style,
        source=source_name,
        turns=turns,
        has_code=has_code,
        quality_score=quality_score,
    )
    return sample, None


def deduplicate_samples(samples: List[CleanSample]) -> Tuple[List[CleanSample], int]:
    best_by_question: Dict[str, CleanSample] = {}
    removed = 0

    for sample in samples:
        key = normalize_question_for_key(sample.input)

        if key not in best_by_question:
            best_by_question[key] = sample
            continue

        old = best_by_question[key]
        if sample.quality_score > old.quality_score:
            best_by_question[key] = sample
            removed += 1
        else:
            removed += 1

    return list(best_by_question.values()), removed


def convert_for_output(samples: List[CleanSample], mode: str) -> List[Dict[str, str]]:
    rows = []
    for s in samples:
        output_text = s.full_output if mode == "full" else s.ablated_output
        rows.append({
            "instruction": s.instruction,
            "input": s.input,
            "output": output_text,
        })
    return rows


def build_subsets(samples: List[CleanSample]) -> Dict[str, List[CleanSample]]:
    return {
        "all": samples,
        "codeonly": [s for s in samples if s.has_code],
        "gsm8kstyle": [s for s in samples if s.style == "gsm8kstyle"],
        "mathstyle": [s for s in samples if s.style == "mathstyle"],
    }


def write_jsonl(records: List[Dict[str, str]], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_report(
    path: Path,
    raw_count: int,
    valid_count_before_dedup: int,
    valid_count_after_dedup: int,
    duplicate_removed: int,
    skipped_reasons: Dict[str, int],
    samples: List[CleanSample],
):
    style_count: Dict[str, int] = {}
    turns_count: Dict[str, int] = {}
    code_count = 0

    for s in samples:
        style_count[s.style] = style_count.get(s.style, 0) + 1
        turns_count[str(s.turns)] = turns_count.get(str(s.turns), 0) + 1
        if s.has_code:
            code_count += 1

    report = {
        "raw_count": raw_count,
        "valid_count_before_dedup": valid_count_before_dedup,
        "valid_count_after_dedup": valid_count_after_dedup,
        "duplicate_removed": duplicate_removed,
        "skipped_reasons": skipped_reasons,
        "style_count_after_dedup": style_count,
        "turns_count_after_dedup": turns_count,
        "samples_with_code_after_dedup": code_count,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def validate_strong_ablated(rows: List[Dict[str, str]], save_path: Path):
    """
    对最终 ablated 数据做强校验。
    若仍残留代码痕迹，输出样例并报错。
    """
    bad = []
    for idx, row in enumerate(rows):
        hits = detect_code_residue(row["output"])
        if hits:
            bad.append({
                "index": idx,
                "hits": hits,
                "preview": row["output"][:1200]
            })

    if bad:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(bad[:50], f, ensure_ascii=False, indent=2)
        raise ValueError(
            f"强消融校验失败：仍有 {len(bad)} 条样本残留代码痕迹。"
            f" 已保存前50条到 {save_path}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="train_data.json", help="主输入文件，默认 train_data.json")
    parser.add_argument("--out_dir", type=str, default=".", help="输出目录")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError("输入文件必须是 JSON 数组")

    valid_samples: List[CleanSample] = []
    skipped_reasons: Dict[str, int] = {}

    for item in raw_data:
        sample, reason = parse_conversations(item, source_name=input_path.name)
        if sample is not None:
            valid_samples.append(sample)
        else:
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

    before_dedup = len(valid_samples)
    deduped_samples, duplicate_removed = deduplicate_samples(valid_samples)
    subsets = build_subsets(deduped_samples)

    outputs = {
        "train_full_clean.jsonl": convert_for_output(subsets["all"], "full"),
        "train_ablated_clean.jsonl": convert_for_output(subsets["all"], "ablated"),

        "train_full_codeonly.jsonl": convert_for_output(subsets["codeonly"], "full"),
        "train_ablated_codeonly.jsonl": convert_for_output(subsets["codeonly"], "ablated"),

        "train_full_gsm8kstyle.jsonl": convert_for_output(subsets["gsm8kstyle"], "full"),
        "train_ablated_gsm8kstyle.jsonl": convert_for_output(subsets["gsm8kstyle"], "ablated"),

        "train_full_mathstyle.jsonl": convert_for_output(subsets["mathstyle"], "full"),
        "train_ablated_mathstyle.jsonl": convert_for_output(subsets["mathstyle"], "ablated"),
    }

    # 先写文件
    for filename, rows in outputs.items():
        write_jsonl(rows, out_dir / filename)

    # 再强校验 ablated
    validate_strong_ablated(
        outputs["train_ablated_clean.jsonl"],
        out_dir / "strong_ablation_residue_examples.json"
    )
    validate_strong_ablated(
        outputs["train_ablated_codeonly.jsonl"],
        out_dir / "strong_ablation_residue_examples.json"
    )
    validate_strong_ablated(
        outputs["train_ablated_gsm8kstyle.jsonl"],
        out_dir / "strong_ablation_residue_examples.json"
    )
    validate_strong_ablated(
        outputs["train_ablated_mathstyle.jsonl"],
        out_dir / "strong_ablation_residue_examples.json"
    )

    save_report(
        path=out_dir / "clean_report.json",
        raw_count=len(raw_data),
        valid_count_before_dedup=before_dedup,
        valid_count_after_dedup=len(deduped_samples),
        duplicate_removed=duplicate_removed,
        skipped_reasons=skipped_reasons,
        samples=deduped_samples,
    )

    print("=" * 72)
    print("强消融数据清洗完成")
    print("=" * 72)
    print(f"输入文件: {input_path}")
    print(f"原始样本数: {len(raw_data)}")
    print(f"有效样本数(去重前): {before_dedup}")
    print(f"有效样本数(去重后): {len(deduped_samples)}")
    print(f"去重移除: {duplicate_removed}")
    print("")
    print(f"全部 full:     {len(outputs['train_full_clean.jsonl'])}")
    print(f"全部 ablated: {len(outputs['train_ablated_clean.jsonl'])}")
    print("")
    print(f"Code-only full:     {len(outputs['train_full_codeonly.jsonl'])}")
    print(f"Code-only ablated: {len(outputs['train_ablated_codeonly.jsonl'])}")
    print("")
    print(f"GSM8K风格 full:     {len(outputs['train_full_gsm8kstyle.jsonl'])}")
    print(f"GSM8K风格 ablated: {len(outputs['train_ablated_gsm8kstyle.jsonl'])}")
    print(f"MATH风格 full:     {len(outputs['train_full_mathstyle.jsonl'])}")
    print(f"MATH风格 ablated: {len(outputs['train_ablated_mathstyle.jsonl'])}")
    print("")
    print("强消融校验：通过")
    print(f"统计报告: {out_dir / 'clean_report.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
