# check_ablated_residue.py
# 用途：检查 train_ablated_codeonly.jsonl 里是否还残留代码或执行痕迹

import json
import re
from pathlib import Path
from collections import Counter

INPUT_PATH = Path("train_ablated_codeonly.jsonl")
REPORT_PATH = Path("ablated_residue_report.json")

PATTERNS = {
    "python_fence": re.compile(r"```(?:python)?", re.I),
    "import_stmt": re.compile(r"^\s*import\s+\w+|^\s*from\s+\w+\s+import\s+", re.M),
    "def_func": re.compile(r"^\s*def\s+\w+\s*\(", re.M),
    "print_call": re.compile(r"\bprint\s*\("),
    "return_stmt": re.compile(r"^\s*return\b", re.M),
    "for_loop": re.compile(r"^\s*for\s+.+\s+in\s+.+:", re.M),
    "while_loop": re.compile(r"^\s*while\s+.+:", re.M),
    "if_stmt": re.compile(r"^\s*if\s+.+:", re.M),
    "assign_like_code": re.compile(r"^\s*[A-Za-z_]\w*\s*=\s*.+", re.M),
    "exec_result_cn": re.compile(r"(代码执行结果|运行结果|程序输出|执行输出)"),
    "python_result_cn": re.compile(r"(根据(?:以上|上述)?Python(?:代码)?(?:执行)?结果|由(?:代码|程序)可得|运行上述代码)"),
}

def detect_residue(text: str):
    hits = []
    for name, pattern in PATTERNS.items():
        if pattern.search(text):
            hits.append(name)
    return hits

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"找不到文件: {INPUT_PATH}")

    total = 0
    residue_count = 0
    reason_counter = Counter()
    examples = []

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            item = json.loads(line)
            output = str(item.get("output", ""))

            hits = detect_residue(output)
            if hits:
                residue_count += 1
                reason_counter.update(hits)

                examples.append({
                    "line_no": idx,
                    "hits": hits,
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", "")[:300],
                    "output_preview": output[:800]
                })

    report = {
        "input_path": str(INPUT_PATH),
        "total_samples": total,
        "residue_samples": residue_count,
        "clean_samples": total - residue_count,
        "residue_ratio": round(residue_count / total, 6) if total else 0.0,
        "reason_counter": dict(reason_counter),
        "examples": examples[:50]   # 只保留前50条样例，方便人工检查
    }

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("ablated 残留检查完成")
    print("=" * 70)
    print(f"输入文件: {INPUT_PATH}")
    print(f"总样本数: {total}")
    print(f"检测到残留的样本数: {residue_count}")
    print(f"无残留样本数: {total - residue_count}")
    print(f"残留比例: {report['residue_ratio']:.6f}")
    print("")
    if reason_counter:
        print("残留类型统计:")
        for k, v in reason_counter.most_common():
            print(f"  {k}: {v}")
    else:
        print("未发现代码或执行痕迹残留。")
    print("")
    print(f"报告文件: {REPORT_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    main()