import json
import re

CODE_PATTERNS = [
    re.compile(r"```python", re.I),
    re.compile(r"```"),
    re.compile(r"\bdef\s+\w+\s*\("),
    re.compile(r"\bprint\s*\("),
    re.compile(r"\bimport\s+\w+"),
    re.compile(r"\bfor\s+\w+\s+in\s+"),
    re.compile(r"\bwhile\s+"),
    re.compile(r"\breturn\b"),
    re.compile(r"\bif\s+.*:"),
]

bad_samples = []

with open("train_ablated_codeonly.jsonl", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        item = json.loads(line)
        text = item["output"]

        matched = []
        for p in CODE_PATTERNS:
            if p.search(text):
                matched.append(p.pattern)

        if matched:
            bad_samples.append((idx, matched, text[:800]))

print("残留真实代码样本数:", len(bad_samples))

for idx, matched, preview in bad_samples[:10]:
    print("\n" + "=" * 80)
    print("样本索引:", idx)
    print("命中规则:", matched)
    print("内容预览:\n", preview)