import json

count = 0
total = 0
with open("train_full_codeonly.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        total += 1
        out = item["output"]
        if "```python" in out.lower():
            count += 1

print("full中带```python的样本数:", count)
print("总样本数:", total)
print("占比:", count / total if total else 0)