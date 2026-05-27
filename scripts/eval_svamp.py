import argparse
import json
import re
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def build_svamp_question(item):
    body = item.get("Body", "").strip()
    question = item.get("Question", "").strip()
    return body + "\n" + question


def normalize_number(x):
    x = str(x).strip().replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", x)
    if nums:
        return nums[-1]
    return None


def extract_answer(text):
    text = str(text)

    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        ans = normalize_number(boxed[-1])
        if ans is not None:
            return ans

    patterns = [
        r"final answer is\s*[:：]?\s*(-?\d+(?:\.\d+)?)",
        r"answer is\s*[:：]?\s*(-?\d+(?:\.\d+)?)",
        r"最终答案\s*[:：]?\s*(-?\d+(?:\.\d+)?)",
    ]

    for p in patterns:
        m = re.findall(p, text, flags=re.IGNORECASE)
        if m:
            return m[-1]

    return normalize_number(text)


def is_equal(pred, gold, tol=1e-6):
    try:
        return abs(float(pred) - float(gold)) < tol
    except Exception:
        return str(pred).strip() == str(gold).strip()


def load_model(base_model_path, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

    model.eval()
    return model, tokenizer


def eval_svamp(model, tokenizer, data, mode="base", limit=None, max_new_tokens=256):
    results = []
    type_counter = Counter()

    test_data = data[:limit] if limit else data

    for idx, item in enumerate(test_data, 1):
        question = build_svamp_question(item)
        gold = normalize_number(item.get("Answer", ""))

        prompt = (
            "Please solve the following math word problem step by step. "
            "At the end, give the final answer in the format \\boxed{answer}.\n\n"
            f"Problem:\n{question}"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = extract_answer(decoded)

        if pred is None:
            pred = "NONE"
            correct = False
            result_type = "extract_error"
        else:
            correct = is_equal(pred, gold)
            result_type = "correct" if correct else "reasoning_error"

        type_counter[result_type] += 1

        results.append({
            "id": idx,
            "question": question,
            "gold": gold,
            "pred": pred,
            "correct": correct,
            "type": result_type,
            "model_output": decoded,
            "raw_item": item
        })

        print(
            f"[{idx}/{len(test_data)}] "
            f"gold={gold}, pred={pred}, correct={correct}, type={result_type}"
        )

    correct_num = sum(r["correct"] for r in results)

    summary = {
        "mode": mode,
        "dataset": "SVAMP",
        "total": len(results),
        "correct": correct_num,
        "accuracy": round(correct_num / len(results), 4),
        "type_counter": dict(type_counter),
        "results": results
    }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="base")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    args = parser.parse_args()

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model, tokenizer = load_model(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path
    )

    summary = eval_svamp(
        model=model,
        tokenizer=tokenizer,
        data=data,
        mode=args.mode,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens
    )

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n评测完成")
    print("mode:", summary["mode"])
    print("total:", summary["total"])
    print("correct:", summary["correct"])
    print("accuracy:", summary["accuracy"])
    print("type_counter:", summary["type_counter"])
    print("saved to:", args.output_path)


if __name__ == "__main__":
    main()
