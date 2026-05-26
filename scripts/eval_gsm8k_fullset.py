import os
import re
import gc
import json
import math
import argparse
import unicodedata
from fractions import Fraction

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

NUMBER_RE = re.compile(r'[-+]?\d[\d,]*\.?\d*(?:/\d+)?')
BOXED_RE = re.compile(r'\\boxed\{([^{}]+)\}')
HASH_RE = re.compile(r'####\s*([^\n\r]+)')
FINAL_PATTERNS = [
    re.compile(r'Final Answer\s*[:：]\s*(.*)', re.I),
    re.compile(r'最终答案(?:是|为)?\s*[:：]?\s*(.*)'),
    re.compile(r'答案(?:是|为)?\s*[:：]?\s*(.*)'),
]

def normalize_text(text):
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\u2212", "-")
    text = text.replace("，", ",").replace("：", ":").replace("．", ".")
    text = text.replace("$", "").replace("¥", "").replace("￥", "")
    return text.strip()

def parse_number(num_str):
    num_str = normalize_text(num_str).replace(",", "")
    if not num_str:
        return None
    try:
        if "/" in num_str and num_str.count("/") == 1:
            return float(Fraction(num_str))
        return float(num_str)
    except Exception:
        return None

def extract_final_segment(text):
    text = normalize_text(text)

    boxed = BOXED_RE.findall(text)
    if boxed:
        return normalize_text(boxed[-1])

    hashed = HASH_RE.findall(text)
    if hashed:
        return normalize_text(hashed[-1])

    for pattern in FINAL_PATTERNS:
        m = pattern.search(text)
        if m:
            return normalize_text(m.group(1))

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return normalize_text(lines[-1])

    return text

def extract_final_number(text):
    text = normalize_text(text)
    seg = extract_final_segment(text)

    nums = NUMBER_RE.findall(seg)
    if nums:
        return nums[0]

    nums = NUMBER_RE.findall(text)
    if nums:
        return nums[-1]

    return ""

def answers_equal(pred_text, gold_text, tol=1e-6):
    pred_num = parse_number(extract_final_number(pred_text))
    gold_num = parse_number(extract_final_number(gold_text))

    if pred_num is not None and gold_num is not None:
        return math.isclose(pred_num, gold_num, rel_tol=tol, abs_tol=tol)

    return normalize_text(extract_final_segment(pred_text)) == normalize_text(extract_final_segment(gold_text))

def build_prompt(question):
    return (
        "请解答下面的数学题。\n"
        "你可以进行必要推理，但最后一行必须严格按下面格式输出，且只能输出一个数字，不要输出单位、解释或其他文字：\n"
        "Final Answer: <number>\n\n"
        f"题目：{question}"
    )

@torch.no_grad()
def generate_one(model, tokenizer, question, max_new_tokens=512):
    messages = [
        {"role": "system", "content": "你是一个擅长数学解题的助手。"},
        {"role": "user", "content": build_prompt(question)},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="base / ablated / full / full_v2")
    parser.add_argument("--test_path", type=str, default="/mnt/workspace/gsm8k_test.json")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="/mnt/workspace/modelscope_cache/Qwen/Qwen2___5-7B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    adapter_map = {
        "base": None,
        "ablated": "/mnt/workspace/LlamaFactory/saves/qwen2.5-7b/lora/train_ablated_codeonly",
        "full": "/mnt/workspace/LlamaFactory/saves/qwen2.5-7b/lora/train_full_codeonly",
        "full_v2": "/mnt/workspace/LlamaFactory/saves/qwen2.5-7b/lora/train_full_codeonly_v2",
    }

    if args.mode not in adapter_map:
        raise ValueError(f"不支持的 mode: {args.mode}")

    with open(args.test_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {args.test_path}")
    print(f"Evaluating mode: {args.mode}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto"
    )

    adapter_path = adapter_map[args.mode]
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    total = len(data)
    correct = 0
    details = []

    for idx, item in enumerate(data, 1):
        question = item.get("instruction") or item.get("question") or item.get("input") or ""
        gold_raw = item.get("gold_answer") or item.get("answer") or item.get("label") or ""

        answer_raw = generate_one(model, tokenizer, question, args.max_new_tokens)
        pred_answer = extract_final_number(answer_raw)
        gold_answer = extract_final_number(gold_raw)
        is_correct = answers_equal(answer_raw, gold_raw)

        if is_correct:
            correct += 1

        details.append({
            "idx": idx,
            "question": question,
            "gold_raw": gold_raw,
            "gold_answer": gold_answer,
            "answer_raw": answer_raw,
            "pred_answer": pred_answer,
            "is_correct": is_correct,
        })

        print(f"[{idx}/{total}] gold={gold_answer}, pred={pred_answer}, correct={is_correct}")

    accuracy = correct / total if total else 0.0

    result = {
        "mode": args.mode,
        "test_path": args.test_path,
        "summary": {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 6),
        },
        "details": details
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n评测完成")
    print("mode:", args.mode)
    print("total:", total)
    print("correct:", correct)
    print("accuracy:", round(accuracy, 6))
    print("saved to:", args.output_path)

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
