import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path


BOXED_RE = re.compile(r'\\boxed\{([^{}]+)\}')
HASH_RE = re.compile(r'####\s*([^\n\r]+)')
FINAL_PATTERNS = [
    re.compile(r'最终答案(?:是|为)?[:：]?\s*([^\n\r。]+)'),
    re.compile(r'答案(?:是|为)?[:：]?\s*([^\n\r。]+)'),
    re.compile(r'the answer is[:：]?\s*([^\n\r.]+)', re.I),
]
NUMBER_RE = re.compile(r'[-+]?\d[\d,]*\.?\d*(?:/\d+)?')
CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S | re.I)
PLAIN_PY_RE = re.compile(
    r'(?:^|\n)python\s*\n(.*?)(?=\n(?:因此|所以|答案|最终答案|输出|Output|$))',
    re.S | re.I
)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def strip_text(s: str) -> str:
    s = str(s).strip()
    s = s.replace("，", ",").replace("：", ":").replace("。", ".")
    s = s.replace("$", "").replace("¥", "").replace("￥", "")
    s = s.replace("\\(", "").replace("\\)", "")
    s = s.replace("{", "").replace("}", "")
    return s.strip()


def try_parse_number(s: str):
    """
    尽量把字符串解析成数值。
    支持:
    - 1300
    - 1,300
    - 18.0
    - 3/4
    - 文本里夹带一个数，如 '1300 千克'
    """
    if s is None:
        return None

    s = strip_text(s)

    boxed = BOXED_RE.findall(s)
    if boxed:
        s = boxed[-1].strip()

    s = s.replace(",", "")
    nums = NUMBER_RE.findall(s)
    if not nums:
        return None

    candidate = nums[-1].strip()

    try:
        if "/" in candidate and candidate.count("/") == 1:
            return float(Fraction(candidate))
        return float(candidate)
    except Exception:
        return None


def normalize_text_answer(s: str) -> str:
    if s is None:
        return ""

    s = strip_text(s)

    boxed = BOXED_RE.findall(s)
    if boxed:
        s = boxed[-1].strip()

    s = s.lower()
    s = s.replace(" ", "")
    s = s.replace("\n", "")
    s = s.replace("\t", "")
    s = s.rstrip(".,;:!？。；：")
    return s


def extract_final_answer(raw_text: str) -> str:
    """
    从模型原始输出里尽量抽最终答案。
    优先级：
    1. \\boxed{}
    2. ####
    3. “最终答案是/答案是”
    4. 最后一个数字
    5. 最后一行文本
    """
    if raw_text is None:
        return ""

    text = str(raw_text).strip()

    boxed = BOXED_RE.findall(text)
    if boxed:
        return strip_text(boxed[-1])

    hashed = HASH_RE.findall(text)
    if hashed:
        return strip_text(hashed[-1])

    for pattern in FINAL_PATTERNS:
        m = pattern.findall(text)
        if m:
            return strip_text(m[-1])

    nums = NUMBER_RE.findall(text.replace(",", ""))
    if nums:
        return strip_text(nums[-1])

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return strip_text(lines[-1]) if lines else ""


def answers_equal(pred: str, gold: str, tol: float = 1e-6) -> bool:
    pred_num = try_parse_number(pred)
    gold_num = try_parse_number(gold)

    if pred_num is not None and gold_num is not None:
        return math.isclose(pred_num, gold_num, rel_tol=tol, abs_tol=tol)

    return normalize_text_answer(pred) == normalize_text_answer(gold)


def extract_python_code(raw_text: str) -> str:
    if raw_text is None:
        return ""

    text = str(raw_text)

    blocks = CODE_BLOCK_RE.findall(text)
    if blocks:
        return blocks[-1].strip()

    m = PLAIN_PY_RE.search(text)
    if m:
        return m.group(1).strip()

    return ""


def run_python_code(code: str, timeout: int = 5):
    """
    仅做简单可执行性统计。
    返回: (success: bool, stdout: str, stderr: str)
    """
    if not code.strip():
        return False, "", "empty_code"

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        temp_path = f.name
        f.write(code)

    try:
        proc = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        success = proc.returncode == 0
        return success, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    except Exception as e:
        return False, "", repr(e)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def evaluate_file(pred_file: str, out_file: str):
    data = load_json(pred_file)
    assert isinstance(data, list), "JSON 顶层必须是 list"

    total = len(data)
    correct = 0
    code_count = 0
    executable_count = 0

    details = []

    for idx, item in enumerate(data):
        raw = (
            item.get("answer_raw")
            or item.get("prediction")
            or item.get("pred")
            or item.get("output")
            or item.get("response")
            or ""
        )

        gold = (
            item.get("gold_answer")
            or item.get("answer")
            or item.get("label")
            or ""
        )

        question = item.get("instruction") or item.get("question") or ""
        pred = extract_final_answer(raw)
        is_correct = answers_equal(pred, gold)

        if is_correct:
            correct += 1

        code = extract_python_code(raw)
        has_code = bool(code.strip())

        exec_success = None
        exec_stdout = ""
        exec_stderr = ""

        if has_code:
            code_count += 1
            exec_success, exec_stdout, exec_stderr = run_python_code(code, timeout=5)
            if exec_success:
                executable_count += 1

        details.append({
            "idx": idx,
            "question": question,
            "gold_answer": gold,
            "pred_answer": pred,
            "is_correct": is_correct,
            "has_code": has_code,
            "code_executable": exec_success,
            "exec_stdout": exec_stdout,
            "exec_stderr": exec_stderr,
            "answer_raw": raw
        })

    accuracy = correct / total if total else 0.0
    code_gen_rate = code_count / total if total else 0.0
    code_exec_rate = executable_count / code_count if code_count else 0.0

    summary = {
        "pred_file": pred_file,
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "code_count": code_count,
        "code_generation_rate": round(code_gen_rate, 4),
        "executable_code_count": executable_count,
        "code_executability_rate": round(code_exec_rate, 4),
    }

    result = {
        "summary": summary,
        "details": details
    }

    dump_json(result, out_file)

    print("=" * 60)
    print(f"pred_file: {pred_file}")
    print(f"total: {total}")
    print(f"correct: {correct}")
    print(f"accuracy: {accuracy:.2%}")
    print(f"code_count: {code_count}")
    print(f"code_generation_rate: {code_gen_rate:.2%}")
    print(f"executable_code_count: {executable_count}")
    print(f"code_executability_rate: {code_exec_rate:.2%}")
    print(f"detail saved to: {out_file}")
    print("=" * 60)

    # 顺便打印错题索引，方便你人工分析
    wrong_ids = [x["idx"] for x in details if not x["is_correct"]]
    print("wrong sample idx:", wrong_ids[:50])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True, help="包含 answer_raw / gold_answer 的 JSON 文件")
    parser.add_argument("--out_file", type=str, default="eval_result_detail.json", help="评测明细输出文件")
    args = parser.parse_args()

    evaluate_file(args.pred_file, args.out_file)


if __name__ == "__main__":
    main()