import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

NUM_RE = re.compile(r'[-+]?\d[\d,]*\.?\d*(?:/\d+)?')
CODE_BLOCK_RE = re.compile(r'```python\s*(.*?)```', re.S | re.I)
BOXED_RE = re.compile(r'\\boxed\{([^{}]+)\}')
FINAL_PATTERNS = [
    re.compile(r'Final\s*Answer\s*[:：]\s*(.+)', re.I),
    re.compile(r'最终答案(?:是|为)?\s*[:：]?\s*(.+)'),
    re.compile(r'答案(?:是|为)?\s*[:：]?\s*(.+)'),
]
EXEC_SECTION_RE = re.compile(
    r'(?:###\s*)?(?:代码执行结果|运行结果|输出结果|程序输出|执行输出)\s*[:：]?.*?(?=(?:\n\s*(?:因此|所以|综上|故|可得|最终答案|Final\s*Answer))|\Z)',
    re.S | re.I,
)
AFTER_CODE_FLUFF_RE = re.compile(
    r'(?:根据(?:以上|上述)?Python(?:代码)?(?:执行)?结果.*?|由(?:代码|程序)可得.*?|可见.*?)(?=(?:\n\s*(?:因此|所以|综上|故|可得|最终答案|Final\s*Answer))|\Z)',
    re.S | re.I,
)
LEADING_BOILERPLATE_RE = re.compile(
    r'^你是一个数学解题大师.*?(?=题目[:：])',
    re.S,
)

STANDARD_INSTRUCTION = (
    '请解决下面的数学题。可以给出必要的推理，并在需要时给出 Python 代码。\n'
    '最后请单独一行输出：Final Answer: <answer>'
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', str(text))
    text = text.replace('\u2212', '-')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text.strip()


def extract_question(sample: Dict[str, Any]) -> str:
    instruction = normalize_text(sample.get('instruction', ''))
    input_text = normalize_text(sample.get('input', ''))

    # 你的数据里真实题目主要在 input 字段
    if input_text:
        input_text = re.sub(r'^题目[:：]\s*', '', input_text).strip()
        return input_text

    # 兼容 instruction 中直接包含题目
    instruction = LEADING_BOILERPLATE_RE.sub('', instruction).strip()
    instruction = re.sub(r'^题目[:：]\s*', '', instruction).strip()
    return instruction


def extract_first_code_block(text: str) -> Optional[str]:
    m = CODE_BLOCK_RE.search(text)
    if not m:
        return None
    code = m.group(1).strip()
    return code if code else None


def canonicalize_number_str(num_str: str) -> str:
    s = normalize_text(num_str)
    s = s.replace(',', '')
    if s.endswith('.0'):
        try:
            f = float(s)
            if f.is_integer():
                return str(int(f))
        except Exception:
            pass
    return s


def extract_boxed_answers(text: str) -> List[str]:
    answers: List[str] = []
    for piece in BOXED_RE.findall(text):
        nums = NUM_RE.findall(piece.replace(',', ''))
        if nums:
            answers.append(canonicalize_number_str(nums[-1]))
    return answers


def extract_final_answer_text(text: str) -> Optional[str]:
    text = normalize_text(text)

    boxed_answers = extract_boxed_answers(text)
    if boxed_answers:
        if len(boxed_answers) == 1:
            return boxed_answers[0]
        return ', '.join(boxed_answers)

    for pat in FINAL_PATTERNS:
        m = pat.search(text)
        if m:
            seg = normalize_text(m.group(1))
            nums = NUM_RE.findall(seg.replace(',', ''))
            if nums:
                if len(nums) == 1:
                    return canonicalize_number_str(nums[0])
                return ', '.join(canonicalize_number_str(x) for x in nums)
            return seg

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    tail = lines[-5:] if len(lines) >= 5 else lines
    for line in reversed(tail):
        nums = NUM_RE.findall(line.replace(',', ''))
        if nums:
            if len(nums) == 1:
                return canonicalize_number_str(nums[0])
            return ', '.join(canonicalize_number_str(x) for x in nums)
    return None


def strip_execution_sections(text: str) -> str:
    text = EXEC_SECTION_RE.sub('\n', text)
    text = AFTER_CODE_FLUFF_RE.sub('\n', text)
    return text


def clean_reasoning_prefix(text: str) -> str:
    text = normalize_text(text)
    text = strip_execution_sections(text)

    # 只保留第一个 python 代码块之前的推理正文 + 第一个代码块
    code = extract_first_code_block(text)
    if code is None:
        return text

    parts = CODE_BLOCK_RE.split(text, maxsplit=1)
    reasoning = normalize_text(parts[0]) if parts else ''

    # 去掉“Python代码验证/生成的Python代码如下”等模板句，减少风格噪声
    reasoning = re.sub(r'(?:###\s*)?Python代码验证[:：]?\s*', '', reasoning, flags=re.I)
    reasoning = re.sub(r'生成的Python代码如下[:：]?\s*', '', reasoning, flags=re.I)
    reasoning = re.sub(r'我们可以使用Python代码.*?(?=```python|$)', '', reasoning, flags=re.S | re.I)
    reasoning = reasoning.strip()

    block = f"```python\n{code}\n```"
    if reasoning:
        return f"{reasoning}\n\n{block}"
    return block


def build_clean_output(raw_output: str, answer_text: str) -> str:
    core = clean_reasoning_prefix(raw_output)
    final_line = f'Final Answer: {answer_text}'
    if core:
        return f'{core}\n\n{final_line}'
    return final_line


def normalize_for_dedup(question: str, output: str) -> Tuple[str, str]:
    q = normalize_text(question)
    o = normalize_text(output)
    q = re.sub(r'\s+', ' ', q)
    o = re.sub(r'\s+', ' ', o)
    return q, o


def clean_sample(sample: Dict[str, Any]) -> Tuple[Optional[Dict[str, str]], str]:
    if not isinstance(sample, dict):
        return None, 'not_dict'

    raw_output = normalize_text(sample.get('output', ''))
    if not raw_output:
        return None, 'empty_output'

    question = extract_question(sample)
    if not question:
        return None, 'empty_question'

    code = extract_first_code_block(raw_output)
    if code is None:
        return None, 'no_python_code'

    answer_text = extract_final_answer_text(raw_output)
    if not answer_text:
        return None, 'no_final_answer'

    # 过长代码块 / 明显脏样本可选过滤
    if len(code) > 2500:
        return None, 'code_too_long'

    cleaned = {
        'instruction': STANDARD_INSTRUCTION,
        'input': f'题目：{question}',
        'output': build_clean_output(raw_output, answer_text),
    }
    return cleaned, 'ok'


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f'第 {i} 行 JSON 解析失败: {e}') from e
    return rows


def write_jsonl(rows: List[Dict[str, str]], path: Path) -> None:
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Clean full code-only training data for LoRA SFT.')
    parser.add_argument('--input', required=True, help='Path to train_full_codeonly.jsonl')
    parser.add_argument('--output', required=True, help='Path to cleaned jsonl output')
    parser.add_argument('--report', default=None, help='Optional path to save cleaning report json')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_suffix('.report.json')

    raw_rows = read_jsonl(input_path)

    cleaned_rows: List[Dict[str, str]] = []
    stats: Dict[str, int] = {}
    seen = set()
    duplicate_removed = 0

    for row in raw_rows:
        cleaned, reason = clean_sample(row)
        stats[reason] = stats.get(reason, 0) + 1
        if cleaned is None:
            continue

        key = normalize_for_dedup(cleaned['input'], cleaned['output'])
        if key in seen:
            duplicate_removed += 1
            continue
        seen.add(key)
        cleaned_rows.append(cleaned)

    write_jsonl(cleaned_rows, output_path)

    report = {
        'input_path': str(input_path),
        'output_path': str(output_path),
        'raw_count': len(raw_rows),
        'cleaned_count_before_dedup': stats.get('ok', 0),
        'cleaned_count_after_dedup': len(cleaned_rows),
        'duplicate_removed': duplicate_removed,
        'reason_counter': stats,
        'output_example': cleaned_rows[0] if cleaned_rows else None,
    }

    with report_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('=' * 70)
    print('full 数据清洗完成')
    print('=' * 70)
    print(f'输入文件: {input_path}')
    print(f'原始样本数: {len(raw_rows)}')
    print(f'清洗保留(去重前): {stats.get("ok", 0)}')
    print(f'清洗保留(去重后): {len(cleaned_rows)}')
    print(f'去重移除: {duplicate_removed}')
    print('')
    for k in sorted(stats.keys()):
        if k != 'ok':
            print(f'{k}: {stats[k]}')
    print('')
    print(f'输出文件: {output_path}')
    print(f'报告文件: {report_path}')
    print('=' * 70)


if __name__ == '__main__':
    main()
