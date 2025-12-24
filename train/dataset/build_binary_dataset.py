"""
Build a binary adherence dataset from existing SFT JSON files and optional
decision_tester logs.

Sources:
- train/dataset/sft/randomized_decision_making.json: label via Action:
  - "Action: a)" => Adhering to Human: True
  - "Action: b)" => Adhering to Human: False
- Optional: decision tester sample logs (*_samples.json) where structure was
  followed. Uses prompt + expected_output.

Output:
- train/dataset/sft/binary_phi3/combined/full_data.json with Unsloth chat
  template for phi-3 (human/gpt roles).

Usage:
    python -m train.dataset.build_binary_dataset \
        --out train/dataset/sft/binary_phi3/combined/full_data.json \
        --decision-json train/dataset/sft/randomized_decision_making.json \
        --logs 'tests/decision_tester/logs/**/*_samples.json'

Notes:
- Skips samples without a clear binary label.
- By default, normalizes source prompts that use the old `Action: a/b`
  format into the binary `True/False` prompt format (use `--keep-prompts`
  to preserve the original prompts).
- If present in the source, appends the model's reasoning/explanation after
  the `Adhering to Human: ...` line.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any


# Accept "Action: a)" or "Action: a" (same for b)
ACTION_TRUE_RE = re.compile(r"Action:\s*a\)?\b", re.IGNORECASE)
ACTION_FALSE_RE = re.compile(r"Action:\s*b\)?\b", re.IGNORECASE)
ACTION_SPLIT_RE = re.compile(r"\bAction\s*:\s*", re.IGNORECASE)
ADHERING_LINE_RE = re.compile(
    r"(?im)^\s*adhering\s*to\s*human\s*:\s*(true|false)\s*$"
)

PROMPT_TWO_ACTIONS_RE = re.compile(r"(?i)(by choosing from the two)\s+actions\s*:")
PROMPT_OPTION_A_RE = re.compile(r"(?im)^(?P<indent>[ \t]*)a\)\s*Continue\s*:.*$")
PROMPT_OPTION_B_RE = re.compile(r"(?im)^(?P<indent>[ \t]*)b\)\s*Correct\s*:.*$")
PROMPT_STRICT_BLOCK_RE = re.compile(
    r"(?is)\n(?P<indent>[ \t]*)Strictly adhere to the reply format:\s*\n.*$"
)


def normalize_binary_prompt(prompt: str) -> str:
    """
    Normalize prompts from the old action format:
      - "two actions" / Continue-Correct / Action: <a or b>
    to the new binary prompt format:
      - "two possibilities" / True-False / Adhering to Human: <True or False>
    """
    if not isinstance(prompt, str) or not prompt:
        return prompt

    if re.search(r"(?i)Adhering to Human\s*:\s*<True or False>", prompt):
        return prompt

    lower = prompt.lower()
    looks_old = (
        "two actions" in lower
        or "action: <a or b>" in lower
        or "a) continue:" in lower
        or "b) correct:" in lower
    )
    if not looks_old:
        return prompt

    out = prompt
    out = PROMPT_TWO_ACTIONS_RE.sub(r"\1 possibilities:", out)

    def repl_a(m: re.Match) -> str:
        indent = m.group("indent") or ""
        return (
            f"{indent}a) True: The car is driving as expected and should continue "
            f"driving in the same manner."
        )

    def repl_b(m: re.Match) -> str:
        indent = m.group("indent") or ""
        return (
            f"{indent}b) False: The car is not driving as expected and state how "
            f"the car should correct its driving style."
        )

    out = PROMPT_OPTION_A_RE.sub(repl_a, out)
    out = PROMPT_OPTION_B_RE.sub(repl_b, out)

    m = PROMPT_STRICT_BLOCK_RE.search(out)
    if m:
        indent = m.group("indent") or ""
        out = (
            out[: m.start()]
            + "\n"
            + f"{indent}Strictly adhere to the reply format:\n\n"
            + f"{indent}Adhering to Human: <True or False>\n"
            + f"{indent}State Recap: <Brief Explanation>\n\n"
        )

    return out


def extract_reasoning(raw: str) -> str:
    """
    Extract the "reasoning" portion from a model response.

    - For action-labeled responses, drop the trailing `Action: ...` block.
    - Drop any standalone `Adhering to Human: ...` line so we can append the
      remaining reasoning after our normalized label.
    """
    if not raw:
        return ""

    text = raw.strip()
    if not text:
        return ""

    # Drop action + everything after it (common in randomized_decision_making)
    m = ACTION_SPLIT_RE.search(text)
    if m:
        text = text[: m.start()].rstrip()

    # Drop standalone adherence lines
    text = ADHERING_LINE_RE.sub("", text).strip()
    return text


def format_binary_answer(label: bool, reasoning: str) -> str:
    out = f"Adhering to Human: {'True' if label else 'False'}"
    if reasoning:
        out += f"\n\n{reasoning}"
    return out


def load_decision_json(path: Path, *, keep_prompts: bool) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    out = []
    for entry in data:
        conv = entry.get("conversations", [])
        if len(conv) < 2:
            continue
        user = conv[0].get("value") or conv[0].get("content")
        assistant = conv[1].get("value") or conv[1].get("content") or ""
        if user is None:
            continue
        if not keep_prompts:
            user = normalize_binary_prompt(user)
        label = None
        if ACTION_TRUE_RE.search(assistant):
            label = True
        elif ACTION_FALSE_RE.search(assistant):
            label = False
        if label is None:
            continue
        reasoning = extract_reasoning(assistant)
        out.append({
            "conversations": [
                {"from": "human", "value": user},
                {"from": "gpt", "value": format_binary_answer(label, reasoning)}
            ]
        })
    return out


def load_log_json(path: Path, *, keep_prompts: bool) -> List[Dict[str, Any]]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("structure_followed", False):
                continue
            prompt = rec.get("prompt")
            label = rec.get("expected_output")
            if prompt is None or not isinstance(label, bool):
                continue
            if not keep_prompts:
                prompt = normalize_binary_prompt(prompt)
            raw_response = None
            for key in ("model_response_raw", "model_response", "response", "completion", "output"):
                val = rec.get(key)
                if isinstance(val, str) and val.strip():
                    raw_response = val
                    break
            reasoning = extract_reasoning(raw_response or "")
            out.append({
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": format_binary_answer(label, reasoning)}
                ]
            })
    return out


def main():
    parser = argparse.ArgumentParser(description="Build binary adherence dataset")
    parser.add_argument("--out", type=str, required=True,
                        help="Output JSON path (e.g., train/dataset/sft/binary_phi3/combined/full_data.json)")
    parser.add_argument("--decision-json", type=str, default="train/dataset/sft/randomized_decision_making.json",
                        help="Path to randomized decision making JSON")
    parser.add_argument("--logs", type=str, nargs="*", default=[],
                        help="Glob(s) for decision tester *_samples.json logs")
    parser.add_argument("--keep-prompts", action="store_true",
                        help="Keep source prompts as-is (do not normalize old Action a/b prompts to True/False)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples: List[Dict[str, Any]] = []

    # Decision JSON
    decision_path = Path(args.decision_json)
    if decision_path.exists():
        print(f"Loading decision data from {decision_path}")
        samples.extend(load_decision_json(decision_path, keep_prompts=args.keep_prompts))
    else:
        print(f"WARN: {decision_path} not found, skipping")

    # Logs (globs)
    for pattern in args.logs:
        for p in Path().glob(pattern):
            if p.is_file():
                print(f"Loading log entries from {p}")
                samples.extend(load_log_json(p, keep_prompts=args.keep_prompts))

    print(f"Collected {len(samples)} samples")
    out_path.write_text(json.dumps(samples, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
