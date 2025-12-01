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
- Keeps prompts as-is; you may post-process if you want the shorter
  binary-only prompt. For now, we reuse source prompts to stay faithful to
  the training distribution.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any


ACTION_TRUE_RE = re.compile(r"Action:\s*a\)", re.IGNORECASE)
ACTION_FALSE_RE = re.compile(r"Action:\s*b\)", re.IGNORECASE)


def load_decision_json(path: Path) -> List[Dict[str, Any]]:
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
        label = None
        if ACTION_TRUE_RE.search(assistant):
            label = True
        elif ACTION_FALSE_RE.search(assistant):
            label = False
        if label is None:
            continue
        out.append({
            "conversations": [
                {"from": "human", "value": user},
                {"from": "gpt", "value": f"Adhering to Human: {'True' if label else 'False'}"}
            ]
        })
    return out


def load_log_json(path: Path) -> List[Dict[str, Any]]:
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
            out.append({
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": f"Adhering to Human: {'True' if label else 'False'}"}
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
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples: List[Dict[str, Any]] = []

    # Decision JSON
    decision_path = Path(args.decision_json)
    if decision_path.exists():
        print(f"Loading decision data from {decision_path}")
        samples.extend(load_decision_json(decision_path))
    else:
        print(f"WARN: {decision_path} not found, skipping")

    # Logs (globs)
    for pattern in args.logs:
        for p in Path().glob(pattern):
            if p.is_file():
                print(f"Loading log entries from {p}")
                samples.extend(load_log_json(p))

    print(f"Collected {len(samples)} samples")
    out_path.write_text(json.dumps(samples, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
