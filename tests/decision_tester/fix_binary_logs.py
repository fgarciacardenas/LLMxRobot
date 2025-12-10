#!/usr/bin/env python3
"""
Trim binary decision_tester logs where the model already emitted
"Adhering to Human: <True/False>" but continued generating.

For each *_samples.json line:
- If "model_response_raw" contains "Adhering to Human: True/False",
  truncate the raw text at the end of that marker and set output_tokens
  to a small fixed value (default: 7).
- Otherwise leave the entry untouched.

Run from repo root, for example:
    python src/LLMxRobot/tests/decision_tester/fix_binary_logs.py \
        --logs src/LLMxRobot/tests/decision_tester/logs/report_logs \
        --token-fix 7
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Tuple

ADHERING_RE = re.compile(r"adhering\s*to\s*human\s*:\s*(true|false)", re.IGNORECASE)


def process_file(path: Path, token_fix: int) -> Tuple[int, int]:
    """Return (lines_processed, lines_modified)."""
    lines = path.read_text().splitlines()
    out_lines = []
    modified = 0
    for line in lines:
        if not line.strip():
            continue
        obj = json.loads(line)
        raw = str(obj.get("model_response_raw", ""))
        m = ADHERING_RE.search(raw)
        if m:
            trimmed = raw[: m.end()]
            if trimmed != raw or obj.get("output_tokens") != token_fix:
                obj["model_response_raw"] = trimmed
                obj["output_tokens"] = token_fix
                modified += 1
        out_lines.append(json.dumps(obj))

    if modified:
        path.write_text("\n".join(out_lines) + "\n")
    return len(out_lines), modified


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim binary decision_tester logs after adherence marker.")
    parser.add_argument(
        "--logs",
        type=Path,
        default=Path("src/LLMxRobot/tests/decision_tester/logs/report_logs"),
        help="Directory to scan for *_samples.json files.",
    )
    parser.add_argument(
        "--token-fix",
        type=int,
        default=7,
        help="Value to set for output_tokens when trimming.",
    )
    parser.add_argument(
        "--require-substrings",
        type=str,
        default=None,
        help="Comma-separated substrings that must appear in the file path (case-insensitive), e.g. 'qwen,binary'.",
    )
    args = parser.parse_args()

    if not args.logs.exists():
        raise SystemExit(f"Log directory not found: {args.logs}")

    files = sorted(args.logs.rglob("*_samples.json"))
    if not files:
        raise SystemExit(f"No *_samples.json files found under {args.logs}")

    required = []
    if args.require_substrings:
        required = [s.strip().lower() for s in args.require_substrings.split(",") if s.strip()]

    total_lines = total_modified = total_files = files_changed = 0
    for fp in files:
        if required:
            low = fp.as_posix().lower()
            if not all(sub in low for sub in required):
                continue
        lines, modified = process_file(fp, args.token_fix)
        total_lines += lines
        total_modified += modified
        total_files += 1
        if modified:
            files_changed += 1

    print(f"Processed {total_files} files ({files_changed} changed), {total_lines} lines, {total_modified} trimmed.")


if __name__ == "__main__":
    main()
