"""
Utility to aggregate decision_tester sample logs.

The decision_tester writes one JSON object per line in files ending with
"_samples.json". Each line includes the model's raw response, the parsed
`sanitized_output`, the `expected_output`, and token counts. This script
computes aggregate metrics such as confusion matrix, adherence parsing rate,
token usage, and hit rates for reaching the max output tokens.

Usage examples:
    python -m tests.decision_tester.analyze_logs \
        tests/decision_tester/logs/unsloth_Phi-3-mini-4k-instruct_full_2025-11-18_15-19-18

    # Or pass explicit files
    python -m tests.decision_tester.analyze_logs \
        tests/decision_tester/logs/**/*_samples.json

Defaults to scanning tests/decision_tester/logs for *_samples.json files.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, List, Dict, Any, Tuple, Optional
import csv


def iter_log_files(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(path.glob("**/*_samples.json"))
        else:
            if path.suffix.lower() == ".json":
                files.append(path)
    # Keep deterministic ordering for reproducibility
    return sorted(f for f in files if f.is_file())


def load_entries(file_path: Path) -> Iterable[Dict[str, Any]]:
    with file_path.open() as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping malformed JSON in {file_path} line {ln}: {e}")


def safe_mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def safe_std(values: List[float]) -> float:
    # population std to match how token stats were reported elsewhere
    return pstdev(values) if len(values) > 1 else 0.0


def summarize_tokens(values: List[int]) -> Tuple[float, float, int, int]:
    return safe_mean(values), safe_std(values), min(values, default=0), max(values, default=0)

def percentile(values: List[int], pct: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(values_sorted[int(k)])
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return float(d0 + d1)

def summarize_percentiles(values: List[int]) -> Dict[str, float]:
    return {
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }

def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    headers = sorted(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze decision tester JSON logs.")
    parser.add_argument("paths", nargs="*", default=["tests/decision_tester/logs"],
                        help="Files or directories to scan for *_samples.json logs.")
    parser.add_argument("--max-output-tokens", type=int, default=512,
                        help="Generation cap used during inference (flag used to count max-token hits).")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Optional path to write aggregate metrics as JSON.")
    parser.add_argument("--csv-out", type=str, default=None,
                        help="Optional path to write per-entry flat CSV (for plotting).")
    parser.add_argument("--group-by", choices=["model", "rag_mode"], default=None,
                        help="Optional grouping for per-group accuracy summary.")
    args = parser.parse_args()

    files = iter_log_files(args.paths)
    if not files:
        print("No log files found. Provide directories or *_samples.json files.")
        return

    all_entries: List[Dict[str, Any]] = []
    for file_path in files:
        all_entries.extend(load_entries(file_path))

    total = len(all_entries)
    if total == 0:
        print("No valid entries found in provided logs.")
        return

    # Counters
    parsed = 0
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "unparsed": 0}
    prompt_tokens: List[int] = []
    rag_tokens: List[int] = []
    output_tokens: List[int] = []
    max_token_hits = 0

    # Token buckets by confusion class
    by_class_tokens = {k: [] for k in ("tp", "tn", "fp", "fn")}

    per_case: Dict[str, Dict[str, int]] = {}

    flat_rows: List[Dict[str, Any]] = []

    for entry in all_entries:
        sanitized = entry.get("sanitized_output")
        expected = entry.get("expected_output")

        # Token stats
        prompt_tokens.append(int(entry.get("prompt_tokens", 0)))
        rag_tokens.append(int(entry.get("rag_tokens", 0)))
        out_tok = int(entry.get("output_tokens", 0))
        output_tokens.append(out_tok)
        if out_tok >= args.max_output_tokens:
            max_token_hits += 1

        row = {
            "test_case": entry.get("test_case"),
            "sample_index": entry.get("sample_index"),
            "rag_mode": entry.get("rag_mode"),
            "rag_k": entry.get("rag_k"),
            "rag_threshold": entry.get("rag_threshold"),
            "rag_max_hits": entry.get("rag_max_hits"),
            "model_response_raw": entry.get("model_response_raw"),
            "sanitized_output": sanitized,
            "expected_output": expected,
            "prompt_tokens": entry.get("prompt_tokens"),
            "rag_tokens": entry.get("rag_tokens"),
            "output_tokens": out_tok,
            "model_name": entry.get("model_name"),
        }
        flat_rows.append(row)

        # Structure adherence
        if isinstance(sanitized, bool):
            parsed += 1
            if isinstance(expected, bool):
                if sanitized and expected:
                    confusion["tp"] += 1
                    by_class_tokens["tp"].append(out_tok)
                elif not sanitized and not expected:
                    confusion["tn"] += 1
                    by_class_tokens["tn"].append(out_tok)
                elif sanitized and not expected:
                    confusion["fp"] += 1
                    by_class_tokens["fp"].append(out_tok)
                elif not sanitized and expected:
                    confusion["fn"] += 1
                    by_class_tokens["fn"].append(out_tok)
        else:
            confusion["unparsed"] += 1

        # Per-test-case accuracy
        case = entry.get("test_case", "<unknown>")
        per_case.setdefault(case, {"correct": 0, "total": 0})
        per_case[case]["total"] += 1
        if isinstance(sanitized, bool) and isinstance(expected, bool) and sanitized == expected:
            per_case[case]["correct"] += 1

    # Aggregate metrics
    structure_rate = parsed / total
    accuracy = (confusion["tp"] + confusion["tn"]) / parsed if parsed else 0.0
    max_token_rate = max_token_hits / total

    print(f"Files analyzed: {len(files)}")
    print(f"Total samples: {total}")
    print(f"Parsed structure (sanitized_output present): {parsed} ({structure_rate:.2%})")
    print(f"Unparsed / missing adherence tag: {confusion['unparsed']} ({1 - structure_rate:.2%})")
    print()

    print("Confusion matrix (parsed only):")
    print(f"  TP: {confusion['tp']}")
    print(f"  TN: {confusion['tn']}")
    print(f"  FP: {confusion['fp']}")
    print(f"  FN: {confusion['fn']}")
    print(f"  Parsed accuracy: {accuracy:.2%}")
    denom_pos = confusion['tp'] + confusion['fn']
    denom_neg = confusion['tn'] + confusion['fp']
    recall = confusion['tp'] / denom_pos if denom_pos else 0.0
    specificity = confusion['tn'] / denom_neg if denom_neg else 0.0
    precision = confusion['tp'] / (confusion['tp'] + confusion['fp']) if (confusion['tp'] + confusion['fp']) else 0.0
    print(f"  Recall: {recall:.2%}, Specificity: {specificity:.2%}, Precision: {precision:.2%}")
    print()

    # Token summaries
    p_mean, p_std, p_min, p_max = summarize_tokens(prompt_tokens)
    r_mean, r_std, r_min, r_max = summarize_tokens(rag_tokens)
    o_mean, o_std, o_min, o_max = summarize_tokens(output_tokens)
    o_pct = summarize_percentiles(output_tokens)
    print("Token usage (all samples):")
    print(f"  Prompt tokens : mean {p_mean:.1f}, std {p_std:.1f}, min {p_min}, max {p_max}")
    print(f"  RAG tokens    : mean {r_mean:.1f}, std {r_std:.1f}, min {r_min}, max {r_max}")
    print(f"  Output tokens : mean {o_mean:.1f}, std {o_std:.1f}, min {o_min}, max {o_max}")
    print(f"                   p50 {o_pct['p50']:.1f}, p90 {o_pct['p90']:.1f}, p95 {o_pct['p95']:.1f}, p99 {o_pct['p99']:.1f}")
    print(f"  Max-token hits (>= {args.max_output_tokens}): {max_token_hits} ({max_token_rate:.2%})")
    print()

    print("Output length by outcome (tokens):")
    for cls in ("tp", "fp", "tn", "fn"):
        vals = by_class_tokens[cls]
        m, s, mn, mx = summarize_tokens(vals)
        print(f"  {cls.upper():2s}: n={len(vals):3d}, mean {m:.1f}, std {s:.1f}, min {mn}, max {mx}")
    print()

    print("Per-test-case accuracy (parsed only):")
    for case, stats in sorted(per_case.items()):
        correct = stats["correct"]
        total_case = stats["total"]
        acc_case = correct / total_case if total_case else 0.0
        print(f"  {case}: {correct}/{total_case} ({acc_case:.2%})")

    # Grouped summary if requested
    if args.group_by:
        grouping: Dict[str, Dict[str, int]] = {}
        for entry in all_entries:
            key = entry.get(args.group_by, f"<missing {args.group_by}>")
            grouping.setdefault(key, {"correct": 0, "total": 0})
            sanitized = entry.get("sanitized_output")
            expected = entry.get("expected_output")
            if isinstance(sanitized, bool) and isinstance(expected, bool):
                grouping[key]["total"] += 1
                if sanitized == expected:
                    grouping[key]["correct"] += 1
        print()
        print(f"Per-{args.group_by} accuracy (parsed only):")
        for key, stats in sorted(grouping.items()):
            total_g = stats["total"]
            acc_g = stats["correct"] / total_g if total_g else 0.0
            print(f"  {key}: {stats['correct']}/{total_g} ({acc_g:.2%})")

    # Optional exports
    if args.csv_out:
        write_csv(flat_rows, Path(args.csv_out))
        print(f"Wrote per-entry CSV to {args.csv_out}")
    if args.json_out:
        payload = {
            "files_analyzed": len(files),
            "total_samples": total,
            "parsed": parsed,
            "unparsed": confusion["unparsed"],
            "structure_rate": structure_rate,
            "confusion": confusion,
            "accuracy_parsed": accuracy,
            "recall": recall,
            "specificity": specificity,
            "precision": precision,
            "tokens": {
                "prompt": {"mean": p_mean, "std": p_std, "min": p_min, "max": p_max},
                "rag": {"mean": r_mean, "std": r_std, "min": r_min, "max": r_max},
                "output": {"mean": o_mean, "std": o_std, "min": o_min, "max": o_max, "percentiles": o_pct},
                "max_token_hits": max_token_hits,
                "max_token_rate": max_token_rate,
            },
            "per_case": {case: {**stats, "accuracy": (stats["correct"] / stats["total"] if stats["total"] else 0.0)}
                         for case, stats in per_case.items()},
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"Wrote aggregate JSON to {args.json_out}")


if __name__ == "__main__":
    main()
