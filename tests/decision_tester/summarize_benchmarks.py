#!/usr/bin/env python3
"""
Summarize decision tester logs into a table for Benchmarks.xlsx.

It scans *_samples.json files under tests/decision_tester/logs/report_logs,
computes per-run metrics, and optionally emits grouped averages per
(Model, RAG, Device, Quantized, Binary). Defaults assume you run from the
repo root.

Example:
    python scripts/summarize_benchmarks.py \
        --logs logs/report_logs \
        --csv-runs benchmarks_runs.csv \
        --csv-agg benchmarks_agg.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

MODEL_PARAM_MAP = {
    "llama3-2": "3.21 B",
    "phi3": "3.80 B",
    "qwen2-5-7b": "7.61 B",
    "qwen2-5-3b": "3.09 B",
}


def infer_rag_label(entry: Dict) -> str:
    rag_mode = (entry.get("rag_mode") or "").lower()
    if rag_mode == "online":
        return "GPT-4o"
    if rag_mode == "offline":
        return "BAAI"
    return "None"


def infer_device(model_name: str, run_dir: Path) -> str:
    blob = f"{model_name} {' '.join(run_dir.parts)}".lower()
    if "gguf" in blob:
        return "GGUF"
    if "local_" in blob or "axelera" in blob:
        return "Axelera"
    return "GPU"


def infer_quantized(model_name: str, run_dir: Path) -> str:
    blob = f"{model_name} {' '.join(run_dir.parts)}".lower()
    quant_markers = ("gguf", "q4", "q5", "int4", "int8", "quant")
    return "Yes" if any(marker in blob for marker in quant_markers) else "No"


def infer_binary(run_dir: Path) -> str:
    return "Yes" if any("binary" in part.lower() for part in run_dir.parts) else "No"


def load_entries(files: Iterable[Path]) -> List[Dict]:
    entries: List[Dict] = []
    for fp in files:
        with fp.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def summarize_run(run_dir: Path) -> Optional[Dict]:
    json_files = sorted(run_dir.glob("*_samples.json"))
    entries = load_entries(json_files)
    if not entries:
        return None

    parsed = [
        e for e in entries
        if isinstance(e.get("sanitized_output"), bool) and isinstance(e.get("expected_output"), bool)
    ]

    # Accuracy where unparsed = incorrect
    total = len(entries)
    per_case_totals: Dict[str, int] = {}
    per_case_correct_all: Dict[str, int] = {}

    # Accuracy on parsed only
    correct_flags = []
    per_case_parsed: Dict[str, List[bool]] = {}

    for e in entries:
        case = e.get("test_case", "<unknown>")
        per_case_totals[case] = per_case_totals.get(case, 0) + 1

        parsed_ok = (
            isinstance(e.get("sanitized_output"), bool)
            and isinstance(e.get("expected_output"), bool)
        )
        if parsed_ok:
            is_correct = e["sanitized_output"] == e["expected_output"]
            correct_flags.append(is_correct)
            per_case_parsed.setdefault(case, []).append(is_correct)
            if is_correct:
                per_case_correct_all[case] = per_case_correct_all.get(case, 0) + 1
        else:
            # unparsed counts as incorrect for the "all" denominator
            per_case_correct_all[case] = per_case_correct_all.get(case, 0)

    parsed_count = len(correct_flags)
    acc_micro_parsed = (sum(correct_flags) / parsed_count) if parsed_count else 0.0
    acc_macro_parsed = (
        sum(sum(v) / len(v) for v in per_case_parsed.values()) / len(per_case_parsed)
        if per_case_parsed else 0.0
    )

    acc_micro_all = (sum(per_case_correct_all.values()) / total) if total else 0.0
    acc_macro_all = (
        sum(per_case_correct_all[c] / per_case_totals[c] for c in per_case_totals) / len(per_case_totals)
        if per_case_totals else 0.0
    )

    structure_rate = (len(parsed) / len(entries)) if entries else 0.0

    output_tokens_all = [int(e.get("output_tokens", 0)) for e in entries]
    output_tokens_parsed = [int(e.get("output_tokens", 0)) for e in parsed]

    first = entries[0]
    # Model label: first segment of parent folder, split by underscore (e.g., Llama3-2_axelera_default -> Llama3-2)
    model_name = run_dir.parent.name.split("_", 1)[0]
    model_params = next(
        (v for k, v in MODEL_PARAM_MAP.items() if model_name.lower().startswith(k)),
        "--",
    )

    return {
        "Model": model_name,
        "RAG": infer_rag_label(first),
        "Device": infer_device(model_name, run_dir),
        "Quantized": infer_quantized(model_name, run_dir),
        "Binary": infer_binary(run_dir),
        "Model Params": model_params,
        "Accuracy micro parsed (%)": round(acc_micro_parsed * 100, 2),
        "Accuracy macro parsed (%)": round(acc_macro_parsed * 100, 2),
        "Accuracy micro all (%)": round(acc_micro_all * 100, 2),
        "Accuracy macro all (%)": round(acc_macro_all * 100, 2),
        "Structure followed (%)": round(structure_rate * 100, 2),
        "Output tokens": round(mean(output_tokens_all), 2) if output_tokens_all else 0.0,
        "Output tokens (filt)": round(mean(output_tokens_parsed), 2) if output_tokens_parsed else 0.0,
        "Run": str(run_dir),
    }


def aggregate_rows(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[tuple, List[Dict]] = {}
    for row in rows:
        key = (row["Model"], row["RAG"], row["Device"], row["Quantized"], row["Binary"])
        grouped.setdefault(key, []).append(row)

    agg_rows: List[Dict] = []
    for key, items in sorted(grouped.items()):
        agg_rows.append({
            "Model": key[0],
            "RAG": key[1],
            "Device": key[2],
            "Quantized": key[3],
            "Binary": key[4],
            "Model Params": items[0].get("Model Params", "--"),
            "Runs": len(items),
            "Accuracy micro parsed (%)": round(mean(i["Accuracy micro parsed (%)"] for i in items), 2),
            "Accuracy macro parsed (%)": round(mean(i["Accuracy macro parsed (%)"] for i in items), 2),
            "Accuracy micro all (%)": round(mean(i["Accuracy micro all (%)"] for i in items), 2),
            "Accuracy macro all (%)": round(mean(i["Accuracy macro all (%)"] for i in items), 2),
            "Structure followed (%)": round(mean(i["Structure followed (%)"] for i in items), 2),
            "Output tokens": round(mean(i["Output tokens"] for i in items), 2),
            "Output tokens (filt)": round(mean(i["Output tokens (filt)"] for i in items), 2),
        })
    return agg_rows


def latex_escape(text: str) -> str:
    return (
        text.replace("_", r"\_")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("#", r"\#")
    )


def format_latex_table(rows: List[Dict], caption: str, label: str) -> str:
    """
    Build a LaTeX sidewaystable with the key metrics.
    Uses Accuracy macro all (%) as the overall accuracy.
    """
    if not rows:
        return "% No rows to render\n"

    # Precompute widths for padded "avg | parsed" cells to keep the '|' visually aligned
    acc_left_w = max(len(f"{r['Accuracy macro all (%)']:.2f}") for r in rows)
    acc_right_w = max(len(f"{r['Accuracy macro parsed (%)']:.2f}") for r in rows)
    tok_left_w = max(len(f"{r['Output tokens']:.0f}") for r in rows)
    tok_right_w = max(len(f"{r['Output tokens (filt)']:.0f}") for r in rows)

    lines = []
    lines.append(r"\section{Summary of results}")
    lines.append(r"\begin{sidewaystable}[h]")
    lines.append(r"  \centering")
    lines.append(r"  \setlength{\tabcolsep}{6pt}")
    lines.append(r"  \renewcommand{\arraystretch}{1.2}")
    lines.append(r"  \begin{tabular}{lcccccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    \multicolumn{9}{c}{\cellcolor{ggufColor} \textbf{Accuracy metrics}} \\")
    lines.append(r"    \midrule")
    lines.append(r"    \makecell{\textbf{Model}\\\textbf{name}} &")
    lines.append(r"    \makecell{\textbf{Model}\\\textbf{Params}} \\")
    lines.append(r"    \makecell{\textbf{RAG}\\\textbf{type}} &")
    lines.append(r"    \makecell{\textbf{Device}\\\textbf{used}} &")
    lines.append(r"    \makecell{\textbf{Quant}\\\textbf{scheme}} &")
    lines.append(r"    \makecell{\textbf{Binary}\\\textbf{output}} &")
    lines.append(r"    \makecell{\textbf{Structure}\\\textbf{followed (\%)}} &")
    lines.append(r"    \makecell{\textbf{Accuracy}\\\textbf{(avg. | parsed)}} &")
    lines.append(r"    \makecell{\textbf{Output tokens}\\\textbf{(avg. | parsed)}} &")
    lines.append(r"    \midrule")

    for row in rows:
        rag = row["RAG"]
        rag_tex = r"\xmark" if rag == "None" else latex_escape(rag)
        # Quant scheme: GGUF -> Q4.M, Axelera -> INT8, else FP16 unless marked quantized
        if row["Device"] == "GGUF":
            quant_scheme = "Q4.M"
        elif row["Device"] == "Axelera":
            quant_scheme = "INT8"
        elif row["Quantized"] == "Yes":
            quant_scheme = "Quantized"
        else:
            quant_scheme = "FP16"

        binary_tex = r"\cmark" if row["Binary"] == "Yes" else r"\xmark"
        structure = f"{row['Structure followed (%)']:.2f}\\%"
        acc_l = f"{row['Accuracy macro all (%)']:.2f}"
        acc_r = f"{row['Accuracy macro parsed (%)']:.2f}"
        acc_pad_l = r"\hphantom{" + "0" * (acc_left_w - len(acc_l)) + "}"
        acc_pad_r = r"\hphantom{" + "0" * (acc_right_w - len(acc_r)) + "}"
        accuracy = f"{acc_pad_l}{acc_l} | {acc_r}{acc_pad_r}\\%"

        tok_l = f"{row['Output tokens']:.0f}"
        tok_r = f"{row['Output tokens (filt)']:.0f}"
        tok_pad_l = r"\hphantom{" + "0" * (tok_left_w - len(tok_l)) + "}"
        tok_pad_r = r"\hphantom{" + "0" * (tok_right_w - len(tok_r)) + "}"
        out_tok = f"{tok_pad_l}{tok_l} | {tok_r}{tok_pad_r}"
        model_params = row.get("Model Params", "--")

        lines.append(
            "    "
            + " & ".join(
                [
                    latex_escape(str(row["Model"])),
                    latex_escape(str(model_params)),
                    rag_tex,
                    latex_escape(str(row["Device"])),
                    latex_escape(quant_scheme),
                    binary_tex,
                    structure,
                    accuracy,
                    out_tok,
                ]
            )
            + r" \\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"\end{sidewaystable}")
    return "\n".join(lines) + "\n"


def print_table(rows: List[Dict], title: str) -> None:
    if not rows:
        print(f"{title}: no rows")
        return

    headers = list(rows[0].keys())
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

    print(f"\n{title}")
    print("-" * sum(widths.values()))
    header_line = " | ".join(f"{h:{widths[h]}}" for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(" | ".join(f"{str(row[h]):{widths[h]}}" for h in headers))


def write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize decision tester logs for Benchmarks.xlsx.")
    parser.add_argument("--logs", type=Path,
                        default=Path("logs/report_logs"),
                        help="Directory containing per-run logs.")
    parser.add_argument("--csv-runs", type=Path, default=None,
                        help="Optional path to write per-run metrics as CSV.")
    parser.add_argument("--csv-agg", type=Path, default=None,
                        help="Optional path to write grouped averages as CSV.")
    parser.add_argument("--latex-out", type=Path, default=None,
                        help="Optional path to write LaTeX table (uses aggregated rows).")
    parser.add_argument("--latex-caption", type=str, default="Accuracy metrics summary",
                        help="Caption for the LaTeX table.")
    parser.add_argument("--latex-label", type=str, default="tab:accuracy-metrics",
                        help="Label for the LaTeX table.")
    args = parser.parse_args()

    if not args.logs.exists():
        raise SystemExit(f"Log directory not found: {args.logs}")

    run_dirs = sorted({fp.parent for fp in args.logs.rglob("*_samples.json")})
    rows = [row for rd in run_dirs if (row := summarize_run(rd))]

    if not rows:
        raise SystemExit("No *_samples.json files found to summarize.")

    print_table(rows, "Per-run metrics")
    agg_rows = aggregate_rows(rows)
    print_table(agg_rows, "Grouped averages (by model/RAG/device/quant/binary)")

    if args.csv_runs:
        write_csv(rows, args.csv_runs)
        print(f"\nWrote per-run CSV to {args.csv_runs}")
    if args.csv_agg:
        write_csv(agg_rows, args.csv_agg)
        print(f"Wrote grouped CSV to {args.csv_agg}")
    if args.latex_out:
        latex = format_latex_table(
            agg_rows,
            caption=args.latex_caption,
            label=args.latex_label,
        )
        args.latex_out.parent.mkdir(parents=True, exist_ok=True)
        args.latex_out.write_text(latex)
        print(f"Wrote LaTeX table to {args.latex_out}")


if __name__ == "__main__":
    main()
