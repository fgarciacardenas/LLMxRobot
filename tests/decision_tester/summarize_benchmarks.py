#!/usr/bin/env python3
"""
Summarize decision tester logs into a table for Benchmarks.xlsx.

It scans *_samples.json files under tests/decision_tester/logs/report_logs,
computes per-run metrics, and optionally emits grouped averages per
(Model, RAG, Device, Quantized, Binary). Defaults assume you run from the
repo root.

Example:
    python src/LLMxRobot/tests/decision_tester/summarize_benchmarks.py \
        --logs logs/report_logs \
        --csv-runs benchmarks_runs.csv \
        --csv-agg benchmarks_agg.csv

Generate multiple LaTeX tables from a JSON config:
    python src/LLMxRobot/tests/decision_tester/summarize_benchmarks.py \
        --tables-config src/LLMxRobot/tests/decision_tester/tables_config.example.json
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

MODEL_PARAM_MAP = {
    "llama3-1": "8.03 B",
    "llama3-2": "3.21 B",
    "phi3": "3.80 B",
    "qwen2-5-7b": "7.61 B",
    "qwen2-5-3b": "3.09 B",
}

MODEL_DISPLAY_MAP = [
    ("llama3-1", "Llama3.1"),
    ("llama3-2", "Llama3.2"),
    ("phi3", "Phi3"),
    ("qwen2-5", "Qwen2.5"),
]


def map_model_name(extracted_model_name: str) -> str:
    extracted_lower = extracted_model_name.lower()
    mapped = next(
        (pretty for needle, pretty in MODEL_DISPLAY_MAP if needle in extracted_lower),
        extracted_model_name,
    )
    if "NI" in extracted_model_name and not mapped.endswith("-NI"):
        mapped = f"{mapped}-NI"
    if "2048" in extracted_model_name and not mapped.endswith("-2048"):
        mapped = f"{mapped}-2048"
    if "SFT" in extracted_model_name.upper() and not mapped.endswith("-SFT"):
        mapped = f"{mapped}-SFT"
    return mapped


def infer_rag_label(entry: Dict) -> str:
    rag_mode = (entry.get("rag_mode") or "").lower()
    if rag_mode == "online":
        return "OpenAI"
    if rag_mode == "offline":
        return "BGE"
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


def infer_quant_scheme(extracted_model_name: str, run_dir: Path, device: str, quantized: str) -> str:
    if device == "Axelera":
        return "INT8"
    if device == "GGUF":
        extracted_upper = extracted_model_name.upper()
        if "Q8" in extracted_upper:
            return "Q8.0"
        if "Q5" in extracted_upper:
            return "Q5.M"
        if "Q4" in extracted_upper:
            return "Q4.M"

        blob_upper = f"{' '.join(run_dir.parts)}".upper()
        if "Q8" in blob_upper:
            return "Q8.0"
        if "Q5" in blob_upper:
            return "Q5.M"
        if "Q4" in blob_upper:
            return "Q4.M"
        return "Q4.M"
    if quantized == "Yes":
        return "Quantized"
    return "FP16"


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
    extracted_model_name = run_dir.parent.name.split("_", 1)[0]
    model_name = map_model_name(extracted_model_name)
    device = infer_device(extracted_model_name, run_dir)
    quantized = infer_quantized(extracted_model_name, run_dir)
    quant_scheme = infer_quant_scheme(extracted_model_name, run_dir, device=device, quantized=quantized)
    model_params = next(
        (v for k, v in MODEL_PARAM_MAP.items() if extracted_model_name.lower().startswith(k)),
        "--",
    )

    return {
        "Model": model_name,
        "RAG": infer_rag_label(first),
        "Device": device,
        "Quantized": quantized,
        "Binary": infer_binary(run_dir),
        "__Quant scheme": quant_scheme,
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
        quant_schemes = {i.get("__Quant scheme") for i in items}
        quant_schemes.discard(None)
        quant_scheme = next(iter(quant_schemes)) if len(quant_schemes) == 1 else None
        agg_rows.append({
            "Model": key[0],
            "RAG": key[1],
            "Device": key[2],
            "Quantized": key[3],
            "Binary": key[4],
            "__Quant scheme": quant_scheme,
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

_MIDRULE: Dict[str, bool] = {"__midrule__": True}

_LATEX_COLUMN_ALIASES: Dict[str, str] = {
    "model": "model",
    "model name": "model",
    "model_params": "model_params",
    "model params": "model_params",
    "rag": "rag",
    "rag type": "rag",
    "device": "device",
    "device used": "device",
    "quant_scheme": "quant_scheme",
    "quant scheme": "quant_scheme",
    "quantization scheme": "quant_scheme",
    "binary": "binary",
    "binary output": "binary",
    "structure": "structure",
    "structure followed (%)": "structure",
    "accuracy": "accuracy",
    "accuracy (avg. | par.)": "accuracy",
    "output_tokens": "output_tokens",
    "output tokens": "output_tokens",
    "output tokens (avg. | par.)": "output_tokens",
}

_DEFAULT_LATEX_COLUMNS: List[str] = [
    "model",
    "model_params",
    "rag",
    "device",
    "quant_scheme",
    "binary",
    "structure",
    "accuracy",
    "output_tokens",
]


def _normalize_latex_columns(columns: Optional[List[Any]]) -> List[str]:
    if not columns:
        return list(_DEFAULT_LATEX_COLUMNS)
    if not isinstance(columns, list):
        raise SystemExit("'columns' must be a list.")

    normalized: List[str] = []
    for col in columns:
        if not isinstance(col, str):
            raise SystemExit(f"Invalid column (must be string): {col!r}")
        key = _LATEX_COLUMN_ALIASES.get(col.strip().lower())
        if not key:
            allowed = ", ".join(sorted(set(_LATEX_COLUMN_ALIASES.values())))
            raise SystemExit(f"Unknown column '{col}'. Allowed: {allowed}")
        normalized.append(key)

    # de-dup while preserving order
    seen = set()
    deduped = []
    for key in normalized:
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _normalize_latex_env(env: Optional[Any]) -> str:
    if not env:
        return "sidewaystable"
    if not isinstance(env, str):
        raise SystemExit("'env' must be a string.")
    env = env.strip()
    if env not in {"table", "sidewaystable"}:
        raise SystemExit("'env' must be either 'table' or 'sidewaystable'.")
    return env


def _latex_column_spec(columns: List[str]) -> str:
    align = {
        "model": "l",
        "model_params": "c",
        "rag": "c",
        "device": "c",
        "quant_scheme": "c",
        "binary": "c",
        "structure": "c",
        "accuracy": "c",
        "output_tokens": "c",
    }
    return "".join(align[c] for c in columns)


def _latex_column_headers(columns: List[str]) -> List[str]:
    headers = {
        "model": r"\makecell{\textbf{Model}\\\textbf{name}}",
        "model_params": r"\makecell{\textbf{Model}\\\textbf{Params}}",
        "rag": r"\makecell{\textbf{RAG}\\\textbf{type}}",
        "device": r"\makecell{\textbf{Device}\\\textbf{used}}",
        "quant_scheme": r"\makecell{\textbf{Quant}\\\textbf{scheme}}",
        "binary": r"\makecell{\textbf{Binary}\\\textbf{output}}",
        "structure": r"\makecell{\textbf{Struct}\\\textbf{(\%)}}",
        "accuracy": r"\makecell{\textbf{Accuracy}\\\textbf{(avg. | par.)}}",
        "output_tokens": r"\makecell{\textbf{Tokens}\\\textbf{(avg. | par.)}}",
    }
    return [headers[c] for c in columns]


def format_latex_table(
    rows_or_markers: List[Dict],
    caption: str,
    label: str,
    *,
    columns: Optional[List[Any]] = None,
    env: Optional[Any] = None,
) -> str:
    """
    Build a LaTeX (sideways)table with the key metrics.
    Uses Accuracy macro all (%) as the overall accuracy.
    """
    columns_norm = _normalize_latex_columns(columns)
    env_norm = _normalize_latex_env(env)

    rows = [r for r in rows_or_markers if not r.get("__midrule__")]
    if not rows:
        return "% No rows to render\n"

    # Precompute widths for padded "avg | parsed" cells to keep the '|' visually aligned
    acc_left_w = (
        max(len(f"{r['Accuracy macro all (%)']:.2f}") for r in rows)
        if "accuracy" in columns_norm
        else 0
    )
    acc_right_w = (
        max(len(f"{r['Accuracy macro parsed (%)']:.2f}") for r in rows)
        if "accuracy" in columns_norm
        else 0
    )
    tok_left_w = (
        max(len(f"{r['Output tokens']:.0f}") for r in rows)
        if "output_tokens" in columns_norm
        else 0
    )
    tok_right_w = (
        max(len(f"{r['Output tokens (filt)']:.0f}") for r in rows)
        if "output_tokens" in columns_norm
        else 0
    )

    lines = []
    lines.append(rf"\begin{{{env_norm}}}")
    lines.append(r"  \centering")
    lines.append(r"  \setlength{\tabcolsep}{6pt}")
    lines.append(r"  \renewcommand{\arraystretch}{1.2}")
    lines.append(rf"  \begin{{tabular}}{{{_latex_column_spec(columns_norm)}}}")
    lines.append(r"    \toprule")
    lines.append(
        rf"    \multicolumn{{{len(columns_norm)}}}{{c}}{{\cellcolor{{ggufColor}} \textbf{{Accuracy metrics}}}} \\"
    )
    lines.append(r"    \midrule")
    lines.append("    " + " & ".join(_latex_column_headers(columns_norm)) + r" \\")
    lines.append(r"    \midrule")

    for row in rows_or_markers:
        if row.get("__midrule__"):
            lines.append(r"    \midrule")
            continue

        rag = row["RAG"]
        rag_tex = r"\xmark" if rag == "None" else latex_escape(rag)
        quant_scheme = (
            row.get("__Quant scheme")
            or (
                "Q4.M"
                if row["Device"] == "GGUF"
                else "INT8"
                if row["Device"] == "Axelera"
                else "Quantized"
                if row["Quantized"] == "Yes"
                else "FP16"
            )
        )

        binary_tex = r"\cmark" if row["Binary"] == "Yes" else r"\xmark"
        structure = f"{row['Structure followed (%)']:.2f}\\%"
        accuracy = ""
        if "accuracy" in columns_norm:
            acc_l = f"{row['Accuracy macro all (%)']:.2f}"
            acc_r = f"{row['Accuracy macro parsed (%)']:.2f}"
            acc_pad_l = r"\hphantom{" + "0" * (acc_left_w - len(acc_l)) + "}"
            acc_pad_r = r"\hphantom{" + "0" * (acc_right_w - len(acc_r)) + "}"
            accuracy = f"{acc_pad_l}{acc_l} | {acc_r}{acc_pad_r}\\%"

        out_tok = ""
        if "output_tokens" in columns_norm:
            tok_l = f"{row['Output tokens']:.0f}"
            tok_r = f"{row['Output tokens (filt)']:.0f}"
            tok_pad_l = r"\hphantom{" + "0" * (tok_left_w - len(tok_l)) + "}"
            tok_pad_r = r"\hphantom{" + "0" * (tok_right_w - len(tok_r)) + "}"
            out_tok = f"{tok_pad_l}{tok_l} | {tok_r}{tok_pad_r}"
        model_params = row.get("Model Params", "--")

        cell_map = {
            "model": latex_escape(str(row["Model"])),
            "model_params": latex_escape(str(model_params)),
            "rag": rag_tex,
            "device": latex_escape(str(row["Device"])),
            "quant_scheme": latex_escape(str(quant_scheme)),
            "binary": binary_tex,
            "structure": structure,
            "accuracy": accuracy,
            "output_tokens": out_tok,
        }

        lines.append(
            "    "
            + " & ".join(cell_map[c] for c in columns_norm)
            + r" \\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(rf"\end{{{env_norm}}}")
    return "\n".join(lines) + "\n"


def print_table(rows: List[Dict], title: str) -> None:
    if not rows:
        print(f"{title}: no rows")
        return

    headers = [k for k in rows[0].keys() if not str(k).startswith("__")]
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
    headers = [k for k in rows[0].keys() if not str(k).startswith("__")]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sort_agg_rows(rows: List[Dict]) -> List[Dict]:
    rag_order = {"None": 0, "OpenAI": 1, "BGE": 2}
    device_order = {"GPU": 0, "Axelera": 1, "GGUF": 2}
    binary_order = {"No": 0, "Yes": 1}
    quant_order = {"No": 0, "Yes": 1}

    def key(r: Dict) -> Tuple:
        return (
            str(r.get("Model", "")),
            rag_order.get(r.get("RAG"), 99),
            device_order.get(r.get("Device"), 99),
            quant_order.get(r.get("Quantized"), 99),
            binary_order.get(r.get("Binary"), 99),
        )

    return sorted(rows, key=key)


def _row_matches_selector(row: Dict, selector: Dict[str, Any]) -> bool:
    """
    Selector keys (all optional):
      - parent: str or [str] exact match on run directory parent name (folder under report_logs)
      - path_contains: [str] case-insensitive substrings required in the run path
      - path_glob: str or [str] fnmatch glob(s) matched against the run path
    """
    run_path = Path(row.get("Run", ""))
    parent_name = run_path.parent.name
    run_str = run_path.as_posix()
    run_low = run_str.lower()

    if "parent" in selector:
        allowed = selector["parent"]
        if isinstance(allowed, str):
            allowed = [allowed]
        if parent_name not in set(allowed):
            return False

    if "path_contains" in selector:
        subs = selector["path_contains"] or []
        if not all(str(s).lower() in run_low for s in subs):
            return False

    if "path_glob" in selector:
        pats = selector["path_glob"]
        if isinstance(pats, str):
            pats = [pats]
        if not any(fnmatch.fnmatch(run_str, pat) for pat in pats):
            return False

    return True


def generate_tables_from_config(config_path: Path) -> None:
    """
    Generate multiple LaTeX tables from a JSON config file.

    Top-level keys:
      - logs_root: base folder to scan (default: logs/report_logs)
      - out_dir: base output folder (default: .)
      - default_env: "table" or "sidewaystable" (default: "sidewaystable")
      - default_columns: list of column ids/aliases (default: all)
      - tables: list of table specs

    Table spec keys:
      - name: identifier (optional)
      - out: output file path (required; relative to out_dir unless absolute)
      - caption: LaTeX caption (optional)
      - label: LaTeX label (optional)
      - env: "table" or "sidewaystable" (optional)
      - columns: list of column ids/aliases (optional)
      - items: ordered list; each item is one of:
          - "midrule" or {"midrule": true}
          - "<folder_name>" (treated as {"parent": "<folder_name>"})
          - {"select": {...}} where select supports parent/path_contains/path_glob
          - {...} selector directly (parent/path_contains/path_glob)
    """
    cfg = json.loads(config_path.read_text())
    logs_root = Path(cfg.get("logs_root", "logs/report_logs"))
    out_dir = Path(cfg.get("out_dir", "."))
    default_env = cfg.get("default_env", "sidewaystable")
    default_columns = cfg.get("default_columns", None)
    tables = cfg.get("tables", [])
    if not isinstance(tables, list) or not tables:
        raise SystemExit("Config must contain a non-empty 'tables' list.")

    if not logs_root.exists():
        raise SystemExit(f"Log directory not found: {logs_root}")

    run_dirs = sorted({fp.parent for fp in logs_root.rglob("*_samples.json")})
    run_rows = [row for rd in run_dirs if (row := summarize_run(rd))]

    for table in tables:
        name = table.get("name", "table")
        out = table.get("out")
        if not out:
            raise SystemExit(f"Table '{name}' is missing required field 'out'.")
        out_path = Path(out)
        if not out_path.is_absolute():
            out_path = out_dir / out_path

        caption = table.get("caption", "Accuracy metrics summary")
        label = table.get("label", f"tab:{name}")
        env = table.get("env", default_env)
        columns = table.get("columns", default_columns)

        items = table.get("items", [])
        if not isinstance(items, list) or not items:
            raise SystemExit(f"Table '{name}' must have a non-empty 'items' list.")

        rendered: List[Dict] = []
        for item in items:
            if item == "midrule" or (isinstance(item, dict) and item.get("midrule") is True):
                if rendered and not rendered[-1].get("__midrule__"):
                    rendered.append(_MIDRULE)
                continue

            if isinstance(item, str):
                selector: Dict[str, Any] = {"parent": item}
            elif isinstance(item, dict) and "select" in item:
                selector = item.get("select") or {}
            elif isinstance(item, dict):
                selector = item
            else:
                raise SystemExit(f"Table '{name}': unsupported item: {item!r}")

            filtered = [r for r in run_rows if _row_matches_selector(r, selector)]
            agg = _sort_agg_rows(aggregate_rows(filtered))
            rendered.extend(agg)

        if rendered and rendered[-1].get("__midrule__"):
            rendered.pop()

        latex = format_latex_table(rendered, caption=caption, label=label, columns=columns, env=env)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(latex)
        print(f"Wrote LaTeX table '{name}' to {out_path}")


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
    parser.add_argument("--tables-config", type=Path, default=None,
                        help="JSON config describing multiple LaTeX tables to generate.")
    args = parser.parse_args()

    if args.tables_config:
        generate_tables_from_config(args.tables_config)
        return

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
