## 📚 **Overview**

`DecisionTester` is an evaluation framework designed to assess the reasoning capabilities of an AI model embedded in an autonomous racing car. It evaluates the car's behavior against various test scenarios using predefined evaluation functions and logs the results for further analysis.

---

## 🚀 **Usage**

### **Run Evaluation on All Datasets**

```bash
python3 -m tests.decision_tester.decision_tester --model nibauman/RobotxLLM_Qwen7B_SFT --dataset all --mini --rag
```

### **Quick GGUF sanity check (no pipeline)**
Run from `src/LLMxRobot` to chat directly with a local `.gguf` file:
```bash
python3 -m tests.decision_tester.gguf_manual_chat \
  --model models/race_llm_q5 \
  --gguf race_llm-Q5_K_M.gguf \
  --chat-template qwen \
  --keep-history
```
`--model` can point to any folder listed under `models/`. `--gguf` is optional (defaults to the first `.gguf` in the folder), and `--chat-template` defaults to an inferred template matching the model name. For Phi-3 quantized GGUFs, use `--chat-template chatml` (or let the default inference pick it).

### **Available Arguments:**
- `--model`: Choose the model (`gpt-4o` or `nibauman/RobotxLLM_Qwen7B_SFT` or any model available from [unsloth](https://huggingface.co/unsloth)).  
- `--dataset`: Choose the dataset (`all`, `stop`, `reverse`, etc.).  
- `--mini`: Run a reduced dataset (`--mini` enables mini evaluation mode).  
- `--quant`: Use GGUF quantized model (`Q5`) for faster inference.
- `--rag`: Uses the RAG for hints.

---

## 📊 **Available Test Scenarios**

| Test Case              | Description                       |
|-------------------------|-----------------------------------|
| **Don't move**         | Ensures the car remains stationary. |
| **Reverse the car**    | Checks if the car moves backward.  |
| **Drive forward**      | Validates forward movement.       |
| **Oscillate!**         | Detects irregular lateral motion. |
| **Drive close to the left wall** | Measures wall proximity. |
| **Drive on the centerline** | Tests alignment with the centerline. |
| **Drive faster than 3 m/s** | Verifies speed threshold. |
| **Drive on the racing line** | Ensures adherence to the optimal path. |

---

## 📁 **Log Files**

- Logs are saved in the `logs` directory with timestamped filenames:
  ```
  tests/decision_tester/logs/{model}_{dataset}_{timestamp}.txt
  ```
- Includes:
  - **Case Accuracies:** Success rates per scenario.  
  - **Incorrect Entries:** Detailed logs of mismatches, including prompts and responses.

---

## 🛡️ **Environment Variables**

Ensure the following are set in your `.env` file if you want to use GPT4o but also needed for RAG embeddings:
```
OPENAI_API_TOKEN=your_openai_key
```

---

Happy Testing! 🚗💨


# Remote Axelera board testing
Command to use:
```bash
pip3 install pexpect
python3 -m tests.decision_tester.decision_tester \
  --dataset all --mini \
  --ssh_interactive \
  --ssh_host finsteraarhorn.ee.ethz.ch \
  --ssh_user sem25h27 \
  --ssh_workdir voyager-sdk \
  --ssh_venv venv/bin/activate \
  --ssh_run "./inference_llm.py llama-3-2-3b-1024-4core-static" \
  --ssh_timeout 360 --ssh_password <pw>
```

# Local Axelera board testing
```bash
python3 -m tests.decision_tester.decision_tester \
  --dataset all --mini \
  --ax_local \
  --local_workdir ~/voyager-sdk \
  --local_venv venv/bin/activate \
  --local_run "./inference_llm.py llama-3-2-3b-1024-4core-static" \
  --local_timeout 420 \
  --local_verbose
```


# Benchmark summary (decision tester logs)
Use `summarize_benchmarks.py` to aggregate decision tester logs and export CSV/LaTeX tables for the Benchmarks spreadsheet.

```bash
# From repo root; point to the decision tester logs
python3 src/LLMxRobot/tests/decision_tester/summarize_benchmarks.py \
  --logs src/LLMxRobot/tests/decision_tester/logs/report_logs \
  --csv-runs benchmarks_runs.csv \   # per-run metrics (optional)
  --csv-agg benchmarks_agg.csv \     # grouped averages (optional)
  --latex-out benchmarks_table.tex   # LaTeX sidewaystable (optional)
```

Notes:
- `--latex-caption` and `--latex-label` customize the LaTeX caption/label.
- Model label comes from the parent folder prefix (e.g., `Llama3-2_axelera_default` -> `Llama3-2`).
- Quantization column renders `Q4.M` for GGUF, `INT8` for Axelera, otherwise `FP16` (or `Quantized` when flagged).
- Model params are auto-filled for known prefixes: Llama3-2 (3.21 B), Phi3 (3.80 B), Qwen2-5-7B (7.61 B), Qwen2-5-3B (3.09 B); otherwise `--`.

## Config-driven LaTeX tables
To regenerate multiple tables with a specific folder order (and optional `\midrule` separators), use `--tables-config`:

```bash
python3 src/LLMxRobot/tests/decision_tester/summarize_benchmarks.py \
  --tables-config src/LLMxRobot/tests/decision_tester/tables_config.example.json
```

See `src/LLMxRobot/tests/decision_tester/tables_config.example.json` for the schema. The `items` list controls ordering; insert `"midrule"` where you want an extra `\midrule` in the table body.
