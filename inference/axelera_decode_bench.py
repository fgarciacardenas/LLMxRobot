import argparse
import json
import os
import signal
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Axelera decode-only benchmark (reads prompts JSONL, runs LocalLLMPipeline REPL)."
    )
    parser.add_argument("--prompts", type=str, required=True, help="JSONL file with {'prompt': ...} lines.")
    parser.add_argument("--limit", type=int, default=0, help="Max prompts to run (0 = all).")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup prompts to run (default: 1).")
    parser.add_argument("--workdir", type=str, default="voyager-sdk", help="Voyager SDK workdir (default: voyager-sdk).")
    parser.add_argument("--venv-activate", type=str, default="venv/bin/activate", help="Venv activate script path.")
    parser.add_argument(
        "--run-cmd",
        type=str,
        default="./inference_llm.py phi3-mini-2048-4core-static",
        help="Command that launches the interactive inference REPL.",
    )
    parser.add_argument("--prompt-timeout", type=int, default=360, help="Per-prompt timeout seconds (default: 360).")
    parser.add_argument("--verbose", action="store_true", help="Stream REPL output (pexpect logfile).")
    parser.add_argument("--hard-exit", action="store_true", help="Use os._exit at the end (recommended if shutdown hangs).")
    args = parser.parse_args()

    os.environ.setdefault("LLMXROBOT_PROFILE_LLM", "1")

    # Read prompts
    prompts = []
    with open(args.prompts, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p = rec.get("prompt")
            if not isinstance(p, str):
                continue
            prompts.append(p)
            if args.limit and len(prompts) >= args.limit:
                break
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompts}")

    from inference.local_pipeline import LocalLLMPipeline

    stop_flag = {"stop": False}

    def _handle_sigint(_signum, _frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    def _emit(event: str, payload: dict):
        if os.getenv("LLMXROBOT_PROFILE_LLM", "").strip().lower() not in ("1", "true", "yes", "on"):
            return
        p = dict(payload)
        p["event"] = event
        p["t_epoch_s"] = time.time()
        p["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p["t_epoch_s"]))
        print("LLMXROBOT_EVENT " + json.dumps(p, sort_keys=True), flush=True)

    print(f"[bench] prompts={len(prompts)} warmup={args.warmup} limit={args.limit}", flush=True)
    print(f"[bench] workdir={args.workdir} venv={args.venv_activate} run_cmd={args.run_cmd}", flush=True)

    llm = LocalLLMPipeline(
        workdir=args.workdir,
        venv_activate=args.venv_activate,
        run_cmd=args.run_cmd,
        prompt_timeout=args.prompt_timeout,
        verbose=bool(args.verbose),
    )

    t0 = time.time()

    def _run_one(idx: int, prompt: str) -> bool:
        if stop_flag["stop"]:
            return False
        _emit(
            "llm_decode_start",
            {
                "idx": idx,
                "prompt_chars": len(prompt),
            },
        )
        out_text, _in_toks, _out_toks = llm(prompt)
        _emit(
            "llm_decode_end",
            {
                "idx": idx,
                "completion_chars": len(out_text or ""),
            },
        )
        dt_s = time.time() - t0
        print(f"[bench] i={idx} dt_s={dt_s:.3f} prompt_chars={len(prompt)} completion_chars={len(out_text or '')}", flush=True)
        return True

    # Warmup on first prompt(s)
    for i in range(min(args.warmup, len(prompts))):
        if not _run_one(-1 - i, prompts[i]):
            break

    ran = 0
    for i, p in enumerate(prompts):
        if stop_flag["stop"]:
            break
        if args.limit and ran >= args.limit:
            break
        if not _run_one(i, p):
            break
        ran += 1

    print(f"[bench] done ran={ran} interrupted={stop_flag['stop']}", flush=True)

    if args.hard_exit or os.getenv("LLMXROBOT_HARD_EXIT", "").strip().lower() in ("1", "true", "yes", "on"):
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(130 if stop_flag["stop"] else 0)

    try:
        llm.close()
    except Exception:
        pass
    return 130 if stop_flag["stop"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

