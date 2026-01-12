import argparse
import json
import os
import signal
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="GGUF decode-only benchmark (reads prompts JSONL, runs llama-cpp-python).")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing a .gguf file.")
    parser.add_argument("--gguf_name", type=str, default="", help="Optional .gguf filename (auto-detect if omitted).")
    parser.add_argument("--prompts", type=str, required=True, help="JSONL file with {'prompt': ...} lines.")
    parser.add_argument("--limit", type=int, default=0, help="Max prompts to run (0 = all).")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup prompts to run (default: 1).")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--n_ctx", type=int, default=4096)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--chat_format", type=str, default="chatml")
    parser.add_argument("--binary-output", action="store_true", help="Enable binary output post-processing (e.g., adherence marker truncation).")
    parser.add_argument("--hard-exit", action="store_true", help="Use os._exit at the end (recommended on Jetson).")
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

    gguf_name = args.gguf_name
    if not gguf_name:
        ggufs = [f for f in os.listdir(args.model_dir) if f.endswith(".gguf")]
        if not ggufs:
            raise FileNotFoundError(f"No .gguf files found under {args.model_dir}")
        gguf_name = sorted(ggufs)[0]

    from inference.inf_gguf import RaceLLMGGGUF

    stop_flag = {"stop": False}

    def _handle_sigint(_signum, _frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    print(f"[bench] model_dir={args.model_dir} gguf={gguf_name} prompts={len(prompts)}", flush=True)
    print(f"[bench] warmup={args.warmup} max_tokens={args.max_tokens} n_ctx={args.n_ctx} n_gpu_layers={args.n_gpu_layers}", flush=True)

    t0 = time.time()
    llm = RaceLLMGGGUF(
        model_dir=args.model_dir,
        gguf_name=gguf_name,
        chat_format=args.chat_format,
        max_tokens=args.max_tokens,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        binary_output=bool(args.binary_output),
    )

    def _run_one(idx: int, prompt: str):
        if stop_flag["stop"]:
            return False
        _out_text, in_toks, out_toks = llm(prompt)
        dt_s = time.time() - t0
        print(f"[bench] i={idx} dt_s={dt_s:.3f} prompt_toks={in_toks} out_toks={out_toks}", flush=True)
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

    return 130 if stop_flag["stop"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
