#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Optional

from llama_cpp import Llama


def infer_chat_template(model_id: str) -> str:
    mid = model_id.lower()
    if "phi-3" in mid or "phi3" in mid:
        # llama.cpp does not register a "phi-3" chat handler; chatml works for Phi-3
        return "chatml"
    if "qwen" in mid:
        return "qwen"
    if "llama-3.2" in mid or "llama-3.1" in mid or "llama-3" in mid:
        return "llama-3"
    return "llama-3"


def normalize_chat_template(chat_template: str) -> str:
    ct = (chat_template or "").lower()
    if ct in ("phi-3", "phi3"):
        return "chatml"
    if ct in ("qwen-2.5", "qwen2.5", "qwen2", "qwen"):
        return "qwen"
    if ct in ("llama-3.2", "llama-3.1", "llama-3", "llama3"):
        return "llama-3"
    return ct or "llama-3"


def resolve_model_dir(model_arg: Optional[str]) -> Path:
    """
    Prefer an explicit path, otherwise pick the first folder under models/.
    """
    base_models = Path("models")

    if model_arg:
        candidate = Path(model_arg)
        if not candidate.is_absolute():
            candidate_in_models = base_models / candidate
            if candidate_in_models.exists():
                candidate = candidate_in_models
        if not candidate.exists():
            raise FileNotFoundError(f"Model directory '{model_arg}' not found.")
        if not candidate.is_dir():
            raise NotADirectoryError(f"Model path '{candidate}' is not a directory.")
        return candidate

    if base_models.exists():
        folders = sorted(p for p in base_models.iterdir() if p.is_dir())
        if folders:
            return folders[0]

    raise FileNotFoundError(
        "No model directory provided and nothing found under models/."
    )


def find_gguf_file(model_dir: Path, preferred: Optional[str]) -> Path:
    if preferred:
        candidate = model_dir / preferred
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"GGUF file '{preferred}' not found in {model_dir}"
        )

    gguf_files = sorted(model_dir.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf file found in {model_dir}")
    return gguf_files[0]


def build_llm(model_path: Path, chat_template: str, args: argparse.Namespace) -> Llama:
    return Llama(
        model_path=str(model_path),
        chat_format=chat_template,
        n_ctx=args.ctx_size,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
        verbose=args.verbose,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal GGUF chat tester (no pipeline). Run from src/LLMxRobot."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model directory (e.g., models/race_llm_q5). Defaults to first folder under models/.",
    )
    parser.add_argument(
        "--gguf",
        type=str,
        default=None,
        help="Specific GGUF filename inside the model directory. Defaults to the first *.gguf.",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="Override chat template (default inferred from model name).",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="",
        help="Optional system prompt to prepend to every turn.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per turn.",
    )
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=4096,
        help="Context length to allocate for the model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling.",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="GPU layers to offload (-1 = all available).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PRNG seed for llama.cpp.",
    )
    parser.add_argument(
        "--keep-history",
        action="store_true",
        help="Keep the chat history between turns.",
    )
    parser.add_argument(
        "--show-usage",
        action="store_true",
        help="Print prompt/completion token counts after each reply.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable llama.cpp verbose logging.",
    )

    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model)
    gguf_path = find_gguf_file(model_dir, args.gguf)
    chat_template = normalize_chat_template(args.chat_template or infer_chat_template(model_dir.name))

    llm = build_llm(gguf_path, chat_template, args)

    base_messages: List[dict] = []
    if args.system:
        base_messages.append({"role": "system", "content": args.system})
    history: List[dict] = list(base_messages)

    print(f"Loaded {gguf_path.name} from {model_dir} (chat_template={chat_template})")
    if not args.keep_history:
        print("History disabled; each turn is independent.")
    print("Type 'exit' or Ctrl+D to quit.")

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        messages = list(history) if args.keep_history else list(base_messages)
        messages.append({"role": "user", "content": user_input})

        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        reply = response["choices"][0]["message"]["content"]
        print(f"Model> {reply}")

        if args.show_usage and "usage" in response:
            usage = response["usage"]
            prompt_toks = usage.get("prompt_tokens")
            completion_toks = usage.get("completion_tokens")
            print(f"(prompt tokens: {prompt_toks}, completion tokens: {completion_toks})")

        if args.keep_history:
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
