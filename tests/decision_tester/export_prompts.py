import argparse
import json
import os

from dotenv import load_dotenv, find_dotenv

from tests.decision_tester.decision_tester import DecisionTester, infer_chat_template_from_model


def main():
    parser = argparse.ArgumentParser(description="Export DecisionTester prompts to JSONL (no LLM inference).")
    parser.add_argument("--model", type=str, required=True, help="Model identifier (used for tokenizer/prompt formatting).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., centerline) or path to .json.")
    parser.add_argument("--mini", action="store_true", help="Downsample dataset (same behavior as DecisionTester --mini).")

    # RAG args (match DecisionTester)
    parser.add_argument("--rag", action="store_true", help="Whether to include RAG in the constructed prompt.")
    parser.add_argument("--rag_offline", action="store_true", help="Use offline embeddings for RAG.")
    parser.add_argument("--rag_index", type=str, default="", help="Path prefix of offline index (e.g., data/rag_index/offline)")
    parser.add_argument("--rag_corpus", type=str, default="prompts", help="Directory containing RAG_memory.txt")
    parser.add_argument("--rag_max_hits", type=int, default=5)
    parser.add_argument("--rag_threshold", type=float, default=0.0)
    parser.add_argument("--rag_fetch_k", type=int, default=5)
    parser.add_argument("--binary_output", action="store_true", help="Match DecisionTester binary_output prompt style.")

    parser.add_argument("--out", type=str, required=True, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=0, help="Max number of prompt entries to export (0 = no limit).")

    args = parser.parse_args()

    load_dotenv(dotenv_path=find_dotenv())

    # Resolve dataset path (allow passing a path directly).
    if args.dataset.endswith(".json") and os.path.exists(args.dataset):
        data_path = args.dataset
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    else:
        dataset_name = args.dataset
        data_path = os.path.join("tests", "decision_tester", "robot_states", f"{dataset_name}.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find dataset: {data_path}")

    # Initialize DecisionTester in "prompt builder" mode (no llm).
    _chat_template = infer_chat_template_from_model(args.model)
    evaluator = DecisionTester(
        llm=None,
        model_name=args.model,
        all_tests=False,
        mini=args.mini,
        local=True,
        use_rag=args.rag,
        quant=True,  # irrelevant for prompt building, but keeps the config consistent
        rag_offline=args.rag_offline,
        rag_index=args.rag_index,
        rag_corpus=args.rag_corpus,
        rag_max_hits=args.rag_max_hits,
        rag_score_threshold=args.rag_threshold,
        rag_fetch_k=args.rag_fetch_k,
        binary_output=args.binary_output,
    )

    data_set = evaluator.load_dataset(data_path)
    if evaluator.mini_eval:
        data_set = data_set[::5]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for test in evaluator.TEST_CASES:
            for i, robot_state in enumerate(data_set):
                prompt, rag_text, rag_details, rag_candidates = evaluator.build_prompt(
                    human_prompt=test["human_prompt"],
                    robot_state=robot_state,
                )
                rec = {
                    "dataset": dataset_name,
                    "test_case": test["human_prompt"],
                    "sample_index": i,
                    "prompt": prompt,
                    "prompt_chars": len(prompt),
                    "rag_text": rag_text,
                    "rag_used_count": len(rag_details),
                    "rag_candidate_count": len(rag_candidates),
                    "binary_output": bool(args.binary_output),
                    "chat_template": _chat_template,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if args.limit and n >= args.limit:
                    break
            if args.limit and n >= args.limit:
                break

    print(f"Exported {n} prompts to {args.out}")


if __name__ == "__main__":
    main()

