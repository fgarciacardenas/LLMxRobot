# inference/rag_build_index.py
import argparse, glob, os, json
from inference.rag_offline import OfflineRetriever, LocalEmbeddings

def load_corpus(corpus_dir: str):
    texts, ids = [], []
    for p in sorted(glob.glob(os.path.join(corpus_dir, "**/*"), recursive=True)):
        if os.path.isdir(p):
            continue
        low = p.lower()
        if low.endswith((".txt", ".md")):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                t = f.read().strip()
            if t:
                texts.append(t)
                ids.append(os.path.relpath(p, corpus_dir))
        elif low.endswith(".jsonl"):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for ln, line in enumerate(f, 1):
                    try:
                        obj = json.loads(line)
                        t = (obj.get("text") or obj.get("content") or "").strip()
                        if t:
                            texts.append(t)
                            ids.append(os.path.relpath(p, corpus_dir) + f":{ln}")
                    except Exception:
                        pass
    return texts, ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", required=True, help="Folder with .txt/.md/.jsonl")
    ap.add_argument("--index_path", required=True, help="Output index prefix (no extension)")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    args = ap.parse_args()

    texts, ids = load_corpus(args.corpus_dir)
    if not texts:
        raise SystemExit(f"No usable files in {args.corpus_dir}")

    retr = OfflineRetriever(index_path=args.index_path, embeddings=LocalEmbeddings(model_name=args.model))
    retr.build(texts, ids)
    print(f"Indexed {len(texts)} items into {args.index_path}.faiss/.npz and {args.index_path}.meta.pkl")

if __name__ == "__main__":
    main()
