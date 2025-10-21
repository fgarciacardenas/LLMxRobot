# inference/token_utils.py
from typing import Optional
from transformers import AutoTokenizer

# Map a model id to a local HF tokenizer id
def _resolve_tokenizer_id(model_id: str) -> Optional[str]:
    mid = (model_id or "").lower()
    if "qwen" in mid:
        return "Qwen/Qwen2.5-7B-Instruct"
    if "phi-3" in mid:
        return "microsoft/Phi-3-mini-4k-instruct"
    if "llama-3.2" in mid or "llama-3" in mid:
        return "unsloth/Llama-3.2-3B-Instruct"  # tokenizer-compatible across 3.x
    # If local folders in models/, pick a default that is close.
    if mid.startswith("models/"):
        return "unsloth/Llama-3.2-3B-Instruct"
    # For OpenAI (gpt-4o), we skip exact counting to avoid extra deps.
    if "gpt-4o" in mid:
        return None
    # Fallback
    return "unsloth/Llama-3.2-3B-Instruct"

def get_tokenizer(model_id: str):
    tok_id = _resolve_tokenizer_id(model_id)
    if tok_id is None:
        return None
    tok = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    # Ensure a pad token id exists for consistent encoding length (not strictly needed for counting)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok

def count_tokens(text: str, tok) -> int:
    if tok is None or text is None:
        return 0
    # fast tokenizers: .encode returns a list of ids
    return len(tok.encode(text))
