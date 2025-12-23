from llama_cpp import Llama
import os
import re

ADHERING_RE = re.compile(r"adhering\s*to\s*human\s*:\s*(true|false)", re.IGNORECASE)

class RaceLLMGGGUF:
    def __init__(self, model_dir, gguf_name, chat_format="llama-3", max_tokens=512, n_ctx=4096, n_gpu_layers=-1, binary_output=False):
        self.max_tokens = max_tokens
        self.path = os.path.join(model_dir, gguf_name)
        self.binary_output = binary_output
        self.chat_format = self._normalize_chat_format(chat_format)
        self.llm = Llama(
            model_path=self.path,
            chat_format=self.chat_format,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            seed=42,
            verbose=False
            )

    def __call__(self, input_text):
        output = self.llm.create_chat_completion(
            messages=[{
                     "role": "user",
                     "content": input_text
                 }],
            max_tokens=self.max_tokens,
            temperature=0.0,
            top_k=1,
            stop=self._build_stop_list(),
        )
        out_text = output['choices'][0]['message']['content']
        # Truncate to first adherence marker if present (binary mode)
        if self.binary_output:
            m = ADHERING_RE.search(out_text)
            if m:
                out_text = out_text[:m.end()]
        input_tokens = output['usage']['prompt_tokens']
        out_tokens = output['usage']['completion_tokens']
        return out_text, input_tokens, out_tokens

    def close(self):
        llm = getattr(self, "llm", None)
        if llm is None:
            return
        try:
            close_fn = getattr(llm, "close", None)
            if callable(close_fn):
                close_fn()
        finally:
            self.llm = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _normalize_chat_format(self, chat_format: str) -> str:
        """
        Map higher-level template names to llama.cpp registered handlers.
        phi-3 SFTs use ChatML; qwen-2.5 -> qwen; llama-3.2 -> llama-3.
        """
        cf = (chat_format or "").lower()
        if cf in ("phi-3", "phi3"):
            return "chatml"
        if cf in ("qwen-2.5", "qwen2.5", "qwen2", "qwen"):
            return "qwen"
        if cf in ("llama-3.2", "llama-3.1", "llama-3", "llama3"):
            return "llama-3"
        return cf or "llama-3"

    def _build_stop_list(self):
        base = []
        # Prefer the chat handler's defaults so we do not override built-in stop tokens (e.g., chatml <|im_end|>)
        handler = getattr(self.llm, "_chat_handler", None) or getattr(self.llm, "chat_handler", None)
        handler_stop = getattr(handler, "stop", None)
        if isinstance(handler_stop, (list, tuple)):
            base.extend(handler_stop)

        # Legacy generic stops (kept for backwards compatibility)
        base.extend([
            "[/INST]", "[\\/INST]", "[;/INST] ", "[INST]", "[/?]", "[/Dk]", "[;/Rationale]",
            "[Rationale]", "[;/Action]", "[;/Explanation]",
        ])
        # Drop dupes while preserving order
        seen = set()
        deduped = []
        for s in base:
            if s not in seen:
                deduped.append(s)
                seen.add(s)
        return deduped

# Loads Prompt with hints
def load_prompt(prompt_type) -> str:
    if 'reasoning' in prompt_type:
        hints_dir = os.path.join('../', 'prompts/reasoning_hints.txt')
        with open(hints_dir, 'r') as f:
            reasoning_hints = f.read()
        return reasoning_hints
    elif 'synthetic' in prompt_type:
        hints_dir = os.path.join('../', 'prompts/example_synthetic.txt')
        with open(hints_dir, 'r') as f:
            synthetic_hints = f.read()
        return synthetic_hints
    else:
        raise ValueError(f"Prompt type {prompt_type} not recognized. Please use 'reasoning' or 'synthetic'.")
