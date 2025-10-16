import subprocess, shlex, re

ASSISTANT_TAG = "Assistant:"
END_SENTINEL  = "__END_OF_CMD__"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)

class LocalLLMPipeline:
    """
    Run ./inference_llm.py ... --prompt "<full prompt>" locally (non-interactive),
    using a heredoc to preserve newlines/quotes verbatim.
    Returns (text, None, None) like your RaceLLMPipeline for the tester.
    """

    def __init__(
        self,
        workdir: str = "voyager-sdk",
        venv_activate: str = "venv/bin/activate",
        run_cmd: str = "./inference_llm.py llama-3-2-3b-1024-4core-static",
        timeout_sec: int = 300,
        verbose: bool = False,
    ):
        self.workdir = workdir
        self.venv = venv_activate
        self.run_cmd = run_cmd
        self.timeout = timeout_sec
        self.verbose = verbose

    def _run_once(self, prompt: str) -> str:
        # bash -lc so we can `source` and use a literal heredoc <<"EOF"
        script = f'''
cd {shlex.quote(self.workdir)} && \
source {shlex.quote(self.venv)} || exit 1
read -r -d "" PROMPT << "EOF"
{prompt}
EOF
{self.run_cmd} --prompt "$PROMPT"
printf "\\n{END_SENTINEL}\\n"
'''
        proc = subprocess.run(
            ["bash", "-lc", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        if self.verbose:
            print("---- LOCAL STDOUT ----\n", proc.stdout)
            print("---- LOCAL STDERR ----\n", proc.stderr)

        if proc.returncode != 0:
            msg = proc.stderr.strip() or proc.stdout.strip()
            raise RuntimeError(f"Local inference failed ({proc.returncode}): {msg}")

        # Keep only the output before the sentinel
        out = proc.stdout.split(END_SENTINEL, 1)[0].strip()
        out = strip_ansi(out)

        # If tool prefixes "Assistant:", keep after it; else keep all
        m = re.search(rf"{re.escape(ASSISTANT_TAG)}\s*(.*)$", out, re.IGNORECASE | re.DOTALL)
        if m:
            out = m.group(1).strip()
        return out

    # mimic transformers.Pipeline __call__
    def __call__(self, text: str, **_) -> tuple[str, None, None]:
        answer = self._run_once(text)
        return answer, None, None
