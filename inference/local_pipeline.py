# inference/local_pipeline.py
import sys
import re
import pexpect

ASSISTANT_TAG = "Assistant:"
USER_TAG = "User:"
PROMPT_MARK = "__PEXPECT_LOCAL_READY__> "

# Atomic capture: everything printed after "Assistant:" up to next "User:"
_ANS_PAT = re.compile(
    rf"{re.escape(ASSISTANT_TAG)}\s*(?P<ans>.*?)\s*{re.escape(USER_TAG)}",
    re.DOTALL,
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)

class LocalLLMPipeline:
    """
    Local interactive runner for:
        ./inference_llm.py <model>      (REPL)
    Expects the REPL to show 'User:' when ready, and to prefix answers with 'Assistant:'.

    Usage matches your tester: llm(prompt) -> (text, None, None)
    """

    def __init__(
        self,
        workdir: str = "voyager-sdk",
        venv_activate: str = "venv/bin/activate",
        run_cmd: str = "./inference_llm.py llama-3-2-3b-1024-4core-static",
        login_timeout: int = 30,      # time to bring up bash + venv + program
        prompt_timeout: int = 360,    # time to wait for first 'User:' and for each turn
        strip_ansi: bool = True,
        verbose: bool = False,
    ):
        self.workdir = workdir
        self.venv = venv_activate
        self.run_cmd = run_cmd
        self.login_timeout = login_timeout
        self.prompt_timeout = prompt_timeout
        self.strip_ansi = strip_ansi
        self.verbose = verbose

        self.child = None
        self._open_session()

    def _open_session(self):
        # Start a local bash with a generous timeout for setup
        self.child = pexpect.spawn(
            "bash",
            ["-lc", "exec bash --noprofile --norc"],  # clean shell
            encoding="utf-8",
            timeout=self.login_timeout,
            env={"TERM": "dumb"},
        )
        if self.verbose:
            self.child.logfile = sys.stdout

        # Create a deterministic shell prompt
        self.child.sendline(f'export PS1="{PROMPT_MARK}"')
        self.child.expect(re.escape(PROMPT_MARK))

        # OPTIONAL: turn off local input echo to reduce clutter
        try:
            self.child.sendline("stty -echo")
            self.child.expect(re.escape(PROMPT_MARK), timeout=2)
        except Exception:
            pass

        # cd, activate venv, launch the interactive REPL
        self.child.sendline(f"cd {self.workdir} && source {self.venv} && {self.run_cmd}")
        self.child.timeout = self.prompt_timeout
        self._expect_user_prompt()

    def _expect_user_prompt(self):
        self.child.expect(USER_TAG)

    def _ask_once(self, prompt: str) -> str:
        # Collapse multi-line prompt into one logical line so REPL treats it as ONE turn
        one_line = " ".join(prompt.split())
        self.child.sendline(one_line)

        # Some REPLs echo 'User:' immediately; skip it quickly if present
        try:
            self.child.expect_exact(USER_TAG, timeout=0.5)
        except pexpect.TIMEOUT:
            pass

        # Atomically capture Assistant: ... next User:
        self.child.expect(_ANS_PAT)
        answer = self.child.match.group("ans") or ""

        # Strip ANSI + common board noise
        if self.strip_ansi:
            answer = strip_ansi(answer)
        NOISE = re.compile(
            r"^(INFO|WARNING|ERROR|\[libtriton.*|Disabling PyTorch.*|None of PyTorch.*)$",
            re.IGNORECASE,
        )
        answer = "\n".join(
            ln for ln in (ln.strip() for ln in answer.splitlines())
            if ln and not NOISE.match(ln)
        ).strip()

        if answer.startswith(":"):
            answer = answer[1:].lstrip()
        return answer

    def close(self):
        try:
            if self.child is not None and self.child.isalive():
                # restore echo if we disabled it
                try:
                    self.child.sendline("stty echo")
                    self.child.expect([pexpect.TIMEOUT, re.escape(PROMPT_MARK)], timeout=1)
                except Exception:
                    pass
                try:
                    self.child.sendline("exit")
                    self.child.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=3)
                except Exception:
                    pass
        finally:
            try:
                if self.child is not None:
                    self.child.close(force=True)
            except Exception:
                pass
            self.child = None

    def __del__(self):
        self.close()

    # tester-compatible API
    def __call__(self, text: str, **_) -> tuple[str, None, None]:
        try:
            out = self._ask_once(text)
            return out, None, None
        except Exception as e:
            raise RuntimeError(f"Local interactive inference failed: {e}")
