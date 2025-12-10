# inference/remote_pipeline.py
import os, sys, re, pexpect, getpass

ASSISTANT_TAG = "Assistant:"
USER_TAG = "User:"
PROMPT_MARK = "__PEXPECT_READY__> "
READY_SENTINEL = "__PEXPECT_READY_SENTINEL__"

# Atomic capture: everything printed after "Assistant:" up to next "User:"
_ANS_PAT = re.compile(
    rf"{re.escape(ASSISTANT_TAG)}\s*(?P<ans>.*?)\s*{re.escape(USER_TAG)}",
    re.DOTALL,
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)

def _get_password(cli_password: str | None):
    if cli_password is not None:
        return cli_password
    return os.getenv("SSH_PASSWORD")  # or None

class RemoteLLMPipeline:
    """
    SSH, activate venv, run ./inference_llm.py ..., and talk to the REPL.
    Waits for 'User:' before sending, captures text between 'Assistant:' and next 'User:'.
    """

    def __init__(
        self,
        ssh_user: str,
        ssh_host: str,
        workdir: str = "voyager-sdk",
        venv_activate: str = "venv/bin/activate",
        run_cmd: str = "./inference_llm.py llama-3-2-3b-1024-4core-static",
        ssh_opts: str = "-tt",
        login_timeout: int = 90,     # a little higher; MOTD can be slow
        prompt_timeout: int = 240,   # model startup can be chatty
        strip_ansi: bool = True,
        ssh_password: str | None = None,
        ssh_key_passphrase: str | None = None,
        ssh_2fa_code: str | None = None,
        ssh_verbose: bool = False,
    ):
        self.ssh_user = ssh_user
        self.ssh_host = ssh_host
        self.workdir = workdir
        self.venv_activate = venv_activate
        self.run_cmd = run_cmd
        self.ssh_opts = ssh_opts
        self.login_timeout = login_timeout
        self.prompt_timeout = prompt_timeout
        self.strip_ansi = strip_ansi
        self.ssh_password = ssh_password
        self.ssh_key_passphrase = ssh_key_passphrase
        self.ssh_2fa_code = ssh_2fa_code
        self.ssh_verbose = ssh_verbose

        self.child = None
        self._open_session()

    def _open_session(self):
        print("Opening SSH session...")
        cmd = f"ssh {self.ssh_opts} {self.ssh_user}@{self.ssh_host}"
        self.child = pexpect.spawn(
            cmd,
            encoding="utf-8",
            timeout=self.login_timeout,
            env={"TERM": "dumb"},
        )

        # SEE EVERYTHING: stream SSH output to stdout for debugging.
        if self.ssh_verbose:
            self.child.logfile = sys.stdout

        # First-time prompts and auth we need to handle
        auth_patterns = [
            r"Are you sure you want to continue connecting.*\?",      # new host key
            r"(?i)password:",                                         # account password
            r"(?i)passphrase for key.*:",                             # SSH key passphrase
            r"(?i)enter passphrase.*:",                               # another passphrase variant
            r"(?i)verification code:",                                # TOTP / OTP
            r"(?i)duo two-factor|send a push|passcode|phone call",    # Duo banners at ETH
            r"[^\n]*[$#>%]\s*$",                                      # a shell prompt (bash/zsh/fish)
            pexpect.TIMEOUT,
            pexpect.EOF,
        ]

        # Drive auth until we *either* see a prompt *or* we know the remote is ready.
        saw_prompt = False
        while True:
            i = self.child.expect(auth_patterns)
            if i == 0:
                # host key prompt
                self.child.sendline("yes")
                continue
            elif i == 1:
                # account password
                pw = _get_password(getattr(self, "ssh_password", None))
                if not pw:
                    raise RuntimeError("SSH asked for a password. Use key-based auth or pass --ssh_password / SSH_PASSWORD.")
                self.child.sendline(pw)
                continue
            elif i in (2, 3):
                # key passphrase
                pw = _get_password(getattr(self, "ssh_key_passphrase", None))
                if not pw:
                    raise RuntimeError("Your SSH key needs a passphrase. Unlock your agent or pass --ssh_key_passphrase.")
                self.child.sendline(pw)
                continue
            elif i in (4, 5):
                # 2FA / Duo. Best is to approve on your device.
                code = os.getenv("SSH_2FA_CODE") or getattr(self, "ssh_2fa_code", None)
                self.child.sendline(code if code else "1")
                continue
            elif i == 6:
                # got a prompt
                saw_prompt = True
                break
            elif i == 7:
                # TIMEOUT waiting for prompt: MOTD/banner likely hid it.
                # Proactively test readiness by printing a sentinel.
                try:
                    self.child.sendline("")  # kick a newline
                    self.child.sendline(f'printf "{READY_SENTINEL}\\n"')
                    self.child.expect_exact(READY_SENTINEL, timeout=15)
                    # We're definitely in a shell now.
                    break
                except pexpect.TIMEOUT:
                    raise TimeoutError(
                        "Timeout waiting for shell prompt (MOTD/banner?). "
                        f"Buffer so far:\n{self.child.before}"
                    )
            else:
                raise RuntimeError(f"SSH ended unexpectedly. Output:\n{self.child.before}")

        # Set a deterministic prompt so subsequent expects are trivial
        # Works in bash/zsh/fish; harmless if not supported.
        self.child.sendline(f'export PS1="{PROMPT_MARK}"')
        # We may not have seen a prompt before; wait for ours now.
        self.child.expect(re.escape(PROMPT_MARK))

        # OPTIONAL: turn off remote input echo to reduce clutter
        try:
            self.child.sendline("stty -echo")
            self.child.expect(re.escape(PROMPT_MARK), timeout=2)
        except Exception:
            pass

        print("Logging in...")
        # Now cd + activate venv + run REPL
        self.child.sendline(f"cd {self.workdir} && source {self.venv_activate} && {self.run_cmd}")
        # after lots of INFO lines, wait for first 'User:'
        self.child.timeout = self.prompt_timeout
        self._expect_user_prompt()

    def _expect_user_prompt(self):
        self.child.expect(USER_TAG)

    def _ask_once(self, prompt: str) -> str:
        # 1) Collapse multi-line prompt to a single logical line so the REPL
        #    treats it as ONE user turn (no splitting at newlines).
        one_line = " ".join(prompt.split())
        self.child.sendline(one_line)

        # 2) Some CLIs immediately echo "User:" (and even our text). Try to skip
        #    any immediate echo quickly so we don't confuse it with the *next* prompt.
        try:
            self.child.expect_exact(USER_TAG, timeout=0.5)
        except pexpect.TIMEOUT:
            pass

        # 3) Atomically capture "Assistant: ... <next> User:" to avoid partial
        #    captures or device logs in between.
        self.child.expect(_ANS_PAT)
        answer = self.child.match.group("ans") or ""

        # 4) Strip ANSI + common noise lines the board prints.
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

        # 5) Tidying: some UIs prefix with a colon right after "Assistant:"
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

    # mimic transformers.Pipeline __call__
    def __call__(self, text: str, **_) -> tuple[str, None, None]:
        try:
            out = self._ask_once(text)
            return out, None, None
        except Exception as e:
            raise RuntimeError(f"Remote interactive inference failed: {e}")
