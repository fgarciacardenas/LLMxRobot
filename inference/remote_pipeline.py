# inference/remote_pipeline.py
import os, sys, re, pexpect, getpass

ASSISTANT_TAG = "Assistant:"
USER_TAG = "User:"
PROMPT_MARK = "__PEXPECT_READY__> "

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
        run_cmd: str = "./inference_llm.py llama-3-2-1b-1024-4core-static",
        ssh_opts: str = "-tt",
        login_timeout: int = 60,
        prompt_timeout: int = 180,
        strip_ansi: bool = True,
        ssh_password: str | None = None, 
        ssh_key_passphrase: str | None = None, 
        ssh_2fa_code: str | None = None
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
        # Comment this out once stable.
        # self.child.logfile = sys.stdout

        # Common first-time and auth prompts we need to handle
        auth_patterns = [
            r"Are you sure you want to continue connecting.*\?",      # new host key
            r"(?i)password:",                                         # account password
            r"(?i)passphrase for key.*:",                             # SSH key passphrase
            r"(?i)enter passphrase.*:",                               # another passphrase variant
            r"(?i)verification code:",                                # TOTP / OTP
            r"(?i)duo two-factor|send a push|passcode|phone call",    # Duo banners at ETH
            r"[^\n]*[$#>%]\s*$",                                      # a shell prompt (bash/zsh/fish) - very loose
            pexpect.TIMEOUT,
            pexpect.EOF,
        ]

        # loop until we see a shell prompt
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
                # If a code is required, read from env/arg:
                code = os.getenv("SSH_2FA_CODE") or getattr(self, "ssh_2fa_code", None)
                if code:
                    self.child.sendline(code)
                else:
                    # If it’s a selection menu (1) Duo Push, etc., choose 1 by default:
                    self.child.sendline("1")
                continue
            elif i == 6:
                # got a prompt
                break
            elif i == 7:
                # timeout – show what we saw
                raise TimeoutError(f"Timeout waiting for shell prompt. Buffer so far:\n{self.child.before}")
            else:
                raise RuntimeError(f"SSH ended unexpectedly. Output:\n{self.child.before}")

        # Set a deterministic prompt so subsequent expects are trivial
        # Works in bash/zsh/fish; harmless if not supported.
        self.child.sendline(f'export PS1="{PROMPT_MARK}"')
        self.child.expect(re.escape(PROMPT_MARK))

        print("Logging in...")
        # Now cd + activate venv + run REPL
        self.child.sendline(f"cd {self.workdir} && source {self.venv_activate} && {self.run_cmd}")
        # after lots of INFO lines, wait for first 'User:'
        self.child.timeout = self.prompt_timeout
        self._expect_user_prompt()

    def _expect_user_prompt(self):
        self.child.expect(USER_TAG)

    def _ask_once(self, prompt: str) -> str:
        self.child.sendline(prompt)
        self.child.expect(ASSISTANT_TAG)
        self.child.expect(USER_TAG)
        answer = (self.child.before or "").strip()
        if answer.startswith(":"):
            answer = answer[1:].lstrip()
        if self.strip_ansi:
            answer = strip_ansi(answer)
        return answer

    def close(self):
        try:
            if self.child is not None:
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
