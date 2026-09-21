# Windows consoles here default to cp949 and SQuAD titles are not always ASCII
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class _Tee:
    """Console plus log file, so output and tracebacks survive a closed window."""

    def __init__(self, console, log):
        self.console = console
        self.log = log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


# run_all.bat sets QAGEN_LOG; direct runs and tests print to the console only
if os.environ.get("QAGEN_LOG"):
    _log = open(os.environ["QAGEN_LOG"], "a", encoding="utf-8", errors="replace")
    _log.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(sys.orig_argv[1:])}\n")
    sys.stdout = _Tee(sys.stdout, _log)
    sys.stderr = _Tee(sys.stderr, _log)
