# Windows consoles here default to cp949 and SQuAD titles are not always ASCII
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
