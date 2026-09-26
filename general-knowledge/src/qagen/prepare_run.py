"""Run directory setup: judge cache and judge validation carried over, code snapshot with manifest.

Run as `python -m general-knowledge.src.qagen.prepare_run` before run_eval. Safe to repeat: the
snapshot is taken once, at the first start, so a resume keeps the code that started the run.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import config

from . import paths

QAGEN_DIR = Path("general-knowledge/src/qagen")
ROOT_FILES = ("config.py", "run_all.bat", "setup_win.bat",
              "general-knowledge/README.md", "general-knowledge/GLOSSARY.md")


def carry_over() -> None:
    """Same judge and prompt as the source run: reuse its verdicts and its agreement check."""
    source = Path(config.RESULTS_ROOT) / config.CACHE_FROM_RUN
    validation = source / "judge" / "validation.json"
    if validation.exists():
        judge_model = json.loads(validation.read_text(encoding="utf-8"))["judge_model"]
        if judge_model != config.GRADER_MODEL:
            raise RuntimeError(f"{config.CACHE_FROM_RUN} was judged by {judge_model}, config says "
                               f"{config.GRADER_MODEL}; its cache and judge check cannot be reused")
    for name in ("grader_cache.jsonl", "judge"):
        src, dst = source / name, paths.run_dir() / name
        if not src.exists() or dst.exists():
            continue
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
        print(f"[prepare] copied {name} from {config.CACHE_FROM_RUN}", flush=True)


def snapshot() -> None:
    code = paths.run_dir() / "code"
    manifest = code / "MANIFEST.txt"
    if manifest.exists():
        print(f"[prepare] snapshot already taken: {manifest}", flush=True)
        return
    (code / "qagen").mkdir(parents=True, exist_ok=True)  # a failed first attempt may have left it
    for path in list(QAGEN_DIR.glob("*.py")) + list(QAGEN_DIR.glob("*.md")):
        shutil.copy2(path, code / "qagen" / path.name)
    for name in ROOT_FILES:
        if Path(name).exists():
            shutil.copy2(name, code / Path(name).name)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    manifest.write_text(
        f"run {config.RUN_NAME}\nstarted {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"git {commit}\npython {sys.version.split()[0]}\n\npip freeze:\n{freeze}",
        encoding="utf-8",
    )
    print(f"[prepare] code snapshot: {code}", flush=True)


def main() -> None:
    paths.run_dir().mkdir(parents=True, exist_ok=True)
    carry_over()
    snapshot()


if __name__ == "__main__":
    main()
