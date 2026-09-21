"""Where a run writes: results under the repo, adapters under the gitignored models folder."""
import json
import time
from pathlib import Path
from typing import Dict, List

import config


def run_dir() -> Path:
    return Path(config.RESULTS_ROOT) / config.RUN_NAME


def adapter_root() -> Path:
    return Path(config.ADAPTER_ROOT) / config.RUN_NAME


STARTED = time.strftime("%Y%m%d_%H%M%S")


def usage_path(stage: str) -> Path:
    """One judge-usage file per process, so restarts add up instead of overwriting."""
    return run_dir() / "usage" / f"{stage}_{STARTED}.json"


def cache_path() -> Path:
    return run_dir() / "grader_cache.jsonl"


def outer_adapter(iteration: int, chain: str = "main") -> Path:
    return adapter_root() / f"outer_{chain}_{iteration}"


def se_rl_adapter(condition: str) -> Path:
    return adapter_root() / f"se_rl_{condition}"


def read_jsonl(path: Path) -> List[Dict]:
    """Records of an append-only log; a last line cut short by a kill is trimmed off."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    complete = raw[: raw.rfind(b"\n") + 1]
    if len(complete) != len(raw):
        path.write_bytes(complete)  # the next append must start on a fresh line
    return [json.loads(line) for line in complete.decode("utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(record, ensure_ascii=False) + "\n")
