"""Where a run writes: results under the repo, adapters under the gitignored models folder."""
from pathlib import Path

import config


def run_dir() -> Path:
    return Path(config.RESULTS_ROOT) / config.RUN_NAME


def adapter_root() -> Path:
    return Path(config.ADAPTER_ROOT) / config.RUN_NAME


def cache_path() -> Path:
    return run_dir() / "grader_cache.jsonl"


def outer_adapter(iteration: int, chain: str = "main") -> Path:
    return adapter_root() / f"outer_{chain}_{iteration}"


def se_rl_adapter(condition: str) -> Path:
    return adapter_root() / f"se_rl_{condition}"
