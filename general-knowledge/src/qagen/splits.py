"""SQuAD splits: SEAL's shuffle, plus the disjoint pools this study needs."""
import hashlib
import json
import random
from functools import lru_cache
from typing import Any, Dict, List

import config

Passage = Dict[str, Any]

DEV_START = 50
OUTER_POOL_START = 80
MIN_GOLD_QUESTIONS = 5  # coarse reward scores otherwise: 1/4 of a question per step


def passage_key(passage: Passage) -> str:
    """SQuAD titles repeat across passages, so identity needs the context too."""
    digest = hashlib.sha1(passage["context"].encode("utf-8")).hexdigest()[:8]
    return f"{passage['title']}#{digest}"


@lru_cache(maxsize=2)
def _shuffled(path: str) -> tuple:
    passages = json.load(open(path, encoding="utf-8"))
    random.Random(42).shuffle(passages)  # same permutation SEAL's make_squad_data uses
    return tuple(passages)


def se_rl_passages() -> List[Passage]:
    return list(_shuffled(config.SQUAD_TRAIN)[: config.SE_RL_PASSAGES])


def dev_passages() -> List[Passage]:
    return list(_shuffled(config.SQUAD_TRAIN)[DEV_START : DEV_START + config.DEV_PASSAGES])


def outer_passages(iteration: int) -> List[Passage]:
    pool = [p for p in _shuffled(config.SQUAD_TRAIN)[OUTER_POOL_START:]
            if len(p["questions"]) >= MIN_GOLD_QUESTIONS]
    start = iteration * config.OUTER_PASSAGES
    return pool[start : start + config.OUTER_PASSAGES]


def val_passages() -> List[Passage]:
    return list(_shuffled(config.SQUAD_VAL)[: config.VAL_PASSAGES])
