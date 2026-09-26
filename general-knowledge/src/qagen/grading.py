"""LLM judge calls: SEAL's grading prompt, background threads, cache, token accounting."""
import hashlib
import itertools
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import openai
from openai import OpenAI

import config

from ..utils import SQUAD_GRADE_TEMPLATE, parse_yes_no
from .paths import append_jsonl, read_jsonl

RETRY_WINDOW_SECONDS = 30 * 60  # outages shorter than this must not end a multi-day run
TRANSIENT_ERRORS = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError)

GradeItem = Tuple[str, str, str]  # question, gold answer, model answer


def _template_id(template: str) -> str:
    return hashlib.sha1(template.encode("utf-8")).hexdigest()[:8]


class Grader:
    """One judge model, frozen for a whole run; verdicts cached by exact text."""

    def __init__(
        self,
        cache_path: Path,
        model: str = config.GRADER_MODEL,
        reasoning_effort: str = config.GRADER_REASONING_EFFORT,
        workers: int = config.GRADER_WORKERS,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = None  # built on the first uncached call, so cached runs need no key
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.lock = threading.Lock()
        self.cache_path = Path(cache_path)
        self.cache: Dict[str, bool] = {}
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.models_seen: set = set()
        self._load_cache()

    def submit(self, items: Sequence[GradeItem]) -> List[Future]:
        """Queue grading so the GPU can run the next adaptation meanwhile."""
        return [self.pool.submit(self._answer_verdict, item) for item in items]

    def grade(self, items: Sequence[GradeItem]) -> List[bool]:
        return [pending.result() for pending in self.submit(items)]

    def grade_uncached(self, items: Sequence[GradeItem]) -> List[bool]:
        """Second opinion from the same judge, for the self-consistency check."""
        pending = [self.pool.submit(self._ask_answer_verdict, item) for item in items]
        return [future.result() for future in pending]

    def usage(self) -> Dict[str, object]:
        cost = (
            self.input_tokens / 1e6 * config.GRADER_PRICE_IN
            + self.output_tokens / 1e6 * config.GRADER_PRICE_OUT
        )
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(cost, 4),
            "models": sorted(self.models_seen),
        }

    def save_usage(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.usage(), indent=2), encoding="utf-8")

    def close(self) -> None:
        self.pool.shutdown(wait=True)

    def _answer_verdict(self, item: GradeItem) -> bool:
        question, gold, prediction = item
        if not prediction.strip():
            return False  # SEAL counts an empty answer wrong without spending a call
        key = self._key(_template_id(SQUAD_GRADE_TEMPLATE), question, gold, prediction)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        verdict = self._ask_answer_verdict(item)
        self._remember(key, verdict)
        return verdict

    def _ask_answer_verdict(self, item: GradeItem) -> bool:
        question, gold, prediction = item
        prompt = SQUAD_GRADE_TEMPLATE.format(question=question, gold=gold, pred=prediction.strip())
        return parse_yes_no(self._ask(prompt))

    def _ask(self, prompt: str) -> str:
        with self.lock:
            if self.client is None:
                self.client = OpenAI()  # reads OPENAI_API_KEY from the environment
        request = {"model": self.model, "input": prompt, "temperature": 0}  # greedy, as SEAL's B.4 grading
        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}
        deadline = time.time() + RETRY_WINDOW_SECONDS
        for attempt in itertools.count():
            try:
                response = self.client.responses.create(**request)
                self._count(response)
                if response.output_text.strip():  # an empty reply is retried, never cached
                    return response.output_text
            except TRANSIENT_ERRORS:
                pass
            if time.time() > deadline:
                raise RuntimeError(f"judge gave no answer for {RETRY_WINDOW_SECONDS // 60} minutes")
            time.sleep(min(60, 2**attempt))

    def _count(self, response) -> None:
        with self.lock:
            self.calls += 1
            self.models_seen.add(getattr(response, "model", self.model))
            usage = getattr(response, "usage", None)
            if usage:
                self.input_tokens += getattr(usage, "input_tokens", 0)
                self.output_tokens += getattr(usage, "output_tokens", 0)

    def _key(self, template_id: str, *fields: str) -> str:
        payload = "|".join([self.model, template_id, *fields])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _remember(self, key: str, verdict: bool) -> None:
        with self.lock:
            self.cache[key] = verdict
            append_jsonl(self.cache_path, {"key": key, "verdict": verdict})

    def _load_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        for entry in read_jsonl(self.cache_path):
            self.cache[entry["key"]] = entry["verdict"]
