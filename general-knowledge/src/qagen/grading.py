"""LLM judge calls: SEAL's grading prompt, background threads, cache, token accounting."""
import hashlib
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from openai import OpenAI

import config

from ..utils import SQUAD_GRADE_TEMPLATE, parse_yes_no

# QA_gen answers are model-written, so their correctness is judged against the passage itself
PASSAGE_GRADE_TEMPLATE = (
    "You are a grading assistant. Decide whether the answer to the question is correct according "
    "to the passage alone. Do not use any outside knowledge. Respond ONLY with 'yes' or 'no'.\n\n"
    "Passage:\n{passage}\n\nQuestion: {question}\nAnswer: {answer}\n"
    "Is the answer correct according to the passage? Respond 'yes' or 'no'."
)

RETRIES = 5

GradeItem = Tuple[str, str, str]  # question, gold answer, model answer
PassageItem = Tuple[str, str, str]  # passage, question, answer


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
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
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

    def grade_against_passage(self, items: Sequence[PassageItem]) -> List[bool]:
        return [pending.result() for pending in
                [self.pool.submit(self._passage_verdict, item) for item in items]]

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

    def _passage_verdict(self, item: PassageItem) -> bool:
        passage, question, answer = item
        key = self._key(_template_id(PASSAGE_GRADE_TEMPLATE), passage, question, answer)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        prompt = PASSAGE_GRADE_TEMPLATE.format(passage=passage, question=question, answer=answer)
        verdict = parse_yes_no(self._ask(prompt))
        self._remember(key, verdict)
        return verdict

    def _ask(self, prompt: str) -> str:
        request = {"model": self.model, "input": prompt}
        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}
        for attempt in range(RETRIES):
            try:
                response = self.client.responses.create(**request)
            except Exception as error:
                failure = error
                time.sleep(2**attempt)
                continue
            self._count(response)
            return response.output_text
        raise RuntimeError(f"judge call failed {RETRIES} times: {failure}")

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
            with self.cache_path.open("a", encoding="utf-8") as cache_file:
                cache_file.write(json.dumps({"key": key, "verdict": verdict}) + "\n")

    def _load_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.cache_path.exists():
            return
        with self.cache_path.open(encoding="utf-8") as cache_file:
            for line in cache_file:
                entry = json.loads(line)
                self.cache[entry["key"]] = entry["verdict"]
