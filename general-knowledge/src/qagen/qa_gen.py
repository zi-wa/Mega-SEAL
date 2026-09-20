"""Self-assessment questions: the generation prompt, parsing, and the string metrics for RQ1."""
import re
import string
from collections import Counter
from typing import Dict, List, Sequence

import config

# base-model prompt in SEAL's style: the first stem is part of the prompt so the format is fixed
QA_GEN_TEMPLATE = (
    "Let's read the following passage and write questions that check whether a reader has learned "
    "its content, each with a short answer. Write every item as \"Question N: ...\" on one line "
    "and \"Answer: ...\" on the next line.\n\n"
    "Passage:\n{title}\n{context}\n\n"
    "Question 1: "
)

_QA_ITEM = re.compile(
    r"Question\s*\d*\s*[:.]?\s*(?P<question>[^\n]+)\n+\s*Answer\s*[:.]?\s*(?P<answer>[^\n]+)",
    re.IGNORECASE,
)
_PUNCTUATION = str.maketrans("", "", string.punctuation)
_ARTICLES = {"a", "an", "the"}


def qa_gen_prompt(passage: Dict[str, str]) -> str:
    return QA_GEN_TEMPLATE.format(title=passage["title"], context=passage["context"])


def parse_qa_pairs(completion: str, max_pairs: int = config.QA_GEN_MAX_PAIRS) -> List[Dict[str, str]]:
    text = "Question 1: " + completion.strip()
    pairs: List[Dict[str, str]] = []
    seen = set()
    for match in _QA_ITEM.finditer(text):
        question = " ".join(match.group("question").split())
        answer = " ".join(match.group("answer").split())
        fingerprint = normalize(question)
        if not question or not answer or not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        pairs.append({"question": question, "answer": answer})
        if len(pairs) == max_pairs:
            break
    return pairs


def normalize(text: str) -> str:
    """SQuAD normalization: lowercase, no punctuation, no articles, single spaces."""
    lowered = text.lower().translate(_PUNCTUATION)
    return " ".join(word for word in lowered.split() if word not in _ARTICLES)


def squad_f1(prediction: str, gold: str) -> float:
    predicted_tokens = normalize(prediction).split()
    gold_tokens = normalize(gold).split()
    if not predicted_tokens or not gold_tokens:
        return float(predicted_tokens == gold_tokens)
    shared = Counter(predicted_tokens) & Counter(gold_tokens)
    overlap = sum(shared.values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_match(prediction: str, gold: str) -> bool:
    return normalize(gold) in normalize(prediction)


def answer_in_passage(answer: str, context: str) -> bool:
    return normalize(answer) in normalize(context)


def duplicate_rate(pairs: Sequence[Dict[str, str]]) -> float:
    if not pairs:
        return 0.0
    answers = [normalize(pair["answer"]) for pair in pairs]
    return 1.0 - len(set(answers)) / len(answers)


def gold_coverage(gold_answers: Sequence[str], generated_answers: Sequence[str],
                  threshold: float = 0.5) -> float:
    """Share of gold answers some generated answer gets close to; 0 when nothing was generated."""
    if not gold_answers:
        return 0.0
    if not generated_answers:
        return 0.0
    covered = sum(
        any(squad_f1(generated, gold) >= threshold for generated in generated_answers)
        for gold in gold_answers
    )
    return covered / len(gold_answers)
