"""Inner loop: self-edit, temporary LoRA, closed-book answers, graded accuracy, ReST-EM rounds."""
import random
from statistics import mean
from typing import Callable, Dict, List, Optional, Sequence

import config

from ..data_generation.make_squad_data import MAKE_SQUAD_DATA_TEMPLATES_BASE
from ..utils import build_train_sequences, format_answer_prompts
from .grading import Grader
from .model_ops import Policy
from .splits import passage_key

QuestionSet = List[Dict[str, str]]
SelfEditSource = Callable[[Dict], List[str]]

SELF_EDIT_TEMPLATE = MAKE_SQUAD_DATA_TEMPLATES_BASE["implications"]


def self_edit_prompt(passage: Dict) -> str:
    return SELF_EDIT_TEMPLATE.format(title=passage["title"], context=passage["context"])


def question_items(title: str, pairs: Sequence[Dict[str, str]]) -> QuestionSet:
    """SEAL puts the title on a topic line so the closed-book question is answerable."""
    return [
        {"question": f"Topic: {title}\n{pair['question']}", "answer": pair["answer"]}
        for pair in pairs
    ]


def gold_questions(passage: Dict) -> QuestionSet:
    return question_items(passage["title"], passage["questions"])


def sample_self_edits(policy: Policy, passage: Dict, count: int, seed: int) -> List[str]:
    prompts = [self_edit_prompt(passage)] * count
    return policy.generate(prompts, config.SELF_EDIT_MAX_TOKENS, seed=seed)


def answer_sets(policy: Policy, question_sets: Dict[str, QuestionSet], seed: int) -> Dict[str, List[str]]:
    """One batched closed-book pass over every question set the caller cares about."""
    names = [name for name, items in question_sets.items() if items]
    flat = [item for name in names for item in question_sets[name]]
    answers = policy.generate(
        format_answer_prompts(flat, instruct_model=False),
        config.ANSWER_MAX_TOKENS,
        seed=seed,
        greedy=True,
    )
    split: Dict[str, List[str]] = {}
    cursor = 0
    for name in names:
        size = len(question_sets[name])
        split[name] = answers[cursor : cursor + size]
        cursor += size
    return split


class PendingScores:
    """Verdict futures for one self-edit, so grading overlaps the next adaptation."""

    def __init__(self, question_sets: Dict[str, QuestionSet]):
        self.question_sets = question_sets
        self.per_seed: List[Dict[str, List]] = []
        self.answers: List[Dict[str, List[str]]] = []  # kept for the judge cross-check

    def add(self, grader: Grader, answers: Dict[str, List[str]]) -> None:
        self.answers.append(answers)
        graded = {}
        for name, items in self.question_sets.items():
            if not items:
                continue
            graded[name] = grader.submit(
                [(item["question"], item["answer"], answer)
                 for item, answer in zip(items, answers[name])]
            )
        self.per_seed.append(graded)

    def means(self) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for name in self.question_sets:
            seed_accuracies = [
                mean(verdict.result() for verdict in graded[name])
                for graded in self.per_seed
                if name in graded
            ]
            scores[name] = mean(seed_accuracies) if seed_accuracies else 0.0
        return scores


def closed_book_scores(policy: Policy, grader: Grader, question_sets: Dict[str, QuestionSet],
                       seed: int) -> PendingScores:
    pending = PendingScores(question_sets)
    pending.add(grader, answer_sets(policy, question_sets, seed))
    return pending


def adapted_scores(policy: Policy, grader: Grader, passage: Dict, self_edit: str,
                   question_sets: Dict[str, QuestionSet], seeds: Sequence[int]) -> PendingScores:
    """SEAL's inner loop measured over several TTT seeds; empty self-edit trains on the passage only."""
    sequences = build_train_sequences(
        self_edit, passage["context"], passage["title"], split_newlines=True
    )
    pending = PendingScores(question_sets)
    for seed in seeds:
        with policy.adapted(sequences, seed=seed):
            answers = answer_sets(policy, question_sets, seed)
        pending.add(grader, answers)
    return pending


def se_rl_round(policy: Policy, grader: Grader, passages: Sequence[Dict],
                reward_questions: Callable[[Dict], QuestionSet], round_index: int, seed: int,
                extra_questions: Optional[Callable[[Dict], QuestionSet]] = None) -> Dict:
    """One ReST-EM round: K self-edits per passage, keep the best one by the reward questions."""
    seeds = list(range(config.TTT_SEEDS))
    records = []
    pairs = []
    for passage_index, passage in enumerate(passages):
        question_sets = {"reward": reward_questions(passage)}
        if extra_questions:
            question_sets["gold"] = extra_questions(passage)
        if not question_sets["reward"]:
            continue  # nothing to reward with; the passage contributes no training pair
        sample_seed = seed * 1000 + round_index * 100 + passage_index
        self_edits = sample_self_edits(policy, passage, config.SELF_EDITS, seed=sample_seed)
        before = closed_book_scores(policy, grader, question_sets, seed=sample_seed)
        pending = [
            adapted_scores(policy, grader, passage, self_edit, question_sets, seeds)
            for self_edit in self_edits
        ]
        scores = [entry.means() for entry in pending]
        reward_accuracies = [entry["reward"] for entry in scores]
        chosen = max(range(len(scores)), key=lambda index: reward_accuracies[index])
        baseline = before.means()
        records.append({
            "key": passage_key(passage),
            "title": passage["title"],
            "closed_book": baseline,
            "reward_accuracies": reward_accuracies,
            "gold_accuracies": [entry.get("gold") for entry in scores],
            "reward_positive": [accuracy > baseline["reward"] for accuracy in reward_accuracies],
            "chosen": chosen,
            "truncated": [policy.token_count(edit) >= config.SELF_EDIT_MAX_TOKENS
                          for edit in self_edits],
        })
        pairs.append((self_edit_prompt(passage), self_edits[chosen]))
        print(f"[se-rl r{round_index}] {passage_index + 1}/{len(passages)} "
              f"{passage['title'][:40]} best {max(reward_accuracies):.2f} "
              f"closed-book {baseline['reward']:.2f}", flush=True)
    return {"round": round_index, "passages": records, "pairs": pairs}


def random_selection_round(policy: Policy, passages: Sequence[Dict], round_index: int,
                           seed: int) -> Dict:
    """Control: same amount of finetuning, self-edit picked without any reward."""
    picker = random.Random(seed * 1000 + round_index)
    records = []
    pairs = []
    for passage_index, passage in enumerate(passages):
        sample_seed = seed * 1000 + round_index * 100 + passage_index
        self_edits = sample_self_edits(policy, passage, config.SELF_EDITS, seed=sample_seed)
        chosen = picker.randrange(len(self_edits))
        records.append({"key": passage_key(passage), "title": passage["title"],
                        "chosen": chosen})
        pairs.append((self_edit_prompt(passage), self_edits[chosen]))
    return {"round": round_index, "passages": records, "pairs": pairs}


def evaluate_passages(policy: Policy, grader: Grader, passages: Sequence[Dict],
                      self_edit_source: SelfEditSource, label: str) -> Dict:
    """Held-out measurement: fresh self-edits, one TTT each, accuracy on the gold questions."""
    records = []
    ttt_before, seconds_before = policy.ttt_count, policy.ttt_seconds
    for passage_index, passage in enumerate(passages):
        question_sets = {"gold": gold_questions(passage)}
        self_edits = self_edit_source(passage)
        if self_edits:
            pending = [
                adapted_scores(policy, grader, passage, self_edit, question_sets, [index])
                for index, self_edit in enumerate(self_edits)
            ]
        else:
            pending = [closed_book_scores(policy, grader, question_sets, seed=passage_index)]
        accuracies = [entry.means()["gold"] for entry in pending]
        records.append({
            "key": passage_key(passage),
            "title": passage["title"],
            "question_count": len(question_sets["gold"]),
            "self_edit_accuracies": accuracies,
            "accuracy": mean(accuracies),
            "truncated": [policy.token_count(edit) >= config.SELF_EDIT_MAX_TOKENS
                          for edit in self_edits],
        })
        running = mean(record["accuracy"] for record in records)
        print(f"[eval {label}] {passage_index + 1}/{len(passages)} "
              f"{passage['title'][:40]} {records[-1]['accuracy']:.2f} running {running:.3f}",
              flush=True)
    return {
        "condition": label,
        "mean_accuracy": mean(record["accuracy"] for record in records),
        "passages": records,
        "ttt_count": policy.ttt_count - ttt_before,
        "gpu_seconds": policy.ttt_seconds - seconds_before,
    }
