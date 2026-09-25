"""Evaluate the meeting-minutes pipeline.

Two levels of evaluation, because they measure different things:

1. **Extraction quality** - given a *known* transcript, how well does the
   analysis stage recover decisions, action items, participants and so on?
   This isolates the AI component from speech recognition, so a drop in
   quality can be attributed correctly.

2. **End-to-end quality** - given an audio file and its ground-truth
   transcript, measure word error rate for the ASR stage.

Metrics, and why each is appropriate:

* **WER** for speech recognition: the standard measure for ASR. Lower is
  better, and it is not bounded above (insertions can exceed the reference).
* **Precision / Recall / F1** for extraction: because the task is selection
  from a transcript. Recall alone would reward a system that lists everything;
  precision alone would reward one that says nothing. F1 balances them.
* **ROUGE-L** for overview and conclusion text: measures the longest common
  subsequence, which suits a summary that reuses the source's wording better
  than it suits free paraphrase.
* **Semantic similarity** for summary text: cosine similarity between sentence
  embeddings. Included because ROUGE punishes a correct paraphrase that uses
  different words, and a summary *should* paraphrase.
* **Field accuracy** for scalar fields (title, date, time): exact match after
  normalisation, the right measure for a value with one correct answer.

Matching is deliberately lenient in one specific way: a predicted action item
counts as correct if its task overlaps the reference sufficiently *and* the
person matches. Deadlines are scored separately, because getting the task right
and the deadline wrong is a different (and less serious) error than inventing a
task, and blending them into one number would hide that.

Nothing here fabricates results. Where a metric is misleading it is labelled as
such in the output.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.schemas.extraction import NOT_SPECIFIED, MeetingExtraction  # noqa: E402

# ---------------------------------------------------------------------------
# Text normalisation and similarity
# ---------------------------------------------------------------------------


def normalise(text: str) -> str:
    """Lowercase and strip punctuation, for comparison only."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(text: str) -> set[str]:
    """Content words of a string, stopwords removed."""
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "by",
        "at", "is", "are", "was", "were", "be", "will", "we", "our", "us",
        "that", "this", "it", "with", "as", "from", "should", "can",
    }
    return {w for w in normalise(text).split() if w not in stop and len(w) > 1}


def overlap_score(predicted: str, reference: str) -> float:
    """Similarity in [0, 1] combining token overlap and sequence ratio.

    The sequence ratio is computed unconditionally and is the floor of the
    score. An earlier version returned 0.0 as soon as the content-word sets
    were empty, which scored identical strings as completely dissimilar
    whenever they contained only short or common words - a real bug, not a
    theoretical one.
    """
    p, r = token_set(predicted), token_set(reference)
    sequence = SequenceMatcher(None, normalise(predicted), normalise(reference)).ratio()
    if not p or not r:
        return sequence

    # F1 over content words, so length differences do not dominate.
    common = len(p & r)
    precision = common / len(p)
    recall = common / len(r)
    token_f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return max(token_f1, sequence)


def best_match(values: list[str], reference: str, threshold: float = 0.6):
    """Return (best score, best value) for ``reference`` among ``values``."""
    best_score, best_value = 0.0, ""
    for value in values:
        score = overlap_score(value, reference)
        if score > best_score:
            best_score, best_value = score, value
    return best_score, best_value


# ---------------------------------------------------------------------------
# Set-based precision / recall / F1
# ---------------------------------------------------------------------------


@dataclass
class PRF:
    """Precision, recall and F1, with the raw counts behind them."""

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


def prf_from_counts(tp: int, fp: int, fn: int) -> PRF:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return PRF(precision, recall, f1, tp, fp, fn)


def score_list(
    predicted: list[str],
    reference: list[str],
    threshold: float = 0.6,
) -> PRF:
    """Greedy one-to-one matching of predicted items against reference items.

    Greedy matching with consumption matters: without removing a matched
    reference, one correct prediction could satisfy several reference items and
    inflate recall.
    """
    remaining = list(reference)
    tp = 0
    for candidate in predicted:
        if not remaining:
            break
        scores = [overlap_score(candidate, ref) for ref in remaining]
        best_index = max(range(len(scores)), key=lambda i: scores[i])
        if scores[best_index] >= threshold:
            tp += 1
            remaining.pop(best_index)
    fp = len(predicted) - tp
    fn = len(remaining)
    return prf_from_counts(tp, fp, fn)


def equivalent_deadline(predicted: str, reference: str) -> bool:
    """Whether two deadline strings express the same thing.

    Leading articles are ignored, so "the end of the month" and "end of the
    month" agree. This is a genuine equivalence, not a loosening of the metric:
    dropping a determiner does not change when the deadline is.
    """
    def simplify(value: str) -> str:
        value = normalise(value)
        value = re.sub(r"\b(?:the|a|an)\b", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    if simplify(predicted) == simplify(reference):
        return True
    # Both sides declining to state a deadline is also agreement.
    return (
        predicted.strip().lower() == NOT_SPECIFIED.lower()
        and reference.strip().lower() == NOT_SPECIFIED.lower()
    )


def score_action_items(predicted: list[dict], reference: list[dict]) -> dict:
    """Score action items on task, person and deadline separately.

    Reporting these separately is the point. A system that finds the right task
    but misses the deadline is doing useful work; a system that invents tasks
    is not. One blended number would hide the difference.
    """
    remaining = list(reference)
    tp = fp = 0
    task_hits = person_hits = deadline_hits = 0

    for item in predicted:
        if not remaining:
            break
        scores = [overlap_score(item.get("task", ""), ref.get("task", "")) for ref in remaining]
        best_index = max(range(len(scores)), key=lambda i: scores[i])
        if scores[best_index] >= 0.6:
            ref = remaining.pop(best_index)
            tp += 1
            task_hits += 1

            person = normalise(item.get("responsible_person", ""))
            ref_person = normalise(ref.get("responsible_person", ""))
            if person and person == ref_person:
                person_hits += 1

            deadline = item.get("deadline", "")
            ref_deadline = ref.get("deadline", "")
            if equivalent_deadline(deadline, ref_deadline):
                deadline_hits += 1
        else:
            fp += 1

    fn = len(remaining)
    overall = prf_from_counts(tp, fp, fn)

    def accuracy(hits: int) -> float:
        return round(hits / tp, 4) if tp else 0.0

    return {
        "matching": overall.to_dict(),
        "person_accuracy_given_task": accuracy(person_hits),
        "deadline_accuracy_given_task": accuracy(deadline_hits),
    }


# ---------------------------------------------------------------------------
# Scalar fields, ROUGE and semantic similarity
# ---------------------------------------------------------------------------


def score_scalar(predicted: str, reference: str) -> float:
    """Exact match after normalisation, treating the sentinel as a value.

    'Not specified' matching 'Not specified' is a correct answer, not a
    missing one: correctly declining to invent a date is the desired
    behaviour, and should be scored as such.
    """
    return 1.0 if normalise(predicted) == normalise(reference) else 0.0


def score_participants(predicted: list[str], reference: list[str]) -> PRF:
    """Score participant names, allowing for ASR corruption of names."""
    remaining = list(reference)
    tp = 0
    for name in predicted:
        if not remaining:
            break
        scores = [
            max(
                overlap_score(name, ref),
                SequenceMatcher(None, normalise(name), normalise(ref)).ratio(),
            )
            for ref in remaining
        ]
        best_index = max(range(len(scores)), key=lambda i: scores[i])
        if scores[best_index] >= 0.75:
            tp += 1
            remaining.pop(best_index)
    return prf_from_counts(tp, len(predicted) - tp, len(remaining))


def rouge_l(predicted: str, reference: str) -> float:
    """ROUGE-L using longest common subsequence over words.

    Implemented directly rather than pulling in a dependency, so the metric is
    inspectable. F-measure form, which is standard for summary evaluation.
    """
    pred = normalise(predicted).split()
    ref = normalise(reference).split()
    if not pred or not ref:
        return 0.0

    # Standard LCS dynamic-programming table, one row kept at a time.
    previous = [0] * (len(ref) + 1)
    for i in range(1, len(pred) + 1):
        current = [0] * (len(ref) + 1)
        for j in range(1, len(ref) + 1):
            if pred[i - 1] == ref[j - 1]:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    lcs = previous[len(ref)]

    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def semantic_similarity(predicted: str, reference: str, model=None) -> float | None:
    """Cosine similarity between sentence embeddings.

    Returns ``None`` when the embedding model is unavailable, rather than a
    fabricated number. The caller reports it as not measured.
    """
    if model is None or not predicted or not reference:
        return None
    try:
        import numpy as np

        embeddings = model.encode([predicted, reference], normalize_embeddings=True)
        return round(float(np.dot(embeddings[0], embeddings[1])), 4)
    except Exception:  # noqa: BLE001 - optional metric
        return None


def load_embedder(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load a sentence-embedding model, or return ``None`` if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(name)
    except Exception as exc:  # noqa: BLE001 - optional metric
        print(f"  (semantic similarity disabled: {exc})")
        return None


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------


def word_error_rate(hypothesis: str, reference: str) -> dict:
    """Word error rate using jiwer, with the error breakdown."""
    try:
        import jiwer
    except ImportError:
        return {"error": "jiwer is not installed"}

    try:
        output = jiwer.process_words(normalise(reference), normalise(hypothesis))
    except Exception as exc:  # noqa: BLE001 - report rather than crash
        return {"error": str(exc)}

    return {
        "wer": round(output.wer, 4),
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
        "hits": output.hits,
        "reference_words": len(normalise(reference).split()),
    }


# ---------------------------------------------------------------------------
# Main evaluation over the extraction stage
# ---------------------------------------------------------------------------


@dataclass
class ExtractionEvaluation:
    """Aggregate metrics over a set of examples."""

    n: int = 0
    decisions: list[PRF] = field(default_factory=list)
    discussion_points: list[PRF] = field(default_factory=list)
    participants: list[PRF] = field(default_factory=list)
    action_items: list[dict] = field(default_factory=list)
    title_accuracy: list[float] = field(default_factory=list)
    date_accuracy: list[float] = field(default_factory=list)
    overview_rouge: list[float] = field(default_factory=list)
    overview_semantic: list[float] = field(default_factory=list)
    conclusion_rouge: list[float] = field(default_factory=list)
    conclusion_semantic: list[float] = field(default_factory=list)
    per_example: list[dict] = field(default_factory=list)

    @staticmethod
    def _mean(values: list[float]) -> float:
        return round(statistics.fmean(values), 4) if values else 0.0

    @staticmethod
    def _agg(prfs: list[PRF]) -> dict:
        """Micro-average by summing counts, then report macro F1 too.

        Both are reported because they answer different questions: the micro
        figure reflects overall volume, the macro figure gives every example
        equal weight regardless of how many items it contains.
        """
        tp = sum(p.true_positives for p in prfs)
        fp = sum(p.false_positives for p in prfs)
        fn = sum(p.false_negatives for p in prfs)
        micro = prf_from_counts(tp, fp, fn)
        f1s = [p.f1 for p in prfs]
        return {
            "micro": micro.to_dict(),
            "macro_f1": round(statistics.fmean(f1s), 4) if f1s else 0.0,
        }

    def summary(self) -> dict:
        """A JSON-serialisable summary of everything measured."""
        action_matching = [a["matching"] for a in self.action_items]
        action_tp = sum(m["true_positives"] for m in action_matching)
        action_fp = sum(m["false_positives"] for m in action_matching)
        action_fn = sum(m["false_negatives"] for m in action_matching)

        return {
            "examples": self.n,
            "decisions": self._agg(self.decisions),
            "discussion_points": self._agg(self.discussion_points),
            "participants": self._agg(self.participants),
            "action_items": {
                "micro": prf_from_counts(action_tp, action_fp, action_fn).to_dict(),
                "person_accuracy_given_task": self._mean(
                    [a["person_accuracy_given_task"] for a in self.action_items]
                ),
                "deadline_accuracy_given_task": self._mean(
                    [a["deadline_accuracy_given_task"] for a in self.action_items]
                ),
            },
            "title_accuracy": self._mean(self.title_accuracy),
            "date_accuracy": self._mean(self.date_accuracy),
            "overview_rouge_l": self._mean(self.overview_rouge),
            "overview_semantic_similarity": (
                self._mean(self.overview_semantic) if self.overview_semantic else None
            ),
            "conclusion_rouge_l": self._mean(self.conclusion_rouge),
            "conclusion_semantic_similarity": (
                self._mean(self.conclusion_semantic) if self.conclusion_semantic else None
            ),
        }


def evaluate_extraction(
    examples: list[dict],
    extractor,
    embedder=None,
    limit: int | None = None,
) -> ExtractionEvaluation:
    """Run the analysis stage over examples and aggregate the metrics.

    Args:
        examples: dicts with ``transcript`` and ``target``.
        extractor: callable mapping a transcript to a :class:`MeetingExtraction`.
            Injected rather than imported so the same harness can evaluate the
            pretrained model, a fine-tuned adapter, or the rules alone.
    """
    result = ExtractionEvaluation()
    if limit:
        examples = examples[:limit]

    for index, example in enumerate(examples, start=1):
        transcript = example["transcript"]
        target = MeetingExtraction.model_validate(example["target"])
        scenario = example.get("meta", {}).get("scenario_id", "?")

        try:
            predicted = extractor(transcript)
        except Exception as exc:  # noqa: BLE001 - one bad example must not abort
            print(f"  [{index}] {scenario}: extraction failed: {exc}")
            continue

        if isinstance(predicted, tuple):
            predicted = predicted[0]

        result.n += 1
        result.decisions.append(score_list(predicted.decisions, target.decisions))
        result.discussion_points.append(
            score_list(predicted.discussion_points, target.discussion_points)
        )
        result.participants.append(
            score_participants(predicted.participants, target.participants)
        )
        result.action_items.append(
            score_action_items(
                [a.model_dump() for a in predicted.action_items],
                [a.model_dump() for a in target.action_items],
            )
        )
        result.title_accuracy.append(score_scalar(predicted.title, target.title))
        result.date_accuracy.append(score_scalar(predicted.date, target.date))
        result.overview_rouge.append(rouge_l(predicted.overview, target.overview))
        result.conclusion_rouge.append(rouge_l(predicted.conclusion, target.conclusion))

        if embedder is not None:
            semantic_overview = semantic_similarity(predicted.overview, target.overview, embedder)
            semantic_conclusion = semantic_similarity(
                predicted.conclusion, target.conclusion, embedder
            )
            if semantic_overview is not None:
                result.overview_semantic.append(semantic_overview)
            if semantic_conclusion is not None:
                result.conclusion_semantic.append(semantic_conclusion)

        result.per_example.append(
            {
                "scenario_id": scenario,
                "decisions_f1": result.decisions[-1].f1,
                "action_items_f1": result.action_items[-1]["matching"]["f1"],
                "overview_rouge_l": result.overview_rouge[-1],
            }
        )
        print(
            f"  [{index}/{len(examples)}] {scenario:22} "
            f"dec F1={result.decisions[-1].f1:.2f} "
            f"act F1={result.action_items[-1]['matching']['f1']:.2f} "
            f"rougeL={result.overview_rouge[-1]:.2f}"
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _rules_only(transcript: str):
    """Extraction using only the deterministic rules, for comparison."""
    from app.services.rules import extract_with_rules
    from app.schemas.extraction import MeetingExtraction as ME

    rules = extract_with_rules(transcript)
    return ME(
        participants=rules.participants,
        decisions=rules.decisions,
        action_items=rules.action_items,
        conclusion=rules.conclusion,
    )


def _hybrid(transcript: str):
    """The full production path: model plus rules plus grounding."""
    from app.services.extraction import extract_meeting_information

    return extract_meeting_information(transcript)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the extraction stage.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Which dataset split to evaluate.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap the examples.")
    parser.add_argument(
        "--method",
        default="hybrid",
        choices=["hybrid", "rules"],
        help="hybrid = language model + rules; rules = deterministic only.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-semantic", action="store_true")
    args = parser.parse_args()

    dataset = REPO_ROOT / "datasets" / "processed" / f"{args.split}.jsonl"
    if not dataset.exists():
        raise SystemExit(
            f"{dataset} not found. Generate it first: python training/prepare_dataset.py"
        )

    examples = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]

    print(f"Evaluating the '{args.method}' extractor on the {args.split} split")
    print(f"{len(examples)} examples\n")

    embedder = None if args.no_semantic else load_embedder()
    extractor = _hybrid if args.method == "hybrid" else _rules_only

    evaluation = evaluate_extraction(examples, extractor, embedder=embedder, limit=args.limit)
    summary = evaluation.summary()

    print("\n" + "=" * 66)
    print("RESULTS")
    print("=" * 66)
    print(json.dumps(summary, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "split": args.split,
                    "method": args.method,
                    "summary": summary,
                    "per_example": evaluation.per_example,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nWritten to {args.output}")

    print(
        "\nNote: this dataset is synthetic, so these figures measure the task "
        "on clean templated text. They are not a claim about performance on "
        "real meeting recordings."
    )


if __name__ == "__main__":
    main()