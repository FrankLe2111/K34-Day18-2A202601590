from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import os
from dataclasses import dataclass

from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _empty_evaluation() -> dict:
    return {**{metric: 0.0 for metric in METRICS}, "per_question": []}


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS and return aggregate plus per-question metrics."""
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("questions, answers, contexts, and ground_truths must have equal lengths")
    if not questions:
        return _empty_evaluation()

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        frame = result.to_pandas()
        per_question = []
        for index, row in frame.iterrows():
            per_question.append(EvalResult(
                question=str(row.get("question", questions[index])),
                answer=str(row.get("answer", answers[index])),
                contexts=list(row.get("contexts", contexts[index])),
                ground_truth=str(row.get("ground_truth", ground_truths[index])),
                **{metric: float(row.get(metric, 0.0) or 0.0) for metric in METRICS},
            ))
        if not per_question:
            return _empty_evaluation()
        return {
            **{
                metric: sum(getattr(item, metric) for item in per_question) / len(per_question)
                for metric in METRICS
            },
            "per_question": per_question,
        }
    except Exception as error:
        print(f"  ⚠️  RAGAS evaluation failed: {error}")
        return _empty_evaluation()


def _value(result, metric: str):
    return result.get(metric, 0.0) if isinstance(result, dict) else getattr(result, metric, 0.0)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze the lowest-scoring questions using a small diagnostic tree."""
    if bottom_n <= 0:
        return []
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer does not match question", "Improve prompt template"),
    }
    failures = []
    for result in eval_results:
        scores = {metric: float(_value(result, metric)) for metric in METRICS}
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        failures.append({
            "question": _value(result, "question"),
            "worst_metric": worst_metric,
            "score": sum(scores.values()) / len(scores),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    failures.sort(key=lambda item: item["score"])
    return failures[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation results to JSON."""
    report = {
        "aggregate": {metric: float(results.get(metric, 0.0)) for metric in METRICS},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, default=lambda value: value.__dict__)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    print(f"Loaded {len(load_test_set())} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
