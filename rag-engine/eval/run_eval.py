"""
Runs the full eval suite (correctness, faithfulness, retrieval relevance,
citation accuracy) over eval/golden_dataset.jsonl against the current
pipeline. Intended to run after every pipeline change as a regression test.

Usage: python eval/run_eval.py
"""
import json


def load_golden_dataset(path: str = "eval/golden_dataset.jsonl") -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip() and not line.startswith("#")]


def run_eval_suite(dataset_path: str = "eval/golden_dataset.jsonl") -> dict:
    """
    For each test case: run retrieval + generation + citation verification,
    score against all four metrics, and aggregate into a summary report.
    """
    raise NotImplementedError


if __name__ == "__main__":
    results = run_eval_suite()
    print(json.dumps(results, indent=2))
