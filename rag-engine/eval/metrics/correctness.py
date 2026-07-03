"""
Answer correctness: LLM-as-judge compares the generated answer against
the golden answer for each test case and scores agreement.
"""


def score_correctness(generated_answer: str, golden_answer: str, question: str) -> float:
    raise NotImplementedError
