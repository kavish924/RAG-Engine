"""
Faithfulness: are all claims in the generated answer actually grounded
in the retrieved context, regardless of whether they were cited?
"""


def score_faithfulness(answer: str, retrieved_chunks: list[dict]) -> float:
    raise NotImplementedError
