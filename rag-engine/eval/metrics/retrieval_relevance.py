"""
Retrieval relevance: were the chunks that actually contain the golden
answer present in the retrieved set (e.g. recall@k against source_docs)?
"""


def score_retrieval_relevance(retrieved_chunks: list[dict], expected_source_docs: list[str]) -> float:
    raise NotImplementedError
