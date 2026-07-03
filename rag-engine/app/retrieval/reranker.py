"""
Cross-encoder / LLM-as-judge reranker. Takes the top-N fused candidates
and scores each chunk's relevance to the actual question, keeping the
top `keep_top_n`. This second pass meaningfully improves precision over
fusion alone, because RRF only knows rank position — it doesn't actually
read the chunk against the question.

Uses an LLM-as-judge by default (no extra ML dependency required). Swap
in a real cross-encoder (e.g. sentence-transformers CrossEncoder) later
if latency/cost becomes a concern.
"""
import json
import re

from app.config import settings


def rerank(query: str, candidates: list[dict], keep_top_n: int = 5) -> list[dict]:
    """
    candidates: list of dicts with at least "id", "text".
    Returns the top `keep_top_n` candidates, each with an added
    "relevance_score" (0-10, from the judge), sorted descending.
    """
    if not candidates:
        return []

    scores = _score_candidates_llm_judge(query, candidates)

    for c, s in zip(candidates, scores):
        c["relevance_score"] = s

    ranked = sorted(candidates, key=lambda c: c["relevance_score"], reverse=True)
    return ranked[:keep_top_n]


def _score_candidates_llm_judge(query: str, candidates: list[dict]) -> list[float]:
    numbered_chunks = "\n\n".join(
        f"[{i}] {c['text'][:500]}" for i, c in enumerate(candidates)
    )

    prompt = f"""You are scoring how relevant each chunk is to a question, on a 0-10 scale
(0 = completely irrelevant, 10 = directly and fully answers the question).

Question: {query}

Chunks:
{numbered_chunks}

Respond ONLY with a JSON array of {len(candidates)} numbers, in order, e.g. [7, 2, 9, 0, 5]
No other text."""

    raw_response = _call_llm(prompt)
    scores = _parse_score_array(raw_response, expected_len=len(candidates))
    return scores


def _call_llm(prompt: str) -> str:
    if settings.llm_provider == "anthropic":
        from anthropic import Anthropic

        client = Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=settings.generation_model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    else:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.generation_model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


def _parse_score_array(raw_response: str, expected_len: int) -> list[float]:
    match = re.search(r"\[[\d\.,\s]+\]", raw_response)
    if not match:
        # fail safe: treat everything as equally (ir)relevant rather than crash
        return [0.0] * expected_len
    try:
        scores = json.loads(match.group())
    except json.JSONDecodeError:
        return [0.0] * expected_len
    if len(scores) != expected_len:
        scores = (scores + [0.0] * expected_len)[:expected_len]
    return [float(s) for s in scores]