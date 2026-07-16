

import re
import requests

from app.config import settings

_RETRIEVAL_WEIGHT = 0.4
_CITATION_WEIGHT = 0.4
_COMPLETENESS_WEIGHT = 0.2


def score_confidence(
    chunks: list[dict],
    verified_citations: list[dict],
    answer: str,
    question: str,
) -> dict:

    retrieval = _score_retrieval_confidence(chunks)
    citation = _score_citation_coverage(verified_citations)
    completeness = _score_answer_completeness(question, answer)

    composite = (
        _RETRIEVAL_WEIGHT * retrieval
        + _CITATION_WEIGHT * citation
        + _COMPLETENESS_WEIGHT * completeness
    )

    return {
        "retrieval_confidence": round(retrieval, 3),
        "citation_coverage": round(citation, 3),
        "answer_completeness": round(completeness, 3),
        "composite": round(composite, 3),
    }



def _score_retrieval_confidence(chunks):

    if not chunks:
        return 0.0

    scores = []

    for chunk in chunks:

        if "relevance_score" in chunk:

            score = chunk["relevance_score"]


            if score > 10:
                score = 10

            if score < 0:
                score = 0

            scores.append(score / 10)

        elif "score" in chunk:

            scores.append(
                max(
                    0,
                    min(
                        1,
                        chunk["score"],
                    ),
                )
            )

        elif "fused_score" in chunk:

            scores.append(0.5)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def _score_citation_coverage(citations):

    if not citations:
        return 0.0

    supported = sum(
        1
        for c in citations
        if c["supported"]
    )

    return supported / len(citations)



def _score_answer_completeness(question: str, answer: str) -> float:

    if not answer:
        return 0.0

    answer = answer.strip()

    if answer.lower().startswith(
        "i don't have enough"
    ):
        return 0.0

    words = len(answer.split())

    if words >= 80:
        return 1.0

    if words >= 50:
        return 0.9

    if words >= 30:
        return 0.8

    if words >= 15:
        return 0.6

    if words >= 8:
        return 0.4

    return 0.2


def _call_llm(prompt):

    provider = settings.llm_provider.lower()

    if provider == "ollama":
        return _call_ollama(prompt)

    elif provider == "anthropic":
        return _call_anthropic(prompt)

    elif provider == "openai":
        return _call_openai(prompt)

    raise ValueError(provider)




def _call_ollama(prompt):

    response = requests.post(
        f"{settings.ollama_base_url}/api/chat",
        json={
            "model": settings.ollama_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
        },
        timeout=300,
    )

    response.raise_for_status()

    return response.json()["message"]["content"]




def _call_anthropic(prompt):

    from anthropic import Anthropic

    client = Anthropic(
        api_key=settings.anthropic_api_key,
    )

    response = client.messages.create(
        model=settings.generation_model,
        max_tokens=20,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.content[0].text



def _call_openai(prompt):

    from openai import OpenAI

    client = OpenAI(
        api_key=settings.openai_api_key,
    )

    response = client.chat.completions.create(
        model=settings.generation_model,
        max_tokens=20,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content



def _parse_score(raw):

    match = re.search(
        r"\d+(\.\d+)?",
        raw,
    )

    if not match:
        return 0.0

    return float(match.group())