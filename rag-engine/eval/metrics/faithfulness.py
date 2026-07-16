

import json
import re
import requests

from app.config import settings



def score_faithfulness(
    answer: str,
    retrieved_chunks: list[dict],
) -> float:
    """
    Returns a score between 0 and 1 representing how well the generated
    answer is grounded in the retrieved context.
    """

    claims = _extract_claims(answer)

    if not claims:
        return 1.0

    context = "\n\n".join(
        chunk["text"]
        for chunk in retrieved_chunks
    )

    verdicts = _judge_claims_grounded(
        claims,
        context,
    )

    if not verdicts:
        return 0.0

    return sum(verdicts) / len(verdicts)




def _extract_claims(answer: str) -> list[str]:

    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer.strip(),
    )

    cleaned = [
        re.sub(
            r"\[\d+(?:\s*,\s*\d+)*\]",
            "",
            s,
        ).strip()
        for s in sentences
    ]

    return [
        s
        for s in cleaned
        if len(s) > 5
    ]




def _judge_claims_grounded(
    claims: list[str],
    context: str,
) -> list[bool]:

    numbered_claims = "\n".join(
        f"[{i}] {claim}"
        for i, claim in enumerate(claims, start=1)
    )

    prompt = f"""
Context

{context}

Claims

{numbered_claims}

For every numbered claim determine whether it is directly supported by
the supplied context.

Use ONLY the supplied context.

Return ONLY a JSON array of booleans.

Example:

[true,false,true]
"""

    raw = _call_llm(prompt)

    return _parse_boolean_array(
        raw,
        len(claims),
    )



def _call_llm(prompt: str) -> str:

    provider = settings.llm_provider.lower()

    if provider == "ollama":
        return _call_ollama(prompt)

    elif provider == "anthropic":
        return _call_anthropic(prompt)

    elif provider == "openai":
        return _call_openai(prompt)

    raise ValueError(provider)



def _call_ollama(prompt: str) -> str:

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



def _call_anthropic(prompt: str) -> str:

    from anthropic import Anthropic

    client = Anthropic(
        api_key=settings.anthropic_api_key,
    )

    response = client.messages.create(
        model=settings.generation_model,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.content[0].text




def _call_openai(prompt: str) -> str:

    from openai import OpenAI

    client = OpenAI(
        api_key=settings.openai_api_key,
    )

    response = client.chat.completions.create(
        model=settings.generation_model,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content




def _parse_boolean_array(
    raw: str,
    expected_len: int,
) -> list[bool]:

    match = re.search(
        r"\[[^\]]*\]",
        raw,
        re.IGNORECASE,
    )

    if not match:
        return [False] * expected_len

    try:
        verdicts = json.loads(
            match.group().lower()
        )
    except json.JSONDecodeError:
        return [False] * expected_len

    if len(verdicts) != expected_len:

        verdicts = (
            verdicts
            + [False] * expected_len
        )[:expected_len]

    return [bool(v) for v in verdicts]