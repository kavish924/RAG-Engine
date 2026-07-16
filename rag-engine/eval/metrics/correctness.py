
import re
import requests

from app.config import settings




def score_correctness(
    generated_answer: str,
    golden_answer: str,
    question: str,
) -> float:

    if golden_answer.strip().upper() == "NOT_IN_CORPUS":
        return _score_refusal_correctness(generated_answer)

    if golden_answer.strip().upper().startswith("AMBIGUOUS"):
        return _score_ambiguous_handling(
            generated_answer,
            golden_answer,
        )

    return _score_standard_correctness(
        generated_answer,
        golden_answer,
        question,
    )




def _score_standard_correctness(
    generated_answer: str,
    golden_answer: str,
    question: str,
) -> float:

    prompt = f"""
You are evaluating a Retrieval-Augmented Generation (RAG) system.

Question:
{question}

Golden Answer:
{golden_answer}

Generated Answer:
{generated_answer}

Rate how correct the generated answer is.

Scoring:
10 = Completely correct
8 = Mostly correct
5 = Partially correct
2 = Mostly incorrect
0 = Completely incorrect

Reply ONLY with a single number between 0 and 10.
"""

    score = _parse_single_score(_call_llm(prompt))

    return max(0.0, min(score / 10.0, 1.0))



def _score_refusal_correctness(
    generated_answer: str,
) -> float:

    prompt = f"""
Should the following answer be considered a correct refusal?

Answer:
{generated_answer}

Reply ONLY with:

yes

or

no
"""

    response = _call_llm(prompt).strip().lower()

    return 1.0 if response.startswith("yes") else 0.0



def _score_ambiguous_handling(
    generated_answer: str,
    golden_answer: str,
) -> float:

    prompt = f"""
The reference says this question is ambiguous.

Reference:
{golden_answer}

Generated answer:
{generated_answer}

Did the generated answer acknowledge ambiguity?

Reply ONLY with:

yes

or

no
"""

    response = _call_llm(prompt).strip().lower()

    return 1.0 if response.startswith("yes") else 0.0




def _call_llm(prompt: str) -> str:

    provider = settings.llm_provider.lower()

    if provider == "ollama":
        return _call_ollama(prompt)

    if provider == "openai":
        return _call_openai(prompt)

    if provider == "anthropic":
        return _call_anthropic(prompt)

    raise ValueError(f"Unsupported provider: {provider}")



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



def _call_openai(prompt: str) -> str:

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



def _call_anthropic(prompt: str) -> str:

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




def _parse_single_score(raw_response: str) -> float:

    match = re.search(r"\d+(\.\d+)?", raw_response)

    if not match:
        return 0.0

    return float(match.group())