"""
Citation accuracy: of the citations the model produced, what fraction
were verified as actually supporting their attached claim
(reuses app.generation.citation_verifier)?
"""


def score_citation_accuracy(verified_citations: list[dict]) -> float:
    raise NotImplementedError
