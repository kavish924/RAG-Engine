"""
Runs the full eval suite across all three chunking strategies
(fixed_size, recursive_structure, semantic) and generates a comparison
report showing which strategy wins on which metric. This data drives
the architecture decisions and gives concrete numbers for interviews.

Usage: python eval/chunking_comparison.py
"""


def compare_chunking_strategies() -> dict:
    """
    Returns: {
      "fixed_size":            {"correctness": .., "faithfulness": .., ...},
      "recursive_structure":   {...},
      "semantic":              {...},
    }
    """
    raise NotImplementedError


if __name__ == "__main__":
    report = compare_chunking_strategies()
    print(report)
