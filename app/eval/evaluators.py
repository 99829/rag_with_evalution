"""
Custom evaluators for the LangSmith eval loop.
"""
import re
from typing import Dict, List, Tuple


def extract_citations(answer: str) -> List[Tuple[str, str]]:
    return re.findall(r"[\[【]([^\]】,]+),\s*p\.(\d+)[\]】]", answer)


def citation_accuracy_evaluator(run, example) -> Dict:
    """
    Custom LangSmith evaluator: precision/recall (F1) of cited sources
    against the golden dataset's expected_sources.
    """
    answer = run.outputs.get("answer", "")
    expected_sources = set(
        (doc, str(page)) for doc, page in example.outputs.get("expected_sources", [])
    )
    cited_sources = set(extract_citations(answer))

    if not cited_sources and not expected_sources:
        score = 1.0
    elif not cited_sources:
        score = 0.0
    else:
        correct = cited_sources & expected_sources
        precision = len(correct) / len(cited_sources) if cited_sources else 0
        recall = len(correct) / len(expected_sources) if expected_sources else 0
        score = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return {
        "key": "citation_accuracy",
        "score": score,
        "comment": f"cited={cited_sources}, expected={expected_sources}",
    }


def citations_grounded_evaluator(run, example) -> Dict:
    """
    Uses the graph's own citations_verified flag as a pass/fail
    groundedness signal — did the model cite anything NOT retrieved?
    """
    verified = run.outputs.get("citations_verified", False)
    flagged = run.outputs.get("flagged_citations", [])
    return {
        "key": "citations_grounded",
        "score": 1.0 if verified else 0.0,
        "comment": f"flagged: {flagged}" if flagged else "all citations grounded",
    }