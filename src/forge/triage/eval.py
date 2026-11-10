"""Score failure triage against labelled cases.

    python -m forge.triage.eval

Each case is the evidence from one failed attempt plus the category a person
assigned. Any diagnoser with the same signature as `classify` can be scored.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from forge.triage.evidence import Evidence, Finding, classify

CASES = Path(__file__).parent / "cases" / "cases.json"


def load_cases() -> list[tuple[str, str, Evidence]]:
    return [
        (c["id"], c["expected"], Evidence.from_dict(c["evidence"]))
        for c in json.loads(CASES.read_text())
    ]


def evaluate(
    diagnoser: Callable[[Evidence], Finding | None],
) -> tuple[int, list[tuple[str, str, str]]]:
    """Returns the number correct and (case, expected, got) for each miss."""
    correct, misses = 0, []
    for case_id, expected, ev in load_cases():
        finding = diagnoser(ev)
        got = finding.category if finding else "(no diagnosis)"
        if got == expected:
            correct += 1
        else:
            misses.append((case_id, expected, got))
    return correct, misses


def main() -> None:
    correct, misses = evaluate(classify)
    total = len(load_cases())
    print(f"rules: {correct}/{total} correct ({correct / total:.0%})")
    for case_id, expected, got in misses:
        print(f"  {case_id}: expected {expected}, got {got}")


if __name__ == "__main__":
    main()
