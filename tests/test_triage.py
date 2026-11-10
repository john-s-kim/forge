from forge.triage.eval import evaluate, load_cases
from forge.triage.evidence import CATEGORIES, Evidence, _scale_flag, classify


def test_cases_are_well_formed():
    cases = load_cases()
    assert len({case_id for case_id, _, _ in cases}) == len(cases) >= 20
    assert all(expected in CATEGORIES for _, expected, _ in cases)


def test_rules_baseline_does_not_regress():
    correct, misses = evaluate(classify)
    assert correct >= 17, misses


def test_scale_flag_handles_both_argument_styles():
    assert _scale_flag(["t.py", "--batch-size", "64"], ("--batch-size",), 0.5, True) == [
        "t.py", "--batch-size", "32",
    ]  # fmt: skip
    assert _scale_flag(["t.py", "--lr=0.1"], ("--lr",), 0.1, False) == ["t.py", "--lr=0.01"]
    assert _scale_flag(["t.py", "--batch-size", "1"], ("--batch-size",), 0.5, True)[-1] == "1"
    assert _scale_flag(["t.py"], ("--lr",), 0.1, False) is None
    assert _scale_flag(["t.py", "--lr", "auto"], ("--lr",), 0.1, False) is None


def test_divergence_outranks_the_crash_it_causes():
    ev = Evidence(
        spec={"command": ["t.py", "--lr", "1.0"]},
        exit_code=1,
        logs=["Traceback (most recent call last):", "AssertionError"],
        metrics={"loss": [(0, 2.0), (1, None)]},
    )
    f = classify(ev)
    assert f.category == "diverged" and f.proposed_patch == {"command": ["t.py", "--lr", "0.1"]}
