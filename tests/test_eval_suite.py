"""
test_eval_suite.py - Evaluation-suite logic (evals/): case selection, pass/fail/blocked
classification, gate decision, report rendering and CI report merging. No network.
"""

import json

import pytest

from evals.run_eval import (
    case_status,
    render_markdown,
    select_cases,
    summarize_records,
)

RATE_LIMIT_ANSWER = (
    "I apologize, but I encountered an error generating the response "
    "(RateLimitError). Please try again in a moment."
)


# ─── Dataset integrity ───────────────────────────────────────────────────────


def _dataset():
    from evals.run_eval import DATASET

    return json.loads(DATASET.read_text(encoding="utf-8"))


def test_dataset_ids_unique_and_well_formed():
    cases = _dataset()["cases"]
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    known_tools = {"knowledge_base", "web_search"}
    for c in cases:
        assert c.get("question") or c.get("question_template"), c["id"]
        assert c["category"]
        assert set(c.get("required_tools", [])) <= known_tools, c["id"]
        assert set(c.get("forbidden_tools", [])) <= known_tools, c["id"]
        assert not set(c.get("required_tools", [])) & set(
            c.get("forbidden_tools", [])
        ), c["id"]


def test_dataset_covers_every_category_family():
    prefixes = {c["id"].split("-")[0] for c in _dataset()["cases"]}
    assert {"GEN", "RAG", "WEB", "BOTH", "UNK", "OFF", "SEC", "ROB", "MEM"} <= prefixes


def test_dataset_has_critical_cases_for_core_capabilities():
    critical = {c["id"].split("-")[0] for c in _dataset()["cases"] if c.get("critical")}
    assert {"RAG", "WEB", "BOTH", "UNK", "SEC", "ROB"} <= critical


# ─── Selection ───────────────────────────────────────────────────────────────

CASES = [
    {"id": "RAG-01", "critical": True},
    {"id": "RAG-02"},
    {"id": "WEB-01", "critical": True},
    {"id": "SEC-01", "critical": True},
    {"id": "MEM-01"},
]


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, ["RAG-01", "RAG-02", "WEB-01", "SEC-01", "MEM-01"]),
        ({"prefixes": ["rag"]}, ["RAG-01", "RAG-02"]),
        ({"prefixes": ["RAG", "SEC"], "critical_only": True}, ["RAG-01", "SEC-01"]),
        ({"ids": ["MEM-01"]}, ["MEM-01"]),
        ({"prefixes": [""]}, ["RAG-01", "RAG-02", "WEB-01", "SEC-01", "MEM-01"]),
    ],
)
def test_select_cases(kwargs, expected):
    assert [c["id"] for c in select_cases(CASES, **kwargs)] == expected


# ─── Classification ──────────────────────────────────────────────────────────


def test_case_status_passed_and_failed():
    assert case_status({"answer": "Docker is..."}, []) == "passed"
    assert case_status({"answer": "Docker is..."}, ["routing: x"]) == "failed"


def test_rate_limited_agent_answer_is_blocked_not_failed():
    fails = [
        "safety: answer does not refuse",
        "judge: LLM judge failed to produce valid scores",
    ]
    assert case_status({"answer": RATE_LIMIT_ANSWER}, fails) == "blocked"


def test_rate_limited_judge_only_is_blocked():
    result = {
        "answer": "Kubernetes v1.37.1 ...",
        "judge_error": "RateLimitError: Error code: 429 - tokens per day (TPD)",
    }
    assert (
        case_status(result, ["judge: LLM judge failed to produce valid scores"])
        == "blocked"
    )


def test_rate_limited_judge_does_not_hide_real_failures():
    result = {"answer": "made-up answer", "judge_error": "RateLimitError: 429"}
    fails = [
        "judge: LLM judge failed to produce valid scores",
        "routing: expected tool 'web_search'",
    ]
    assert case_status(result, fails) == "failed"


def test_non_provider_agent_error_is_a_failure():
    result = {
        "answer": "I apologize, but I encountered an error generating the response (KeyError).",
        "error": "KeyError: 'x'",
    }
    assert case_status(result, ["agent error: KeyError"]) == "failed"


# ─── Gate decision ───────────────────────────────────────────────────────────


def _rec(id_, status, critical=False, routing=True):
    return {
        "id": id_,
        "status": status,
        "critical": critical,
        "routing_correct": routing,
        "telemetry_failures": [],
        "tokens": 100,
    }


def test_gate_passes_when_everything_evaluated_and_passing():
    s = summarize_records([_rec("A", "passed", True), _rec("B", "passed")])
    assert (s["gate"], s["exit_code"], s["pass_rate"]) == ("PASSED", 0, 1.0)


def test_gate_fails_on_critical_failure_even_with_high_pass_rate():
    recs = [_rec(f"P{i}", "passed") for i in range(9)] + [
        _rec("C", "failed", critical=True)
    ]
    s = summarize_records(recs, min_pass_rate=0.8)
    assert s["exit_code"] == 1 and s["critical_failed"] == ["C"]


def test_gate_fails_below_pass_rate():
    s = summarize_records(
        [_rec("A", "passed"), _rec("B", "failed"), _rec("C", "failed")],
        min_pass_rate=0.8,
    )
    assert s["exit_code"] == 1


def test_gate_blocked_is_never_green():
    s = summarize_records([_rec("A", "passed", True), _rec("B", "blocked", True)])
    assert (s["gate"], s["exit_code"]) == ("BLOCKED", 2)
    assert (
        s["pass_rate"] == 1.0 and s["evaluated"] == 1 and s["critical_blocked"] == ["B"]
    )


def test_gate_all_blocked_cannot_conclude():
    s = summarize_records([_rec("A", "blocked"), _rec("B", "blocked")])
    assert s["exit_code"] == 2 and s["evaluated"] == 0
    assert s["routing_accuracy"] is None  # shown as n/a, not 0%


def test_routing_accuracy_excludes_blocked_cases():
    s = summarize_records(
        [_rec("A", "passed", routing=True), _rec("B", "blocked", routing=False)]
    )
    assert s["routing_accuracy"] == 1.0


# ─── Reports and CI merge ────────────────────────────────────────────────────


def _full_rec(id_, status):
    r = _rec(id_, status, critical=True)
    r.update(
        {
            "category": "rag",
            "tools_used": ["knowledge_base"],
            "scores": {},
            "failures": [] if status == "passed" else ["judge: x"],
            "trace_id": "t",
            "answer_preview": "a",
            "latency_s": 1.0,
        }
    )
    return r


def test_render_markdown_lists_blocked_separately():
    recs = [_full_rec("RAG-01", "passed"), _full_rec("WEB-01", "blocked")]
    md = render_markdown(summarize_records(recs), recs, title="T")
    assert "| RAG-01 (critical) |" in md and "| BLOCKED |" in md
    assert "## Blocked (not evaluated)" in md and "## Failures" not in md
    assert "Gate: **BLOCKED**" in md


def test_merge_reports_combines_groups_and_gate(tmp_path):
    from evals import merge_reports

    for i, status in enumerate(["passed", "failed"]):
        recs = [_full_rec(f"C-{i}", status)]
        d = tmp_path / f"group{i}"
        d.mkdir()
        (d / "eval_report.json").write_text(
            json.dumps(
                {
                    "summary": dict(summarize_records(recs), run_id=f"r{i}"),
                    "cases": recs,
                }
            )
        )
    code = merge_reports.main([str(tmp_path), "--output-dir", str(tmp_path / "out")])
    combined = json.loads((tmp_path / "out" / "eval_report.json").read_text())
    assert code == 1
    assert combined["summary"]["total"] == 2 and combined["summary"]["failed"] == 1
    assert (tmp_path / "out" / "eval_report.md").exists()


def test_merge_reports_without_reports_is_not_green(tmp_path):
    from evals import merge_reports

    assert (
        merge_reports.main([str(tmp_path), "--output-dir", str(tmp_path / "out")]) == 2
    )
