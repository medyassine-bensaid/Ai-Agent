"""
test_scripts.py - CI helper scripts (scripts/junit_summary.py) and the healthcheck.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


junit_summary = _load("junit_summary", "scripts/junit_summary.py")

PASSING = """<?xml version="1.0"?>
<testsuites><testsuite name="pytest" tests="3" failures="0" errors="0" skipped="1" time="1.5">
<testcase classname="tests.test_a" name="test_one"/>
<testcase classname="tests.test_a" name="test_two"/>
<testcase classname="tests.test_a" name="test_three"><skipped/></testcase>
</testsuite></testsuites>"""

FAILING = """<?xml version="1.0"?>
<testsuite name="pytest" tests="2" failures="1" errors="0" skipped="0" time="0.5">
<testcase classname="tests.test_b" name="test_ok"/>
<testcase classname="tests.test_b" name="test_bad"><failure message="boom"/></testcase>
</testsuite>"""


def test_suite_counts_parses_testsuites_wrapper(tmp_path):
    f = tmp_path / "junit-agent-core.xml"
    f.write_text(PASSING)
    c = junit_summary.suite_counts(f)
    assert (c["tests"], c["passed"], c["skipped"], c["failures"]) == (3, 2, 1, 0)


def test_summary_exit_code_and_failed_names(tmp_path, capsys):
    (tmp_path / "junit-agent-core.xml").write_text(PASSING)
    (tmp_path / "junit-monitoring.xml").write_text(FAILING)
    out = tmp_path / "summary.md"
    assert junit_summary.main([str(tmp_path), "--output", str(out)]) == 1
    md = out.read_text()
    assert "| agent-core | 3 | 2 | 0 | 0 | 1 |" in md
    assert "| **Total** | **5** | **3** | **1** |" in md
    assert "`tests.test_b::test_bad`" in md


def test_summary_all_green(tmp_path):
    (tmp_path / "junit-x.xml").write_text(PASSING)
    assert junit_summary.main([str(tmp_path)]) == 0


def test_summary_missing_results_is_a_failure(tmp_path):
    assert junit_summary.main([str(tmp_path)]) == 1


def test_healthcheck_fails_without_required_keys(monkeypatch):
    healthcheck = _load("healthcheck", "scripts/healthcheck.py")
    import config.settings as s

    monkeypatch.setattr(s.Settings, "GROQ_API_KEY", None)
    with pytest.raises(SystemExit) as exc:
        healthcheck.check()
    assert exc.value.code == 1


def test_healthcheck_passes_with_keys():
    healthcheck = _load("healthcheck", "scripts/healthcheck.py")
    with pytest.raises(SystemExit) as exc:
        healthcheck.check()
    assert exc.value.code == 0


audit_summary = _load("audit_summary", "scripts/audit_summary.py")

AUDIT = {
    "dependencies": [
        {"name": "requests", "version": "2.34.2", "vulns": []},
        {
            "name": "langchain-core",
            "version": "0.3.63",
            "vulns": [
                {"id": "PYSEC-1", "fix_versions": ["0.3.81"]},
                {"id": "PYSEC-2", "fix_versions": ["0.3.85"]},
            ],
        },
        {
            "name": "chromadb",
            "version": "1.5.9",
            "vulns": [{"id": "PYSEC-3", "fix_versions": []}],
        },
    ]
}


def test_audit_summary_table_and_annotations(tmp_path, capsys):
    import json

    f = tmp_path / "audit.json"
    f.write_text(json.dumps(AUDIT))
    assert (
        audit_summary.main([str(f), "--output", str(tmp_path / "a.md"), "--annotate"])
        == 0
    )
    md = (tmp_path / "a.md").read_text()
    assert "**3 known vulnerabilities in 2 packages.**" in md
    assert "| langchain-core | 0.3.63 | PYSEC-1, PYSEC-2 | 0.3.81, 0.3.85 |" in md
    assert "| chromadb | 1.5.9 | PYSEC-3 | no fix |" in md
    out = capsys.readouterr().out
    assert "::warning title=Vulnerable dependency langchain-core 0.3.63::" in out


def test_audit_summary_clean_and_unreadable_reports(tmp_path, capsys):
    import json

    clean = tmp_path / "clean.json"
    clean.write_text(
        json.dumps({"dependencies": [{"name": "x", "version": "1", "vulns": []}]})
    )
    assert audit_summary.main([str(clean)]) == 0
    assert "No known vulnerabilities" in capsys.readouterr().out
    assert (
        audit_summary.main([str(tmp_path / "missing.json")]) == 0
    )  # advisory: never blocks
