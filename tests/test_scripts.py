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
