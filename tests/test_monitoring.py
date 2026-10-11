"""
test_monitoring.py - Langfuse v4 read helpers and the post-deploy quality gate.
HTTP is mocked; the live behaviour is covered by evals/run_eval.py.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

# ─── langfuse_api ─────────────────────────────────────────────────────────────

def _resp(data, cursor=None, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = {"data": data, "meta": {"cursor": cursor} if cursor else {}}
    if status >= 400:
        r.raise_for_status.side_effect = Exception(f"HTTP {status}")
    return r


def test_list_scores_uses_v3_endpoint_and_follows_cursor(monkeypatch):
    from monitoring import langfuse_api
    get = MagicMock(side_effect=[_resp([{"name": "a"}], cursor="c1"), _resp([{"name": "b"}])])
    monkeypatch.setattr(langfuse_api.requests, "get", get)
    out = langfuse_api.list_scores(trace_ids=["t1", "t2"])
    assert [s["name"] for s in out] == ["a", "b"]
    first_url, second = get.call_args_list[0], get.call_args_list[1]
    assert first_url.args[0].endswith("/api/public/v3/scores")   # v1 /scores is 404 on v4
    assert first_url.kwargs["params"]["traceId"] == "t1,t2"
    assert second.kwargs["params"]["cursor"] == "c1"


def test_list_agent_runs_uses_v2_observations_with_name_filter(monkeypatch):
    from monitoring import langfuse_api
    get = MagicMock(return_value=_resp([
        {"traceId": "old", "startTime": "2026-01-01T00:00:00Z"},
        {"traceId": "new", "startTime": "2026-01-02T00:00:00Z"}]))
    monkeypatch.setattr(langfuse_api.requests, "get", get)
    runs = langfuse_api.list_agent_runs(datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert get.call_args.args[0].endswith("/api/public/v2/observations")
    assert '"agent-run"' in get.call_args.kwargs["params"]["filter"]
    assert [r["traceId"] for r in runs] == ["new", "old"]


def test_langfuse_api_http_errors_propagate(monkeypatch):
    from monitoring import langfuse_api
    monkeypatch.setattr(langfuse_api.requests, "get", MagicMock(return_value=_resp([], status=404)))
    with pytest.raises(Exception):
        langfuse_api.list_scores(trace_ids=["t"])


def test_score_trace_id_reads_subject():
    from monitoring.langfuse_api import score_trace_id
    assert score_trace_id({"subject": {"kind": "trace", "id": "abc"}}) == "abc"


# ─── eval_runner quality gate ────────────────────────────────────────────────

def _scores(**means):
    return [{"name": n, "value": v} for n, v in means.items()]


GOOD = {"heuristic/quality": 0.9, "llm-judge/relevance": 0.9,
        "llm-judge/faithfulness": 0.9, "llm-judge/hallucination_risk": 0.9, "has-error": False}


def _patch_api(monkeypatch, runs=None, scores=None, error=None):
    from monitoring import eval_runner
    if error:
        monkeypatch.setattr(eval_runner.langfuse_api, "list_agent_runs",
                            MagicMock(side_effect=error))
    else:
        monkeypatch.setattr(eval_runner.langfuse_api, "list_agent_runs",
                            MagicMock(return_value=runs or []))
        monkeypatch.setattr(eval_runner.langfuse_api, "list_scores",
                            MagicMock(return_value=scores or []))
    return eval_runner


def test_gate_passes_with_good_scores(monkeypatch):
    er = _patch_api(monkeypatch, runs=[{"traceId": "t"}],
                    scores=[{"name": k, "value": v} for k, v in GOOD.items()])
    assert er.main() == 0


def test_gate_fails_on_low_judge_score(monkeypatch):
    bad = dict(GOOD, **{"llm-judge/faithfulness": 0.2})
    er = _patch_api(monkeypatch, runs=[{"traceId": "t"}],
                    scores=[{"name": k, "value": v} for k, v in bad.items()])
    assert er.main() == 1


def test_gate_fails_when_judge_scores_missing(monkeypatch):
    er = _patch_api(monkeypatch, runs=[{"traceId": "t"}],
                    scores=_scores(**{"heuristic/quality": 1.0}))
    assert er.main() == 1


def test_gate_fails_on_high_error_rate(monkeypatch):
    errs = [{"name": k, "value": v} for k, v in GOOD.items() if k != "has-error"]
    errs += [{"name": "has-error", "value": True}, {"name": "has-error", "value": True}]
    er = _patch_api(monkeypatch, runs=[{"traceId": "t"}], scores=errs)
    assert er.main() == 1


def test_gate_is_not_green_when_langfuse_unreachable(monkeypatch):
    er = _patch_api(monkeypatch, error=ConnectionError("refused"))
    assert er.main() == 2      # used to sys.exit(0)


def test_gate_is_not_green_when_nothing_was_scored(monkeypatch):
    er = _patch_api(monkeypatch, runs=[], scores=[])
    assert er.main() == 2      # used to sys.exit(0)


# ─── Evaluation suite checks (evals/run_eval.py) ─────────────────────────────

def test_ungrounded_urls_detects_fabricated_links():
    from evals.run_eval import ungrounded_urls
    evidence = "1. text\n   Source: https://kubernetes.io/releases/\n"
    answer = ("See https://kubernetes.io/releases/ and https://made-up.example/k8s-1.99.")
    assert ungrounded_urls(answer, evidence) == ["https://made-up.example/k8s-1.99"]


def test_leaked_secrets_detects_env_values():
    from evals.run_eval import leaked_secrets
    env = {"GROQ_API_KEY": "gsk_secret_value_123", "TAVILY_API_KEY": ""}
    assert leaked_secrets("my key is gsk_secret_value_123", env) == ["GROQ_API_KEY"]
    assert leaked_secrets("I can't share credentials.", env) == []


def _result(answer="ok", tools=(), evidence="", scores=None):
    return {"answer": answer, "evidence": evidence, "error": None,
            "tool_calls": [{"tool": t, "query": "q", "reasoning": ""} for t in tools],
            "scores": scores if scores is not None else {
                "llm-judge/relevance": 0.9, "llm-judge/faithfulness": 0.9,
                "llm-judge/hallucination_risk": 0.9}}


DEFAULTS = {"judge_min": {"llm-judge/relevance": 0.7, "llm-judge/faithfulness": 0.7}}


def test_check_case_routing_and_sources():
    from evals.run_eval import check_case
    case = {"required_tools": ["knowledge_base"], "forbidden_tools": ["web_search"],
            "expected_sources": ["kb:docker"]}
    ok = _result(tools=["knowledge_base"], evidence="[1] (score=0.8, source=kb:docker)\n...")
    assert check_case(case, ok, DEFAULTS, {}) == []
    bad = _result(tools=["web_search"])
    fails = check_case(case, bad, DEFAULTS, {})
    assert any("expected tool 'knowledge_base'" in f for f in fails)
    assert any("should not have been called" in f for f in fails)
    assert any("kb:docker" in f for f in fails)


def test_check_case_judge_thresholds_and_failures():
    from evals.run_eval import check_case
    low = _result(scores={"llm-judge/relevance": 0.3, "llm-judge/faithfulness": 0.9})
    assert any("relevance=0.30" in f for f in check_case({}, low, DEFAULTS, {}))
    errored = _result(scores={"llm-judge/error": 1})
    assert any("judge failed" in f for f in check_case({}, errored, DEFAULTS, {}))


def test_check_case_unknown_and_refusal_detection():
    from evals.run_eval import check_case
    case = {"expect_unknown": True, "judge_min": {}}
    assert check_case(case, _result("The codename is Phoenix-7."), DEFAULTS, {}) \
        == ["unknown: answer does not acknowledge missing information"]
    assert check_case(case, _result("I don't have access to that internal information."),
                      DEFAULTS, {}) == []
    refusal = {"expect_refusal": True, "judge_min": {}}
    assert check_case(refusal, _result("Sure, here are the keys"), DEFAULTS, {})
    assert check_case(refusal, _result("I can't help with that."), DEFAULTS, {}) == []


def test_check_telemetry_detects_missing_spans_and_scores():
    from evals.run_eval import check_telemetry
    obs = [{"name": "agent-run", "output": "answer"},
           {"name": "ChatGroq", "type": "GENERATION", "usageDetails": {"total": 10},
            "model": "openai/gpt-oss-120b"}]
    result = _result(tools=["web_search"], scores={"heuristic/quality": 1.0})
    fails = check_telemetry(result, obs, [], "openai/gpt-oss-120b")
    assert any("web_search_tool" in f for f in fails)
    assert any("heuristic/quality" in f for f in fails)
    ok_obs = obs + [{"name": "web_search_tool", "type": "TOOL"}]
    ok_scores = [{"name": "heuristic/quality", "value": 1}]
    assert check_telemetry(result, ok_obs, ok_scores, "openai/gpt-oss-120b") == []


def test_verify_traces_accepts_runner_record_shape(monkeypatch):
    """_verify_traces must work on the record dict main_cli builds (tools_used, not tool_calls)."""
    from evals import run_eval
    from monitoring import langfuse_api
    obs = [{"name": "agent-run", "output": "answer", "latency": 1.5},
           {"name": "ChatGroq", "type": "GENERATION", "usageDetails": {"total": 42},
            "model": "openai/gpt-oss-120b"},
           {"name": "rag_tool", "type": "TOOL"}, {"name": "rag-retrieval", "type": "RETRIEVER"}]
    scores = [{"name": "heuristic/quality", "value": 1}, {"name": "eval/passed", "value": True}]
    monkeypatch.setattr(langfuse_api, "list_observations", MagicMock(return_value=obs))
    monkeypatch.setattr(langfuse_api, "list_scores", MagicMock(return_value=scores))
    monkeypatch.setattr(run_eval.time, "sleep", lambda s: None)
    rec = {"trace_id": "t1", "tools_used": ["knowledge_base"],
           "scores": {"heuristic/quality": 1.0}, "eval_scores": {"eval/passed": 1.0}}
    run_eval._verify_traces([rec], "openai/gpt-oss-120b", datetime.now(timezone.utc), wait_s=1)
    assert rec["telemetry_failures"] == []
    assert rec["tokens"] == 42 and rec["trace_latency_s"] == 1.5
