"""
test_regressions.py - Regression tests for defects found during validation
(see docs/VALIDATION_REPORT.md). No network: Groq, Tavily and Langfuse are mocked.
"""

import io
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

# ─── Logger must not crash on consoles that cannot encode emoji ──────────────


def test_safe_print_survives_non_utf8_stdout(monkeypatch):
    from utils.logger import logger

    raw = io.BytesIO()
    cp1252 = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", cp1252)
    logger.log_iteration(0, 5)  # contains an emoji
    logger.log_error("boom ❌")
    cp1252.flush()
    assert b"ITERATION 0/5" in raw.getvalue()


# ─── LLM judge ────────────────────────────────────────────────────────────────


def _judge_response(content, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason, message=SimpleNamespace(content=content)
            )
        ]
    )


GOOD_JUDGE = (
    '{"relevance":0.9,"faithfulness":0.8,"completeness":0.7,'
    '"hallucination_risk":1.0,"reasoning":"ok"}'
)


def test_parse_judge_valid():
    from monitoring.langfuse_scorer import _parse_judge

    out = _parse_judge("```json\n" + GOOD_JUDGE + "\n```")
    assert out["relevance"] == 0.9 and out["reasoning"] == "ok"


@pytest.mark.parametrize(
    "raw",
    [
        "",  # empty content (token budget exhausted)
        '{"relevance":0.9}',  # missing metrics
        GOOD_JUDGE.replace("0.9", "9"),  # out of range
        GOOD_JUDGE.replace("0.9", '"high"'),  # wrong type
    ],
)
def test_parse_judge_rejects_invalid(raw):
    from monitoring.langfuse_scorer import _parse_judge

    with pytest.raises(ValueError):
        _parse_judge(raw)


def test_judge_budget_fits_reasoning_model_and_sends_evidence(monkeypatch):
    import monitoring.langfuse_scorer as sc

    client = MagicMock()
    client.chat.completions.create.return_value = _judge_response(GOOD_JUDGE)
    monkeypatch.setattr(sc, "_get_judge_client", lambda: client)
    out = sc._llm_judge("q?", "an answer", evidence="EVIDENCE-TEXT")
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["max_tokens"] >= 1024  # was 300: exhausted by reasoning tokens
    assert "EVIDENCE-TEXT" in kwargs["messages"][0]["content"]
    assert out["faithfulness"] == 0.8


def test_judge_truncated_output_is_an_error_not_empty_scores(monkeypatch):
    import monitoring.langfuse_scorer as sc

    client = MagicMock()
    client.chat.completions.create.return_value = _judge_response(
        "", finish_reason="length"
    )
    monkeypatch.setattr(sc, "_get_judge_client", lambda: client)
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)
    out = sc._llm_judge("q?", "a")
    assert "error" in out
    assert client.chat.completions.create.call_count == 2  # one retry


def test_inline_score_records_judge_failure_and_correct_has_error(monkeypatch):
    import monitoring.langfuse_scorer as sc

    lf = MagicMock()
    monkeypatch.setattr(sc, "get_client", lambda: lf)
    monkeypatch.setattr(sc, "_llm_judge", lambda *a, **k: {"error": "boom"})
    submitted = sc.inline_score(
        "t" * 32, "q", "I apologize, but I encountered an error: x"
    )
    names = {
        c.kwargs["name"]: c.kwargs["value"] for c in lf.create_score.call_args_list
    }
    assert names["llm-judge/error"] == 1
    assert names["has-error"] == 1  # 1 now means "is an error answer"
    assert submitted == names


def test_inline_score_success_path(monkeypatch):
    import monitoring.langfuse_scorer as sc

    lf = MagicMock()
    monkeypatch.setattr(sc, "get_client", lambda: lf)
    monkeypatch.setattr(sc, "_llm_judge", lambda *a, **k: sc._parse_judge(GOOD_JUDGE))
    submitted = sc.inline_score("t" * 32, "q", "Docker is a container platform.")
    assert submitted["has-error"] == 0
    assert set(submitted) >= {
        "llm-judge/relevance",
        "llm-judge/faithfulness",
        "llm-judge/completeness",
        "llm-judge/hallucination_risk",
        "heuristic/quality",
    }
    lf.flush.assert_called_once()


def test_scorer_import_does_not_require_groq_key(monkeypatch):
    import importlib

    import monitoring.langfuse_scorer as sc

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    importlib.reload(sc)  # used to raise KeyError at import time


# ─── Agent graph: ReAct loop, tool budget, grounding prompt ──────────────────


def _llm_reply(content):
    return SimpleNamespace(content=content)


def _state(**kw):
    base = {
        "query": "q",
        "messages": [],
        "thought": "",
        "tool_calls": [],
        "action": "",
        "search_query": None,
        "iteration": 0,
        "ready_to_answer": False,
        "final_answer": "",
    }
    base.update(kw)
    return base


def test_reasoning_can_call_second_tool_after_observation():
    from agent import graph

    state = _state(
        tool_calls=[
            {"tool": "knowledge_base", "query": "kubernetes", "reasoning": "x"}
        ],
        messages=[
            AIMessage(
                content="Tool: knowledge_base\nQuery: kubernetes\nResult:\nK8s notes"
            )
        ],
        ready_to_answer=True,
        iteration=1,
    )
    decision = '{"reasoning":"need latest version","action":"use_web_search","search_query":"latest kubernetes release"}'
    with patch.object(graph, "llm") as llm:
        llm.invoke.return_value = _llm_reply(decision)
        out = graph.reasoning_node(state)
        prompt = llm.invoke.call_args.args[0][1].content
    assert "K8s notes" in prompt  # observations are fed back
    assert out["ready_to_answer"] is False
    assert [tc["tool"] for tc in out["tool_calls"]] == ["knowledge_base", "web_search"]
    assert graph.should_continue(out) == "tool_execution"


def test_reasoning_stops_at_tool_budget():
    from agent import graph
    from config.settings import settings

    calls = [
        {"tool": "web_search", "query": f"q{i}", "reasoning": "x"}
        for i in range(settings.MAX_TOOL_CALLS)
    ]
    with patch.object(graph, "llm") as llm:
        out = graph.reasoning_node(_state(tool_calls=calls))
        llm.invoke.assert_not_called()
    assert out["ready_to_answer"] is True


def test_reasoning_skips_duplicate_tool_call():
    from agent import graph

    state = _state(
        tool_calls=[{"tool": "web_search", "query": "helm latest", "reasoning": "x"}]
    )
    decision = (
        '{"reasoning":"again","action":"use_web_search","search_query":"helm latest"}'
    )
    with patch.object(graph, "llm") as llm:
        llm.invoke.return_value = _llm_reply(decision)
        out = graph.reasoning_node(state)
    assert len(out["tool_calls"]) == 1
    assert out["ready_to_answer"] is True


def test_graph_loops_from_tool_execution_back_to_reasoning():
    from agent.graph import agent_graph

    edges = set(agent_graph.builder.edges)
    assert ("tool_execution", "reasoning") in edges
    assert ("tool_execution", "answer") not in edges


def test_prompts_describe_kb_and_grounding_rules():
    from agent.prompts import FINAL_ANSWER_PROMPT, REACT_PROMPT

    assert "{kb_topics}" in REACT_PROMPT and "{observations}" in REACT_PROMPT
    assert "Never cite a source that is not" in FINAL_ANSWER_PROMPT
    assert "do not\n  claim to have searched" in FINAL_ANSWER_PROMPT


# ─── run_agent_detailed exposes routing/evidence and scores with evidence ────


def test_run_agent_detailed_returns_evidence_and_scores_with_it(monkeypatch):
    import main

    obs = "Tool: web_search\nQuery: k8s\nResult:\n1. v1.36 Source: https://kubernetes.io/releases"
    final = {
        "final_answer": "Kubernetes v1.36 (https://kubernetes.io/releases)",
        "tool_calls": [{"tool": "web_search", "query": "k8s", "reasoning": "x"}],
        "messages": [AIMessage(content=obs)],
    }
    scorer = MagicMock(return_value={"heuristic/quality": 0.5})
    monkeypatch.setattr(main, "_get_scorer", lambda: scorer)
    monkeypatch.setattr(main.langfuse_client, "get_current_trace_id", lambda: "a" * 32)
    with patch("main.agent_graph") as g:
        g.invoke.return_value = final
        res = main.run_agent_detailed("latest k8s?", session_id="s")
    assert res["tool_calls"][0]["tool"] == "web_search"
    assert res["evidence"] == obs
    assert res["trace_id"] == "a" * 32
    assert scorer.call_args.kwargs["evidence"] == obs
    assert res["scores"] == {"heuristic/quality": 0.5}


# ─── RAG: relevance threshold and change-aware ingestion ─────────────────────


class _FakeEmbedder:
    """Deterministic bag-of-words embedder over a tiny vocabulary."""

    VOCAB = ["docker", "container", "image", "kubernetes", "pod", "cake", "chocolate"]

    def encode(self, texts):
        import numpy as np

        vecs = []
        for t in texts:
            v = np.array([t.lower().count(w) for w in self.VOCAB], dtype=float) + 1e-3
            vecs.append(v / np.linalg.norm(v))
        return np.array(vecs)


@pytest.fixture
def rag(tmp_path, monkeypatch):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    import tools.rag_tool as R

    client = chromadb.PersistentClient(
        path=str(tmp_path), settings=ChromaSettings(anonymized_telemetry=False)
    )
    col = client.get_or_create_collection(
        name="test_docs", metadata={"hnsw:space": "cosine"}
    )
    monkeypatch.setattr(R, "_collection", col)
    monkeypatch.setattr(R, "_embedder", _FakeEmbedder())
    return R


def test_rag_retrieves_relevant_chunk_with_source(rag):
    rag.ingest_text("Docker container image basics", source="kb:docker")
    rag.ingest_text("Kubernetes pod scheduling", source="kb:kubernetes")
    docs = rag._retrieve("docker image")
    assert docs and docs[0]["source"] == "kb:docker"
    assert all(d["score"] >= rag.MIN_SCORE for d in docs)


def test_rag_drops_irrelevant_chunks(rag):
    rag.ingest_text("Docker container image basics", source="kb:docker")
    assert rag._retrieve("chocolate cake") == []
    out = rag.rag_tool.invoke({"query": "chocolate cake"})
    assert "No relevant documentation" in out


def test_rag_reingest_updates_changed_chunk_and_is_idempotent(rag):
    assert rag.ingest_text("Docker container old text", source="kb:docker") == 1
    assert (
        rag.ingest_text("Docker container old text", source="kb:docker") == 0
    )  # duplicate
    assert rag.ingest_text("Docker image new text", source="kb:docker") == 1  # changed
    col = rag._get_collection()
    assert col.count() == 1
    assert col.get()["documents"] == ["Docker image new text"]


# ─── Tavily calls must be bounded by a timeout ───────────────────────────────


def test_tavily_request_has_timeout(monkeypatch):
    import tools.web_search as ws
    from config.settings import settings

    resp = MagicMock()
    resp.json.return_value = {
        "results": [{"url": "https://example.com/a", "content": "c"}]
    }
    post = MagicMock(return_value=resp)
    monkeypatch.setattr(ws.requests, "post", post)
    out = ws.make_tavily_search("tvly-test").invoke({"query": "helm latest"})
    assert post.call_args.kwargs["timeout"] == settings.WEB_SEARCH_TIMEOUT
    assert out[0]["url"] == "https://example.com/a"


def test_web_search_tool_reports_timeout_as_failure(monkeypatch):
    import requests

    import tools.web_search as ws

    monkeypatch.setattr(
        ws.requests, "post", MagicMock(side_effect=requests.Timeout("read timed out"))
    )
    monkeypatch.setattr(ws, "tavily_search", ws.make_tavily_search("tvly-test"))
    out = ws.web_search_tool.invoke({"query": "helm latest"})
    assert "Web search failed" in out
    assert "read timed out" in out  # real cause, not "'str' object has no attribute"


# ─── Groq tool_use_failed (gpt-oss emits a native tool call) is recovered ────


class _ToolUseFailed(Exception):
    def __init__(self, failed_generation):
        super().__init__("Error code: 400 - tool_use_failed")
        self.body = {
            "error": {"code": "tool_use_failed", "failed_generation": failed_generation}
        }


def test_reasoning_recovers_native_tool_call():
    from agent import graph

    err = _ToolUseFailed(
        '{"name": "web_search", "arguments": {"query": "Terraform v1.15.0 release notes"}}'
    )
    with patch.object(graph, "llm") as llm:
        llm.invoke.side_effect = err
        out = graph.reasoning_node(_state())
    assert out["tool_calls"] == [
        {
            "tool": "web_search",
            "query": "Terraform v1.15.0 release notes",
            "reasoning": "Model requested the web_search tool directly",
        }
    ]
    assert out["ready_to_answer"] is False


def test_reasoning_other_llm_errors_still_fall_back_to_answer():
    from agent import graph

    with patch.object(graph, "llm") as llm:
        llm.invoke.side_effect = _ToolUseFailed("not json")
        out = graph.reasoning_node(_state())
    assert out["ready_to_answer"] is True and out["tool_calls"] == []


# ─── Provider errors must not leak into the user-facing answer ───────────────


def test_answer_node_error_does_not_expose_provider_details():
    from agent import graph

    secret_detail = "Rate limit reached in organization org_01SECRET; see https://console.groq.com/settings/billing"
    with patch.object(graph, "llm") as llm:
        llm.invoke.side_effect = RuntimeError(secret_detail)
        out = graph.answer_node(_state(thought="t"))
    assert "org_01SECRET" not in out["final_answer"]
    assert "console.groq.com" not in out["final_answer"]
    assert "I apologize" in out["final_answer"]  # still detected as an error answer
