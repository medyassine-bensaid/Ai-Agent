"""
test_agent.py — Unit tests for the DevOps ReAct Agent.

All Langfuse / network calls are patched globally via tests/conftest.py.
Run:  pytest tests/ -v --cov=. --cov-report=term-missing
"""
import pytest
from unittest.mock import patch, MagicMock


# ─── Config ───────────────────────────────────────────────────────────────────

def test_settings_has_langfuse_keys():
    """Settings class must expose Langfuse key attributes."""
    from config.settings import Settings
    assert hasattr(Settings, "LANGFUSE_SECRET_KEY")
    assert hasattr(Settings, "LANGFUSE_PUBLIC_KEY")
    assert hasattr(Settings, "LANGFUSE_BASE_URL")


def test_settings_validation_raises_on_missing_keys(monkeypatch):
    """settings.validate() must raise ValueError when a key is missing."""
    import importlib
    import config.settings as s

    monkeypatch.delenv("GROQ_API_KEY",       raising=False)
    monkeypatch.delenv("TAVILY_API_KEY",      raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

    importlib.reload(s)   # re-read env after monkeypatching

    with pytest.raises(ValueError, match="Missing required"):
        s.Settings.validate()

    # Restore env for subsequent tests
    importlib.reload(s)


def test_settings_validation_passes_with_all_keys():
    """validate() must not raise when all required keys are present."""
    from config.settings import Settings
    # conftest.py seeds all keys, so this must pass cleanly
    Settings.validate()


def test_settings_app_env_default():
    from config.settings import settings
    assert settings.APP_ENV in ("test", "development", "staging", "production")


# ─── State schema ─────────────────────────────────────────────────────────────

def test_agent_state_schema():
    from agent.state import AgentState
    required = {
        "query", "messages", "thought", "tool_calls",
        "action", "search_query", "iteration", "ready_to_answer", "final_answer"
    }
    assert required.issubset(AgentState.__annotations__.keys())


def test_tool_call_schema():
    from agent.state import ToolCall
    assert set(ToolCall.__annotations__.keys()) == {"tool", "query", "reasoning"}


# ─── Memory store ─────────────────────────────────────────────────────────────

def test_memory_add_and_retrieve():
    from utils.memory_store import ConversationMemory
    mem = ConversationMemory(max_messages=5)
    mem.add_message("user",      "How does CI/CD work?")
    mem.add_message("assistant", "CI/CD automates the delivery pipeline.")
    ctx = mem.get_context()
    assert "CI/CD" in ctx


def test_memory_clear():
    from utils.memory_store import ConversationMemory
    mem = ConversationMemory()
    mem.add_message("user", "Hello")
    mem.clear()
    assert mem.get_context() == "No previous conversation."


def test_memory_respects_max_messages():
    from utils.memory_store import ConversationMemory
    mem = ConversationMemory(max_messages=2)
    for i in range(20):
        mem.add_message("user", f"msg {i}")
    assert len(mem.messages) <= 8   # 2 * 2 * 2 — generous upper bound


def test_memory_metadata_context():
    from utils.memory_store import ConversationMemory
    mem = ConversationMemory()
    mem.update_metadata("user_expertise", "intermediate")
    meta = mem.get_metadata_context()
    assert "intermediate" in meta


# ─── Prompts ──────────────────────────────────────────────────────────────────

def test_system_prompt_has_context_placeholder():
    from agent.prompts import SYSTEM_PROMPT
    assert "{context}" in SYSTEM_PROMPT


def test_react_prompt_has_required_placeholders():
    from agent.prompts import REACT_PROMPT
    assert "{query}"   in REACT_PROMPT
    assert "{context}" in REACT_PROMPT


def test_final_answer_prompt_has_required_placeholders():
    from agent.prompts import FINAL_ANSWER_PROMPT
    assert "{query}"             in FINAL_ANSWER_PROMPT
    assert "{reasoning_history}" in FINAL_ANSWER_PROMPT


# ─── Knowledge base tool ──────────────────────────────────────────────────────

def test_knowledge_base_docker_hit():
    from tools.knowledge_base import knowledge_base_tool
    result = knowledge_base_tool.invoke({"topic": "docker"})
    assert "Docker" in result or "docker" in result.lower()


def test_knowledge_base_kubernetes_hit():
    from tools.knowledge_base import knowledge_base_tool
    result = knowledge_base_tool.invoke({"topic": "kubernetes"})
    assert "kubernetes" in result.lower() or "Kubernetes" in result


def test_knowledge_base_cicd_hit():
    from tools.knowledge_base import knowledge_base_tool
    result = knowledge_base_tool.invoke({"topic": "ci/cd"})
    assert "CI" in result or "continuous" in result.lower()


def test_knowledge_base_miss_returns_fallback():
    from tools.knowledge_base import knowledge_base_tool
    result = knowledge_base_tool.invoke({"topic": "xyznonexistent_topic_abc"})
    assert "No specific knowledge" in result or "web_search" in result


# ─── Web search tool ──────────────────────────────────────────────────────────

def test_web_search_returns_formatted_results():
    from tools.web_search import web_search_tool
    mock_results = [
        {"content": "GitHub Actions is a CI/CD platform.", "url": "https://docs.github.com/actions"}
    ]
    with patch("tools.web_search.tavily_search") as mock_tavily:
        mock_tavily.invoke.return_value = mock_results
        result = web_search_tool.invoke({"query": "github actions tutorial"})

    assert "GitHub Actions" in result
    assert "docs.github.com" in result
    assert "Search Results" in result


def test_web_search_handles_empty_results():
    from tools.web_search import web_search_tool
    with patch("tools.web_search.tavily_search") as mock_tavily:
        mock_tavily.invoke.return_value = []
        result = web_search_tool.invoke({"query": "obscure query"})

    assert "No results" in result


def test_web_search_handles_tool_exception():
    from tools.web_search import web_search_tool
    with patch("tools.web_search.tavily_search") as mock_tavily:
        mock_tavily.invoke.side_effect = Exception("network error")
        result = web_search_tool.invoke({"query": "any query"})

    assert "failed" in result.lower() or "error" in result.lower()


# ─── Graph routing ────────────────────────────────────────────────────────────

def test_router_max_iterations_returns_answer():
    from agent.graph import should_continue
    state = {"iteration": 99, "ready_to_answer": False, "tool_calls": [], "messages": []}
    assert should_continue(state) == "answer"


def test_router_ready_to_answer_returns_answer():
    from agent.graph import should_continue
    state = {"iteration": 1, "ready_to_answer": True, "tool_calls": [], "messages": []}
    assert should_continue(state) == "answer"


def test_router_pending_tool_returns_tool_execution():
    from agent.graph import should_continue
    state = {
        "iteration":       1,
        "ready_to_answer": False,
        "tool_calls":      [{"tool": "web_search", "query": "kubernetes tutorial", "reasoning": "x"}],
        "messages":        [],    # no observation stored yet
    }
    assert should_continue(state) == "tool_execution"


def test_router_tool_already_executed_returns_answer():
    """If an observation for the tool already exists in messages, go to answer."""
    from langchain_core.messages import AIMessage
    from agent.graph import should_continue
    state = {
        "iteration":       1,
        "ready_to_answer": False,
        "tool_calls":      [{"tool": "web_search", "query": "kubernetes tutorial", "reasoning": "x"}],
        "messages":        [AIMessage(content="Tool: web_search\nQuery: kubernetes tutorial\nResult: ...")],
    }
    assert should_continue(state) == "answer"


# ─── Graph nodes ──────────────────────────────────────────────────────────────

def test_tool_execution_node_web_search():
    """tool_execution_node must call web_search and update state correctly."""
    from agent.graph import tool_execution_node
    mock_obs = "Kubernetes is a container orchestrator."

    with patch("agent.graph.web_search_tool") as mock_ws:
        mock_ws.invoke.return_value = mock_obs
        state = {
            "tool_calls":      [{"tool": "web_search", "query": "k8s basics", "reasoning": "x"}],
            "messages":        [],
            "ready_to_answer": False,
        }
        result = tool_execution_node(state)

    assert result["ready_to_answer"] is True
    assert any("k8s basics" in str(m.content) for m in result["messages"])


def test_tool_execution_node_knowledge_base():
    from agent.graph import tool_execution_node

    with patch("agent.graph.knowledge_base_tool") as mock_kb:
        mock_kb.invoke.return_value = "Docker packages apps into containers."
        state = {
            "tool_calls":      [{"tool": "knowledge_base", "query": "docker basics", "reasoning": "x"}],
            "messages":        [],
            "ready_to_answer": False,
        }
        result = tool_execution_node(state)

    assert result["ready_to_answer"] is True


def test_tool_execution_node_no_tool_calls():
    from agent.graph import tool_execution_node
    state = {"tool_calls": [], "messages": [], "ready_to_answer": False}
    result = tool_execution_node(state)
    assert result["ready_to_answer"] is True


def test_tool_execution_node_handles_exception():
    from agent.graph import tool_execution_node
    with patch("agent.graph.web_search_tool") as mock_ws:
        mock_ws.invoke.side_effect = Exception("timeout")
        state = {
            "tool_calls":      [{"tool": "web_search", "query": "test", "reasoning": "x"}],
            "messages":        [],
            "ready_to_answer": False,
        }
        result = tool_execution_node(state)
    assert result["ready_to_answer"] is True
    assert any("Error" in str(m.content) for m in result["messages"])


# ─── run_agent (end-to-end, fully mocked) ─────────────────────────────────────

def test_run_agent_returns_string_answer():
    """run_agent must return a non-empty string for any input."""
    with patch("main.agent_graph") as mock_graph:
        mock_graph.invoke.return_value = {
            "final_answer": "Docker is a containerisation platform.",
            "iteration":    1,
        }
        from main import run_agent
        result = run_agent("What is Docker?", session_id="test-session-001")

    assert isinstance(result, str)
    assert len(result) > 0


def test_run_agent_handles_graph_exception():
    """run_agent must not raise even if the graph throws."""
    with patch("main.agent_graph") as mock_graph:
        mock_graph.invoke.side_effect = RuntimeError("graph error")
        from main import run_agent
        result = run_agent("crash test", session_id="test-session-002")

    assert isinstance(result, str)
    assert "error" in result.lower() or "apologize" in result.lower()


# ─── Langfuse scorer ──────────────────────────────────────────────────────────

def test_score_answer_quality_long_structured():
    from monitoring.langfuse_scorer import score_answer_quality
    answer = (
        "CI/CD consists of:\n"
        "1. Continuous Integration\n"
        "2. Continuous Delivery\n"
        "```yaml\npipeline:\n  - build\n```\n" * 5
    )
    score = score_answer_quality(answer)
    assert 0.0 <= score <= 1.0
    assert score >= 0.5   # long + structured should score well


def test_score_answer_quality_empty():
    from monitoring.langfuse_scorer import score_answer_quality
    assert score_answer_quality("") == 0.0


def test_score_answer_quality_error_message():
    from monitoring.langfuse_scorer import score_answer_quality
    score = score_answer_quality("I apologize, but I encountered an error generating the response.")
    assert score < 0.4   # errors should score poorly


def test_score_answer_quality_short():
    from monitoring.langfuse_scorer import score_answer_quality
    score = score_answer_quality("Yes.")
    assert score <= 0.3
