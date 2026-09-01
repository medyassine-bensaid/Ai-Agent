"""
conftest.py — Global pytest fixtures.

Patches Langfuse SDK for ALL tests so no real network calls are made.
Also seeds the required env vars so imports don't crash on missing keys.
"""
import os
import pytest
from unittest.mock import MagicMock, patch

# ── 1. Seed dummy env vars BEFORE any import of config.settings ───────────────
#    These only apply to the test process; they never touch the real environment.
os.environ.setdefault("GROQ_API_KEY",         "sk-dummy-groq-key-for-tests")
os.environ.setdefault("TAVILY_API_KEY",        "tvly-dummy-tavily-key")
os.environ.setdefault("LANGFUSE_SECRET_KEY",   "sk-lf-dummy-secret")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY",   "pk-lf-dummy-public")
os.environ.setdefault("LANGFUSE_BASE_URL",     "http://localhost:9999")  # non-existent → no network
os.environ.setdefault("APP_ENV",               "test")
os.environ.setdefault("APP_VERSION",           "0.0.0-test")


# ── 2. Patch Langfuse SDK globally so it never sends HTTP requests ─────────────
@pytest.fixture(autouse=True)
def mock_langfuse_globally():
    """
    Auto-used fixture: patches every Langfuse entry-point used by the project.
    Applied to every test automatically — no need to add it to individual tests.
    """
    mock_client = MagicMock()
    mock_client.update_current_observation = MagicMock()
    mock_client.update_current_trace       = MagicMock()
    mock_client.score_current_trace        = MagicMock()
    mock_client.create_score               = MagicMock()
    mock_client.flush                      = MagicMock()
    mock_client.shutdown                   = MagicMock()

    # CallbackHandler is a no-op
    mock_handler = MagicMock()
    mock_handler.flush = MagicMock()

    with patch("langfuse.get_client",           return_value=mock_client), \
         patch("langfuse.langchain.CallbackHandler", return_value=mock_handler), \
         patch("langfuse.observe",              lambda *a, **kw: lambda fn: fn), \
         patch("langfuse.propagate_attributes") as mock_prop:

        # Make propagate_attributes work as a context manager
        mock_prop.return_value.__enter__ = MagicMock(return_value=None)
        mock_prop.return_value.__exit__  = MagicMock(return_value=False)

        yield mock_client
