"""
Configuration management for the DevOps Agent.
Keeps all settings in one place for easy modification.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Configuration settings for the agent"""

    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # Model Configuration
    MODEL_NAME = "openai/gpt-oss-120b"
    TEMPERATURE = 0.1  # Low temp for consistency
    MAX_TOKENS = 8000
    LLM_MAX_RETRIES = 5  # SDK retries with backoff on 429 / transient errors

    # Agent Configuration
    MAX_ITERATIONS = 5  # Prevent infinite loops
    MAX_TOOL_CALLS = 3  # Tool calls per turn (e.g. knowledge base + web search)
    MAX_CONTEXT_MESSAGES = 10  # Keep last N messages

    # Tool Configuration
    WEB_SEARCH_MAX_RESULTS = 5
    WEB_SEARCH_TIMEOUT = 30  # seconds; a stalled Tavily call must not hang the agent

    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")  # sk-lf-...
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")  # pk-lf-...
    LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")

    APP_ENV = os.getenv("APP_ENV", "development")  # development | staging | production
    APP_VERSION = os.getenv("APP_VERSION", "0.1.0")

    @classmethod
    def validate(cls):
        """Validate required settings"""
        missing = [k for k in ("GROQ_API_KEY", "TAVILY_API_KEY") if not getattr(cls, k)]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )


settings = Settings()
