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
    MODEL_NAME = "llama-3.3-70b-versatile"
    TEMPERATURE = 0.1  # Low temp for consistency
    MAX_TOKENS = 8000
    
    # Agent Configuration
    MAX_ITERATIONS = 5  # Prevent infinite loops
    MAX_CONTEXT_MESSAGES = 10  # Keep last N messages
    
    # Tool Configuration
    WEB_SEARCH_MAX_RESULTS = 5

    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")         # sk-lf-...
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")         # pk-lf-...
    LANGFUSE_BASE_URL   = os.getenv("LANGFUSE_BASE_URL")
    
    APP_ENV     = os.getenv("APP_ENV", "development")              # development | staging | production
    APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
    
    @classmethod
    def validate(cls):
        """Validate required settings"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment")
        if not cls.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY not found in environment")

settings = Settings()