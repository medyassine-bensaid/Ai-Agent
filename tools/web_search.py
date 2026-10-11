from typing import Dict

import requests
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import (
    TAVILY_API_URL,
    TavilySearchAPIWrapper,
)
from langchain_core.tools import tool

from config.settings import settings
from utils.logger import logger


class _TimeoutTavilyWrapper(TavilySearchAPIWrapper):
    """langchain_community's wrapper posts to Tavily without a timeout, so one
    stalled connection hangs the whole agent run. Same request, bounded wait."""

    # Same parameter order as TavilySearchAPIWrapper.raw_results: the tool passes them positionally
    def raw_results(self, query: str, max_results=5, search_depth="advanced",
                    include_domains=None, exclude_domains=None, include_answer=False,
                    include_raw_content=False, include_images=False) -> Dict:
        params = {
            "api_key": self.tavily_api_key.get_secret_value(),
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_domains": include_domains or [],
            "exclude_domains": exclude_domains or [],
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_images": include_images,
        }
        response = requests.post(f"{TAVILY_API_URL}/search", json=params,
                                 timeout=settings.WEB_SEARCH_TIMEOUT)
        response.raise_for_status()
        return response.json()


def make_tavily_search(api_key: str) -> TavilySearchResults:
    return TavilySearchResults(
        max_results=settings.WEB_SEARCH_MAX_RESULTS,
        api_wrapper=_TimeoutTavilyWrapper(tavily_api_key=api_key or "missing"),
    )


# Initialize Tavily search
tavily_search = make_tavily_search(settings.TAVILY_API_KEY)

@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for current DevOps information, tools, trends, and updates.
    Use this when you need information about:
    - Recent tool versions or features
    - Current best practices
    - Latest DevOps trends
    - Comparisons between tools
    - Recent blog posts or documentation

    Args:
        query: The search query (keep it concise and specific)

    Returns:
        Search results with relevant information
    """
    try:
        logger.log_action(f"Searching web for: '{query}'", "web_search")
        results = tavily_search.invoke({"query": query})

        if isinstance(results, str):
            # TavilySearchResults returns the exception text instead of raising
            raise RuntimeError(results)
        if not results:
            return "No results found. Try a different search query."

        # Format results for better readability
        formatted_results = "Search Results:\n\n"
        for i, result in enumerate(results, 1):
            formatted_results += f"{i}. {result.get('content', 'No content')}\n"
            formatted_results += f"   Source: {result.get('url', 'No URL')}\n\n"

        logger.log_observation(f"Found {len(results)} results")
        return formatted_results

    except Exception as e:
        error_msg = f"Web search failed: {str(e)}"
        logger.log_error(error_msg)
        return error_msg
