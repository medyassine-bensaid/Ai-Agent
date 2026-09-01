from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from config.settings import settings
from utils.logger import logger

# Initialize Tavily search
tavily_search = TavilySearchResults(
    max_results=settings.WEB_SEARCH_MAX_RESULTS,
    api_key=settings.TAVILY_API_KEY
)

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