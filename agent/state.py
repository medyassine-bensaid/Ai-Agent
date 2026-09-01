from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import add_messages

class ToolCall(TypedDict):
    """Structured tool call with query"""
    tool: str
    query: str
    reasoning: str

class AgentState(TypedDict):
    """State schema for the ReAct agent"""
    
    # The user's query
    query: str
    
    # Conversation history
    messages: Annotated[List, add_messages]
    
    # Agent's reasoning process
    thought: str
    
    # Structured tool calls with extracted queries
    tool_calls: List[ToolCall]
    
    # Current action decision
    action: str  # "use_web_search", "use_knowledge_base", "answer_directly"
    
    # Search query (extracted during reasoning)
    search_query: Optional[str]
    
    # Iteration counter
    iteration: int
    
    # Final answer flag
    ready_to_answer: bool
    
    # Final response
    final_answer: str