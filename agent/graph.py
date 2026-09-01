import json
import atexit
from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agent.state import AgentState, ToolCall
from agent.prompts import SYSTEM_PROMPT, REACT_PROMPT, FINAL_ANSWER_PROMPT
from tools import web_search_tool, knowledge_base_tool
from config.settings import settings
from utils.logger import logger
from utils.memory_store import conversation_memory

from langfuse import observe, get_client

# Single Langfuse client — shared across all nodes
langfuse = get_client()
atexit.register(langfuse.flush)  # FIX: flush on process exit

# Initialize LLM
llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model_name=settings.MODEL_NAME,
    temperature=settings.TEMPERATURE,
    max_tokens=settings.MAX_TOKENS
)

@observe(name="reasoning-node")
def reasoning_node(state: AgentState) -> AgentState:
    """
    Agent analyzes the query and outputs structured decision.
    
    Responsibilities:
    - Understand the question
    - Decide on action (search, query KB, or answer)
    - Extract optimal search query if needed
    
    Output: Structured JSON with reasoning, action, and search_query
    """
    logger.log_iteration(state["iteration"], settings.MAX_ITERATIONS)
    
    # Get context from memory
    context = conversation_memory.get_context()
    metadata_context = conversation_memory.get_metadata_context()
    full_context = f"{metadata_context}\n{context}"
    
    # Check if we already used tools - if so, we should answer now
    if state["tool_calls"]:
        logger.log_action("Already gathered information, ready to answer", "decision")
        state["ready_to_answer"] = True
        state["action"] = "answer_directly"
        return state
    
    # Format the reasoning prompt
    prompt = REACT_PROMPT.format(
        query=state["query"],
        context=full_context
    )
    
    # Get structured response from LLM
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=full_context)),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        raw_content = response.content.strip()
        
        # Clean up markdown formatting if present
        if raw_content.startswith("```json"):
            raw_content = raw_content.replace("```json", "").replace("```", "").strip()
        elif raw_content.startswith("```"):
            raw_content = raw_content.replace("```", "").strip()
        
        # Parse JSON response
        decision = json.loads(raw_content)
        
        # Validate required fields
        if "reasoning" not in decision or "action" not in decision:
            raise ValueError("Missing required fields in JSON response")
        
        reasoning = decision["reasoning"]
        action = decision["action"]
        search_query = decision.get("search_query")
        
        logger.log_thought(f"{reasoning} → Action: {action}")
        
        # Update state with structured output
        state["thought"] = reasoning
        state["action"] = action
        state["search_query"] = search_query
        
        # Determine if ready to answer
        if action == "answer_directly":
            state["ready_to_answer"] = True
        elif action in ["use_web_search", "use_knowledge_base"]:
            state["ready_to_answer"] = False
            
            # Validate search query
            if not search_query or len(search_query.strip()) < 2:
                logger.log_error(f"Invalid search query: '{search_query}', using original question")
                search_query = state["query"]
            
            # Store structured tool call
            state["tool_calls"].append(ToolCall(
                tool=action.replace("use_", ""),
                query=search_query,
                reasoning=reasoning
            ))
        else:
            logger.log_error(f"Unknown action: {action}, defaulting to answer")
            state["ready_to_answer"] = True
    
    except json.JSONDecodeError as e:
        logger.log_error(f"Failed to parse JSON response: {e}")
        logger.log_error(f"Raw response: {raw_content[:200]}")
        # Fallback: answer directly with existing knowledge
        state["ready_to_answer"] = True
        state["action"] = "answer_directly"
        state["thought"] = "Unable to parse structured response, answering with existing knowledge"
    
    except Exception as e:
        logger.log_error(f"Reasoning error: {e}")
        state["ready_to_answer"] = True
        state["action"] = "answer_directly"
        state["thought"] = f"Error during reasoning: {str(e)}"
    
    state["iteration"] += 1
    return state

@observe(name="tool-execution-node")
def tool_execution_node(state: AgentState) -> AgentState:
    """
    Pure tool executor - NO reasoning, NO prompt engineering.
    
    Responsibilities:
    - Execute tool with pre-determined query
    - Capture observation
    - Handle tool failures gracefully
    
    Input: state["tool_calls"] with structured ToolCall objects
    Output: Observation added to messages
    """
    
    if not state["tool_calls"]:
        logger.log_error("Tool execution called with no tool_calls")
        state["ready_to_answer"] = True
        return state
    
    # Get the most recent tool call
    tool_call = state["tool_calls"][-1]
    tool_name = tool_call["tool"]
    query = tool_call["query"]
    
    logger.log_action(f"Executing {tool_name} with query: '{query}'", tool_name)
    
    try:
        # Execute the appropriate tool
        if tool_name == "web_search":
            observation = web_search_tool.invoke({"query": query})
        elif tool_name == "knowledge_base":
            # FIX: use semantic RAG instead of dict lookup
            from tools.rag_tool import rag_tool
            observation = rag_tool.invoke({"query": query})
        else:
            observation = f"Error: Unknown tool '{tool_name}'"
            logger.log_error(f"Unknown tool: {tool_name}")
        
        # Validate observation
        if not observation or len(observation.strip()) < 10:
            logger.log_error(f"Tool returned empty/invalid result")
            observation = f"Tool '{tool_name}' returned no useful information for query '{query}'"
        
        logger.log_observation(observation)
        
        # Store observation in messages
        state["messages"].append(AIMessage(
            content=f"Tool: {tool_name}\nQuery: {query}\nResult:\n{observation}"
        ))
        
        # After successful tool execution, we're ready to answer
        state["ready_to_answer"] = True
        
    except Exception as e:
        error_msg = f"Tool execution failed: {str(e)}"
        logger.log_error(error_msg)
        
        # Store error as observation
        state["messages"].append(AIMessage(
            content=f"Tool: {tool_name}\nQuery: {query}\nError: {error_msg}"
        ))
        
        # Even on failure, proceed to answer (agent will acknowledge limitation)
        state["ready_to_answer"] = True
    
    return state

@observe(name="answer-node")
def answer_node(state: AgentState) -> AgentState:
    """
    Generate final answer based on all gathered information.
    
    Responsibilities:
    - Synthesize all observations
    - Generate comprehensive response
    - Update conversation memory
    """
    
    # Compile all information gathered
    reasoning_history = f"Initial reasoning: {state['thought']}\n\n"
    
    for msg in state["messages"]:
        if isinstance(msg, AIMessage):
            reasoning_history += f"{msg.content}\n\n"
    
    # Get context
    context = conversation_memory.get_context()
    metadata_context = conversation_memory.get_metadata_context()
    full_context = f"{metadata_context}\n{context}"
    
    # Generate final answer
    prompt = FINAL_ANSWER_PROMPT.format(
        query=state["query"],
        reasoning_history=reasoning_history
    )
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=full_context)),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        final_answer = response.content
        
        logger.log_response(final_answer)
        
        state["final_answer"] = final_answer
        
        # Update memory
        conversation_memory.add_message("assistant", final_answer)
    
    except Exception as e:
        error_msg = f"I apologize, but I encountered an error generating the response: {str(e)}"
        logger.log_error(error_msg)
        state["final_answer"] = error_msg
    
    return state

def should_continue(state: AgentState) -> Literal["tool_execution", "answer", "end"]:
    """
    Routing logic based on state.
    
    Decision tree:
    1. Max iterations → answer
    2. ready_to_answer → answer
    3. Has pending tool call → tool_execution
    4. Otherwise → end (error state)
    """
    
    # Safety: max iterations check
    if state["iteration"] >= settings.MAX_ITERATIONS:
        logger.log_error(f"Max iterations ({settings.MAX_ITERATIONS}) reached")
        return "answer"
    
    # If ready to answer, go to answer
    if state["ready_to_answer"]:
        return "answer"
    
    # If we have a pending tool call (not yet executed)
    if state["tool_calls"]:
        # Check if this tool has been executed
        last_tool = state["tool_calls"][-1]
        tool_query = last_tool["query"]
        
        # Check if we have an observation for this tool
        has_observation = any(
            tool_query in msg.content 
            for msg in state["messages"] 
            if isinstance(msg, AIMessage)
        )
        
        if not has_observation:
            return "tool_execution"
    
    # Default: end (shouldn't reach here in normal flow)
    logger.log_error("Reached unexpected routing state")
    return "answer"

def build_agent_graph():
    """Build the LangGraph ReAct agent with clean separation of concerns"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes (each with single responsibility)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("answer", answer_node)
    
    # Set entry point
    workflow.set_entry_point("reasoning")
    
    # Routing logic
    workflow.add_conditional_edges(
        "reasoning",
        should_continue,
        {
            "tool_execution": "tool_execution",
            "answer": "answer",
            "end": END
        }
    )
    
    # After tool execution, always go to answer
    workflow.add_edge("tool_execution", "answer")
    
    # Answer is terminal
    workflow.add_edge("answer", END)
    
    return workflow.compile()

# Create the agent
agent_graph = build_agent_graph()