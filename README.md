This repository provides a basic AI agent using the **ReAct (Reasoning + Acting)** pattern. it demonstrates the fundamental components required to create autonomous AI agents capable of reasoning, tool usage, and multi-turn conversations.

**Use Case:** DevOps  (you chnage the domain you want !)

---

### ReAct Pattern

```
┌─────────────────────────────────────────────┐
│  User Query: "What's the latest in K8s?"   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
          ┌────────────────┐
          │   REASONING    │ ◄──┐
          │  (Decide what  │    │
          │   to do next)  │    │
          └────────┬───────┘    │
                   │             │
                   ▼             │
          ┌────────────────┐    │
          │     ACTION     │    │
          │  (Use tools:   │    │
          │  search, query)│    │
          └────────┬───────┘    │
                   │             │
                   ▼             │
          ┌────────────────┐    │
          │  OBSERVATION   │    │
          │ (Process tool  │    │
          │    results)    │────┘
          └────────┬───────┘
                   │
                   ▼
          ┌────────────────┐
          │    RESPONSE    │
          │ (Final answer) │
          └────────────────┘
```

---

## 🏗️ Architecture Components

### Core Phases (Non-Negotiable)

| Phase | Component | Description |
|-------|-----------|-------------|
| **1. State Management** | `state.py` | Maintains conversation state, tool calls, and reasoning history |
| **2. Reasoning** | `reasoning_node()` | LLM decides next action based on query and context |
| **3. Tool Execution** | `tool_execution_node()` | Executes selected tools without additional reasoning |
| **4. Observation** | State messages | Captures and stores tool outputs |
| **5. Response Generation** | `answer_node()` | Synthesizes all information into final answer |
| **6. Memory** | `memory_store.py` | Maintains conversation history and user context |
| **7. Control Flow** | `should_continue()` | Routes between reasoning, action, and response |

---

## 📁 Project Structure

```
.
├── agent/
│   ├── graph.py          # LangGraph workflow (reasoning loop)
│   ├── state.py          # Agent state schema
│   └── prompts.py        # System and task prompts
├── tools/
│   ├── web_search.py     # Tavily web search integration
│   └── knowledge_base.py # Static knowledge retrieval
├── utils/
│   ├── logger.py         # Colored console logging
│   └── memory_store.py   # Conversation memory management
├── config/
│   └── settings.py       # Configuration and API keys
├── main.py               # CLI entry point
├── requirements.txt      # Python dependencies
└── .env.example          # Environment variables template
```

---

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-agent-architecture.git
   cd ai-agent-architecture
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

   Required variables:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   MODEL_NAME=llama-3.3-70b-versatile
   TEMPERATURE=0.1
   MAX_TOKENS=2048
   MAX_ITERATIONS=5
   WEB_SEARCH_MAX_RESULTS=3
   ```

5. **Run the agent**
   ```bash
   python main.py
   ```

---

## 💻 Usage Examples

### Basic Interaction

```
You: What is CI/CD?

Agent is thinking...

💭 THOUGHT: This is a foundational DevOps concept → Action: use_knowledge_base
🔧 ACTION: Using tool 'knowledge_base' - Querying for: 'CI/CD'
👁️ OBSERVATION: Found 1 knowledge base entries

✨ FINAL RESPONSE:
CI/CD stands for Continuous Integration/Continuous Deployment...
[Full response with examples and best practices]
```

### Web Search for Current Information

```
You: What are the latest features in Kubernetes 1.30?

Agent is thinking...

💭 THOUGHT: Need current version information → Action: use_web_search
🔧 ACTION: Using tool 'web_search' - Searching for: 'Kubernetes 1.30 features'
👁️ OBSERVATION: Found 3 results

✨ FINAL RESPONSE:
Kubernetes 1.30 introduces several key features...
[Response with citations from web sources]
```

### Commands

- `exit` or `quit` - End session
- `clear` - Clear conversation history

---

### Adjusting Agent Behavior

**Temperature** (Creativity vs Consistency)
```python
TEMPERATURE = 0.1  # More deterministic (recommended for agents)
# 0.0 = Deterministic, 1.0 = Creative
```
**Max Iterations** (Safety limit)
```python
MAX_ITERATIONS = 5  # Prevent infinite loops
```
---

## 🛠️ Extending the Agent

### Adding New Tools

1. **Create tool file** in `tools/`
   ```python
   # tools/calculator.py
   from langchain_core.tools import tool
   
   @tool
   def calculator_tool(expression: str) -> str:
       """Evaluate mathematical expressions."""
       try:
           result = eval(expression)  # Use safe_eval in production
           return f"Result: {result}"
       except Exception as e:
           return f"Error: {str(e)}"
   ```

2. **Import in graph.py**
   ```python
   from tools.calculator import calculator_tool
   ```

3. **Add to action options** in `prompts.py`
   ```python
   "action": "use_web_search" OR "use_knowledge_base" OR "use_calculator" OR "answer_directly"
   ```

### Adapting to Different Domains

Modify `agent/prompts.py`:

```python
SYSTEM_PROMPT = """You are a [Domain] Expert Agent...

Your expertise covers:
- [Topic 1]
- [Topic 2]
- [Topic 3]

Your role:
1. Explain [domain] concepts clearly
2. Provide practical guidance
...
"""
```

---

## 📊 How It Works: Deep Dive

### 1. State Flow

```python
AgentState {
    query: str              # User's question
    messages: List          # Conversation history
    thought: str            # Current reasoning
    tool_calls: List[ToolCall]  # Structured tool executions
    action: str             # Current action decision
    iteration: int          # Loop counter
    ready_to_answer: bool   # Flag to generate response
    final_answer: str       # Generated response
}
```

### 2. Decision Making (Reasoning Node)

The agent uses **structured JSON output** to make decisions:

```json
{
  "reasoning": "User asks about current info, need web search",
  "action": "use_web_search",
  "search_query": "Kubernetes 1.30 features"
}
```

### 3. Tool Execution (Separation of Concerns)

**Reasoning Node:** Decides *what* to do  
**Tool Execution Node:** Does *how* to do it

This separation prevents:
- Infinite reasoning loops
- Tool misuse
- Unclear execution flow

### 4. Memory Management

**Conversation Context:**
```
Recent conversation:
USER: What is Docker?
ASSISTANT: Docker is a containerization platform...
USER: How does it compare to VMs?
```

**Metadata:**
```python
{
    "user_expertise": "intermediate",
    "topics_discussed": ["docker", "containers"],
    "user_stack": ["python", "kubernetes"]
}
```

## ⚠️ Common Pitfalls & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Infinite loops | No max iterations | Set `MAX_ITERATIONS` limit |
| Hallucinations | LLM inventing facts | Use web search for current info |
| Poor tool selection | Vague prompts | Provide clear tool descriptions |
| Context loss | No memory | Implement conversation history |
| JSON parsing errors | Unstructured outputs | Enforce JSON format in prompts |

---

<div align="center">


</div>
# AI-agent
