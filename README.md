# DevOps ReAct Agent: LLMOps reference project

A DevOps question-answering agent built on the **ReAct (Reasoning + Acting)** pattern with
LangGraph, plus the LLMOps tooling around it:

- **Agent**: LangGraph ReAct loop, Groq-hosted `openai/gpt-oss-120b`, conversation memory
- **Tools**: semantic **RAG** over a ChromaDB knowledge base, and **Tavily** web search
- **Observability**: every run traced in self-hosted **Langfuse v4** (spans, tool calls, token usage)
- **Evaluation**: inline LLM-as-a-judge + heuristic scores on every trace, and an
  end-to-end evaluation suite (routing, grounding, RAG, safety, telemetry integrity)
- **CI/CD**: GitHub Actions running lint, unit tests, Docker build and the live evaluation
  against an ephemeral Langfuse; manual self-hosted deployment (Docker Compose / Minikube)

Validation evidence for the current state of the project is in
[`docs/VALIDATION_REPORT.md`](docs/VALIDATION_REPORT.md).

---

## Architecture

```
User question
    │
    ▼
┌───────────┐  decision (JSON)  ┌────────────────┐
│ reasoning │ ────────────────► │ tool_execution │  knowledge_base → RAG (Chroma + MiniLM)
│   (LLM)   │ ◄──────────────── │                │  web_search     → Tavily
└─────┬─────┘   observation     └────────────────┘
      │ ready to answer (or MAX_TOOL_CALLS / MAX_ITERATIONS reached)
      ▼
┌───────────┐
│  answer   │  grounded final answer (cites URLs / kb: sources it was given)
│   (LLM)   │
└─────┬─────┘
      ▼
inline scoring → Langfuse (llm-judge/*, heuristic/quality, has-error)
```

| Phase | Component | Description |
|-------|-----------|-------------|
| State | `agent/state.py` | Query, messages/observations, tool calls, iteration counter, final answer |
| Reasoning | `reasoning_node()` | LLM picks `use_knowledge_base`, `use_web_search` or `answer_directly`, seeing observations gathered so far |
| Tool execution | `tool_execution_node()` | Runs the chosen tool, stores the observation, loops back to reasoning |
| Response | `answer_node()` | Synthesises the answer under explicit grounding rules (`agent/prompts.py`) |
| Memory | `utils/memory_store.py` | Recent conversation + user metadata (in-process, single user) |
| Control flow | `should_continue()` | Routes reasoning → tool / answer; bounded by `MAX_TOOL_CALLS` and `MAX_ITERATIONS` |

**Routing policy:** established concepts covered by the knowledge base (CI/CD, Docker,
Kubernetes, Infrastructure as Code, DevOps culture) → RAG; current or fast-changing facts
(versions, releases, news) → web search; a question needing both triggers both tools;
greetings, off-domain and conversational questions → direct answer.

### Project structure

```
agent/        graph.py (LangGraph workflow) · prompts.py · state.py
tools/        rag_tool.py (Chroma RAG) · web_search.py (Tavily) · knowledge_base.py (seed corpus)
              rag_evaluator.py (RAG-specific judge metrics)
monitoring/   langfuse_scorer.py (inline judge) · langfuse_api.py (v4 read API helpers)
              eval_runner.py (score quality gate) · langfuse_exporter.py (Prometheus) · pr_comment.py
evals/        dataset.json (18 end-to-end cases) · run_eval.py (runner + telemetry verification)
tests/        unit + regression tests (no network)
config/       settings.py
main.py       CLI entry point (run_agent / run_agent_detailed)
.github/      workflows/cicd.yml (CI) · workflows/deploy.yml (manual self-hosted deploy)
k8s/          Kustomize manifests for Minikube
```

---

## Setup

Requirements: Python 3.11, Docker (for Langfuse), a [Groq](https://console.groq.com) API key
and a [Tavily](https://tavily.com) API key.

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock
cp .env.example .env                # then fill in the keys
```

`requirements.lock` pins the exact Linux/CPU dependency set used by CI and the Docker
image; `requirements.txt` lists the direct dependencies it was compiled from.

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | yes | Agent LLM and LLM judge |
| `TAVILY_API_KEY` | yes | Web search |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | yes | Project API keys from Langfuse → Settings |
| `LANGFUSE_BASE_URL` | yes | `http://localhost:3000` for a local Langfuse |
| `JUDGE_MODEL` | no | Judge model (default `openai/gpt-oss-120b`) |
| `RAG_MIN_SCORE` | no | Minimum cosine similarity for retrieved chunks (default `0.3`) |
| `CHROMA_PATH` | no | Vector store location (default `data/chroma`) |

### Langfuse (self-hosted)

The project targets Langfuse **v4** (tested with server 4.50.0, Python SDK 4.16). Run it with
the upstream compose file (`git clone https://github.com/langfuse/langfuse && docker compose up -d`),
create a project, and copy its API keys into `.env`. Change every `# CHANGEME` secret in
that compose file before exposing it beyond your machine.

---

## Usage

```bash
python main.py
```

```
You: What is CI/CD?
💭 THOUGHT: Foundational concept covered by the knowledge base → Action: use_knowledge_base
🔧 ACTION: Using tool 'knowledge_base' - Executing knowledge_base with query: 'CI/CD'
👁️  OBSERVATION: Retrieved 2 relevant chunks: [1] (score=0.71, source=kb:ci/cd) ...
✨ FINAL RESPONSE: CI/CD stands for Continuous Integration / Continuous Deployment ...
```

Commands: `exit` / `quit` end the session, `clear` resets the conversation memory.

Add documents to the knowledge base from Python:

```python
from tools.rag_tool import ingest_file
ingest_file("docs/runbook.md")   # chunks, embeds and upserts; changed chunks are re-embedded
```

---

## Testing and evaluation

| Layer | Command | Needs network |
|---|---|---|
| Lint | `ruff check .` | no |
| Unit + regression tests (Groq/Tavily/Langfuse mocked) | `pytest` | no |
| End-to-end evaluation (real Groq, Tavily, RAG, judge, Langfuse) | `python -m evals.run_eval` | yes |
| Single eval case | `python -m evals.run_eval --case RAG-01` | yes |
| Must-pass cases only (6, ~1/3 of the tokens) | `python -m evals.run_eval --critical-only` | yes |
| Quality gate over recent Langfuse scores | `python -m monitoring.eval_runner` | Langfuse |
| RAG judge metrics on recent RAG traces | `python -m tools.rag_evaluator` | yes |

The evaluation suite (`evals/dataset.json`) covers general questions, RAG (direct,
paraphrased, multi-chunk, identifiers), web search, questions needing both tools, unknown
answers, off-domain questions, prompt-injection/secret exfiltration, harmful requests, long
inputs, a real Tavily failure (invalid key), ambiguity and multi-turn memory. For every case it checks:

- **routing**: which tools actually ran (required / forbidden)
- **RAG evidence**: the expected `kb:` source was retrieved
- **grounding**: every URL in the answer appears in the tool evidence (no fabricated links)
- **behaviour**: required content, unknown-answer acknowledgement, refusals, tool-failure acknowledgement
- **secrets**: no API key value appears in the answer
- **LLM judge**: the judge ran and relevance / faithfulness / hallucination scores meet thresholds
- **telemetry**: the trace, its generation and tool observations, and every submitted score
  can be read back from the Langfuse API with the same values

Reports are written to `reports/eval_report.md` and `reports/eval_report.json`. Exit codes:
`0` gate passed (pass rate ≥ `EVAL_MIN_PASS_RATE`, default 0.8, and every critical case passed),
`1` gate failed, `2` could not run (missing configuration / Langfuse unreachable). An
evaluation that could not run is never reported as a pass.

---

## Observability

Each `run_agent` call produces one Langfuse trace (`agent-run`) containing the LangGraph node
spans, `ChatGroq` generations (model, token usage), the `rag_tool` / `rag-retrieval` and
`web_search_tool` / `tavily_search_results_json` tool spans, and these scores:

| Score | Type | Meaning |
|---|---|---|
| `llm-judge/relevance`, `llm-judge/faithfulness`, `llm-judge/completeness` | NUMERIC 0–1 | Judge scores; faithfulness is judged against the tool evidence |
| `llm-judge/hallucination_risk` | NUMERIC 0–1 | **1.0 = grounded**, 0.0 = invented facts |
| `llm-judge/error` | BOOLEAN | Present only if the judge failed (no judge scores were written) |
| `heuristic/quality` | NUMERIC 0–1 | Length/structure heuristic |
| `has-error` | BOOLEAN | 1 = the answer is an error message |
| `eval/passed`, `eval/routing-correct` | BOOLEAN | Written by the evaluation suite (traces tagged `eval`) |

See [`MONITORING_LANGFUSE_README.md`](MONITORING_LANGFUSE_README.md) for details, including
the Langfuse v4 read APIs, the Prometheus exporter and Grafana.

---

## CI/CD

`.github/workflows/cicd.yml` (GitHub-hosted runners, on push / PR / manual):

1. **Lint**: `ruff check` (no auto-fix in CI)
2. **Unit tests**: `pytest` with JUnit + coverage reports uploaded as artifacts
3. **Docker build**: builds the image and runs the healthcheck and import smoke test inside it
4. **Live evaluation**: starts an ephemeral Langfuse v4 (upstream compose, headless init with
   throwaway keys), runs the evaluation suite (the 6 critical cases on push/PR; all 18 cases
   when dispatched manually with `full_eval`), then the score quality gate
   `python -m monitoring.eval_runner`; uploads `reports/` and writes the report to the job summary

Repository secrets required for step 4: `GROQ_API_KEY`, `TAVILY_API_KEY`. On pushes the job
fails if they are missing; on pull requests from forks (which never receive secrets) it is
skipped with a warning.

`.github/workflows/deploy.yml` (manual `workflow_dispatch`, self-hosted runner) builds and pushes
the image to Docker Hub, deploys with Docker Compose and/or Minikube, and runs the post-deploy
score gate. It needs `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, the API key secrets above,
`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`, and the repository variables `RUNNER_HOME`
(Minikube) and optionally `LANGFUSE_BASE_URL`.

---

## Extending the agent

**Add a tool:** implement it in `tools/` as a LangChain `@tool`, add an action
(`use_<tool>`) to `REACT_PROMPT` in `agent/prompts.py`, dispatch it in
`tool_execution_node()`, map it in `TOOL_OBSERVATIONS` in `evals/run_eval.py`, and add
evaluation cases to `evals/dataset.json`.

**Change domain:** edit `SYSTEM_PROMPT` in `agent/prompts.py`, replace the seed corpus in
`tools/knowledge_base.py` (or ingest your own documents), and update the evaluation dataset.

**Tuning:** `config/settings.py` holds `TEMPERATURE`, `MAX_ITERATIONS`, `MAX_TOOL_CALLS`,
`WEB_SEARCH_MAX_RESULTS`, `WEB_SEARCH_TIMEOUT` and `LLM_MAX_RETRIES`.

## Known limitations

- Groq's free tier allows 200 000 tokens/day for `openai/gpt-oss-120b`. One full evaluation
  run (agent + judge, including reasoning tokens) uses a large share of that, and a run that
  hits the limit fails rather than passes. Use `--critical-only`, a paid tier, or a
  different `JUDGE_MODEL` (Groq quotas are per model).

- Conversation memory is a process-global singleton: the CLI is single-user and concurrent
  sessions in one process would share memory.
- The built-in knowledge base is small (5 topics); answers outside it rely on the model or web search.
- LLM routing and judge scores are non-deterministic; the evaluation gate uses a pass-rate
  threshold plus must-pass critical cases rather than requiring every case to pass every run.
