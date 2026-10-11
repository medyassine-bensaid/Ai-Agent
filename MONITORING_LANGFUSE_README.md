# Monitoring & Langfuse Integration

The observability stack has four layers:

1. **Tracing**: every agent run is a Langfuse trace with spans for each graph node, LLM
   generation and tool call (Langfuse Python SDK v4, OpenTelemetry-based).
2. **Inline scoring**: after each answer an LLM-as-a-judge (Groq) and a heuristic write
   scores onto the trace.
3. **Evaluation**: `evals/run_eval.py` runs a fixed case set end to end and verifies the
   resulting traces and scores through the Langfuse API; `monitoring/eval_runner.py` gates
   on recent scores.
4. **Infra metrics**: `monitoring/langfuse_exporter.py` republishes score averages as
   Prometheus gauges for Grafana.

Target versions: Langfuse server **v4** (validated on 4.50.0), `langfuse` Python SDK 4.16.

---

## 1. Tracing (`main.py`, `agent/graph.py`, `tools/rag_tool.py`)

- `main.py` creates one Langfuse client (`get_client()`) and one LangChain
  `CallbackHandler`, flushes on exit and on `SIGTERM`.
- `run_agent_detailed()` is decorated with `@observe(name="agent-run")`: this is the trace
  root. The trace output is set explicitly to the answer text; on an agent exception the span
  is marked `level=ERROR` with a status message. `session_id` is propagated with
  `propagate_attributes`.
- Graph nodes are `@observe` spans (`reasoning-node`, `tool-execution-node`, `answer-node`);
  the callback handler adds the LangGraph chain spans and `ChatGroq` **generations** (model
  name and token usage).
- Tool spans: `web_search_tool` → `tavily_search_results_json`, and `rag_tool` →
  `rag-retrieval` (typed `retriever`; its output is the retrieved chunks with source and
  similarity score; metadata holds `num_docs`, `top_score`, `min_score`).

Resulting trace tree (as returned by the observations API):

```
agent-run (SPAN, root)
└─ LangGraph (CHAIN)
   ├─ reasoning (CHAIN) ─ reasoning-node (SPAN) ─ ChatGroq (GENERATION, usage)
   ├─ tool_execution (CHAIN) ─ tool-execution-node (SPAN) ─ web_search_tool (TOOL) ─ tavily_search_results_json (TOOL)
   ├─ reasoning ... (repeats while the agent gathers information)
   └─ answer (CHAIN) ─ answer-node (SPAN) ─ ChatGroq (GENERATION, usage)
```

## 2. Inline scoring (`monitoring/langfuse_scorer.py`)

`inline_score(trace_id, question, answer, evidence)` is called by `run_agent_detailed()`
for every answer. The judge prompt receives the question, the answer and the **tool
evidence** (observations) the answer was generated from, so faithfulness is judged against
what was actually retrieved.

| Score | Type | Meaning |
|---|---|---|
| `llm-judge/relevance` | NUMERIC 0–1 | Answer addresses the question |
| `llm-judge/faithfulness` | NUMERIC 0–1 | Claims supported by the evidence or stable domain knowledge |
| `llm-judge/completeness` | NUMERIC 0–1 | Covers what the question needs |
| `llm-judge/hallucination_risk` | NUMERIC 0–1 | **1.0 = grounded, 0.0 = invented facts** (name kept for dashboard continuity) |
| `llm-judge/error` | BOOLEAN | Written instead of the four scores above when the judge fails |
| `heuristic/quality` | NUMERIC 0–1 | Length / structure heuristic |
| `has-error` | BOOLEAN | **1 = the answer is an error message**, 0 = normal answer |

Judge implementation notes:

- `openai/gpt-oss-120b` is a reasoning model: hidden reasoning tokens count against
  `max_tokens`. The budget is `JUDGE_MAX_TOKENS` (default 2048) with `reasoning_effort=low`
  and JSON mode. With the old budget of 300 tokens, the judge returned empty content for
  realistic answers and **no judge scores were written**.
- Output is validated: all four metrics must be numbers in [0, 1]. Invalid output,
  truncation or API errors are retried once (with backoff) and then recorded as
  `llm-judge/error`, never silently skipped and never defaulted to a number.
- Up to `JUDGE_EVIDENCE_CHARS` (default 20 000) characters of evidence are passed;
  truncating at 6 000 hid the facts the judge needed to verify.

> **Historical data:** before this fix `has-error` was written inverted (1 = no error), and
> no `llm-judge/*` scores exist for traces scored with the 300-token budget.

RAG-specific judge metrics (`rag/context_relevance`, `rag/faithfulness`,
`rag/answer_relevance`) are produced by `python -m tools.rag_evaluator`, which finds recent
traces containing a `rag_tool` span and judges the retrieved context against the answer.

## 3. Reading data back: Langfuse v4 APIs (`monitoring/langfuse_api.py`)

Self-hosted Langfuse v4 runs in **events_only** mode, where the legacy read endpoints return
**404**: `GET /api/public/traces`, `GET /api/public/traces/{id}` and `GET /api/public/scores`.
All read code goes through `monitoring/langfuse_api.py`, which uses:

| Purpose | Endpoint | Notes |
|---|---|---|
| Observations (incl. the `agent-run` root of each trace) | `GET /api/public/v2/observations` | `fromStartTime`/`toStartTime`, `traceId`, JSON `filter`, `fields=core,basic,usage,model,io`, cursor pagination |
| Scores | `GET /api/public/v3/scores` | `traceId=a,b`, `fromTimestamp`, `name`, `fields=details,subject`, cursor pagination |

HTTP errors propagate. Callers decide how to fail, and none of them treat "could not read"
as success.

## 4. Quality gates

**`python -m evals.run_eval`** (see README): runs the 18-case suite through the real agent,
writes `eval/passed` and `eval/routing-correct` scores onto each trace (traces are tagged
`eval` and the case category, session `eval-<run>-<case>`), then polls Langfuse until every
trace, its generation/tool observations and every submitted score are readable with the
submitted values. Exit 0 / 1 / 2 = passed / failed / could not run.

**`python -m monitoring.eval_runner`**: post-deploy gate over the `agent-run` traces of the
last `EVAL_WINDOW_HOURS` (default 1):

| Check | Threshold (env) |
|---|---|
| mean `heuristic/quality` | ≥ `QUALITY_THRESHOLD` (0.5) |
| mean `llm-judge/relevance`, `faithfulness`, `hallucination_risk` | ≥ `JUDGE_THRESHOLD` (0.7) each; missing judge scores fail |
| mean `has-error` | ≤ `MAX_ERROR_RATE` (0.2) |

Exit 1 when a check fails, **exit 2 when Langfuse is unreachable or no scored trace exists**
(previously both cases exited 0, so the gate could never fail).

## 5. Prometheus exporter and Grafana

`python -m monitoring.langfuse_exporter` (started inside the agent container by
`docker-compose.yml`) polls the v3 scores API every 30 s over the last
`EXPORTER_WINDOW_HOURS` (default 24) and serves on `:8001/metrics`:

`llm_judge_relevance`, `llm_judge_faithfulness`, `llm_judge_completeness`,
`llm_judge_hallucination_risk`, `llm_heuristic_quality`, `llm_error_rate` (mean of
`has-error`), `llm_scores_fetched`.

Prometheus (`monitoring/prometheus.yml`, host port 9080) scrapes `agent:8001`; Grafana
(host port `${GRAFANA_PORT:-3001}`) visualises it. Ports are bound to 127.0.0.1.

## 6. Container wiring (`docker-compose.yml`)

Inside the container `localhost:3000` is not the host's Langfuse, so the agent uses
`LANGFUSE_DOCKER_BASE_URL` (default `http://host.docker.internal:3000`, with an
`extra_hosts` host-gateway mapping). The Chroma store is persisted in the `chroma_data`
volume. The healthcheck (`scripts/healthcheck.py`) validates configuration and the Langfuse
client.

## 7. CI

`.github/workflows/cicd.yml` job **"4 · Live evaluation"** starts an ephemeral Langfuse v4
from the upstream `docker-compose.yml` (pinned commit) with headless initialisation
(`LANGFUSE_INIT_*`, random throwaway keys), runs the evaluation suite and the score gate
against it, uploads `reports/` and writes the report to the job summary. No production
Langfuse credentials are needed in CI.

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `404 ... not available on deployments running in Langfuse v4 events_only mode` | Code calling a legacy endpoint; use `monitoring/langfuse_api.py` |
| No `llm-judge/*` scores, only `llm-judge/error` | See the score comment; usually Groq rate limits (429) or an invalid key |
| Traces missing after a script exits | Call `get_client().flush()` before exit (`main.py` does on exit/SIGTERM) |
| Scores visible via API but not in a dashboard | Check the project, time range and environment filters; score names are case-sensitive |
