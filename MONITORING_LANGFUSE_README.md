# Monitoring & Langfuse Integration 

---


The stack has three layers:

1. **Tracing** — every agent run and every graph node is wrapped in a Langfuse trace/span.
2. **Scoring** — after each answer, an LLM-as-judge (via Groq) plus heuristics push numeric scores back onto the trace.
3. **Infra metrics** — a standalone exporter polls Langfuse's REST API and republishes the score averages as Prometheus gauges, which Grafana visualizes.

---

## 1. Tracing setup — `main.py`

The CLI entry point creates a single shared Langfuse client and a single shared
`CallbackHandler`, registers a `flush()` on process exit and on `SIGTERM`, and wraps
`run_agent()` with `@observe`:

```python
from langfuse.langchain import CallbackHandler
from langfuse import observe, get_client

# ── Langfuse setup ───────────────────────────────────────────────────────────
langfuse_client = get_client()
langfuse_handler = CallbackHandler()   # one instance, reused across all runs

def _flush_langfuse():
    """Ensure all buffered traces are sent before process dies."""
    try:
        langfuse_client.flush()
    except Exception:
        pass

atexit.register(_flush_langfuse)

# Handle SIGTERM (docker stop) gracefully
def _sigterm_handler(signum, frame):
    _flush_langfuse()
    sys.exit(0)

signal.signal(signal.SIGTERM, _sigterm_handler)
```
*(`main.py`, lines 11–32)*

`run_agent()` is decorated with `@observe(name="agent-run")`, invokes the graph with the
shared callback handler, and — in a `finally` block — triggers inline scoring and a flush
on every single call, regardless of success or failure:

```python
@observe(name="agent-run")
def run_agent(query: str) -> str:
    """Run the agent with a query and score the output inline."""
    ...
    answer = ""
    try:
        final_state = agent_graph.invoke(
            initial_state,
            config={"callbacks": [langfuse_handler]}
        )
        answer = final_state["final_answer"]
        return answer

    except Exception as e:
        error_msg = f"Agent error: {str(e)}"
        logger.log_error(error_msg)
        answer = f"I apologize, but I encountered an error: {str(e)}\nPlease try rephrasing your question."
        return answer

    finally:
        # ── Inline scoring using the Langfuse client's own trace ID method ──
        try:
            trace_id = langfuse_client.get_current_trace_id()
            if trace_id and answer:
                inline_score = _get_scorer()
                if inline_score:
                    inline_score(trace_id, query, answer)
        except Exception as e:
            logger.log_error(f"Inline scoring failed: {e}")

        _flush_langfuse()
```
*(`main.py`, lines 58–99)*

The scorer import is deliberately lazy so a missing module or missing `GROQ_API_KEY`
never crashes the CLI:

```python
def _get_scorer():
    try:
        from monitoring.langfuse_scorer import inline_score
        return inline_score
    except Exception as e:
        logger.log_error(f"Scorer import failed: {e}")
        return None
```
*(`main.py`, lines 35–41)*

---

## 2. Per-node tracing — `agent/graph.py`

The three-node ReAct graph shares a second Langfuse client (also with an `atexit` flush)
and decorates each node individually, so every reasoning step, tool call, and answer
generation shows up as its own span nested under the `agent-run` trace from `main.py`:

```python
from langfuse import observe, get_client

# Single Langfuse client — shared across all nodes
langfuse = get_client()
atexit.register(langfuse.flush)  # FIX: flush on process exit
```
*(`agent/graph.py`, lines 15–19)*

```python
@observe(name="reasoning-node")
def reasoning_node(state: AgentState) -> AgentState:
    ...
```
*(`agent/graph.py`, line 29)*

```python
@observe(name="tool-execution-node")
def tool_execution_node(state: AgentState) -> AgentState:
    ...
```
*(`agent/graph.py`, line 133)*

```python
@observe(name="answer-node")
def answer_node(state: AgentState) -> AgentState:
    ...
```
*(`agent/graph.py`, line 200)*

---

## 3. Inline LLM-as-judge scoring — `monitoring/langfuse_scorer.py`

Every trace is scored immediately after the answer is produced. The judge model runs
through Groq's **OpenAI-compatible endpoint** (Langfuse's native evaluators don't talk to
Groq directly), and a Langfuse client is created fresh per call:

```python
def _make_lf():
    return Langfuse(
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        host=os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000"),
    )

groq_judge = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)
JUDGE_MODEL = "llama-3.3-70b-versatile"
```
*(`monitoring/langfuse_scorer.py`, lines 7–18)*

The judge prompt asks for strict JSON with four metrics:

```python
JUDGE_PROMPT = """\
You are an expert evaluator for a DevOps AI assistant.
Reply ONLY with raw JSON, no markdown:
{{"relevance":0.0,"faithfulness":0.0,"completeness":0.0,"hallucination_risk":0.0,"reasoning":"one line"}}
hallucination_risk: 1.0=grounded, 0.0=hallucinated
Question: {question}
Answer: {answer}"""
```
*(`monitoring/langfuse_scorer.py`, lines 20–26)*

A cheap, deterministic heuristic score runs alongside the LLM judge (length, apology
phrases, numbered lists, bullet lists, code fences):

```python
def _heuristic(answer: str) -> float:
    if not answer: return 0.0
    s = 0.4 if len(answer)>500 else (0.3 if len(answer)>200 else 0.1)
    if re.search(r"i apologize|error generating", answer, re.I): s -= 0.3
    if re.search(r"\d+\.", answer): s += 0.2
    if re.search(r"^[-*]", answer, re.M): s += 0.2
    if re.search(r"```", answer): s += 0.2
    return max(0.0, min(1.0, s))
```
*(`monitoring/langfuse_scorer.py`, lines 42–49)*

`inline_score()` is the function called from `main.py`'s `finally` block. It pushes four
`NUMERIC` LLM-judge scores, one `NUMERIC` heuristic score, and one `BOOLEAN` `has-error`
score onto the given `trace_id`, then flushes:

```python
def inline_score(trace_id: str, question: str, answer: str):
    """Called from main.py with the current trace_id, query, and answer.
    Creates a Langfuse client and pushes evaluation scores."""
    if not trace_id or not answer:
        print(f"  [scorer] SKIP — trace_id={trace_id!r} answer_len={len(answer or '')}")
        return

    lf_client = _make_lf()
    print(f"  [scorer] scoring trace {trace_id[:12]}…")
    judge = _llm_judge(question, answer)

    for metric in ("relevance","faithfulness","completeness","hallucination_risk"):
        v = judge.get(metric)
        if isinstance(v, (int,float)):
            lf_client.create_score(
                trace_id=trace_id,
                name=f"llm-judge/{metric}",
                value=float(v),
                data_type="NUMERIC",
                comment=judge.get("reasoning","")
            )

    lf_client.create_score(
        trace_id=trace_id,
        name="heuristic/quality",
        value=_heuristic(answer),
        data_type="NUMERIC"
    )
    lf_client.create_score(
        trace_id=trace_id,
        name="has-error",
        value=0 if re.search(r"i apologize|error generating",answer,re.I) else 1,
        data_type="BOOLEAN"
    )

    lf_client.flush()
    print(f"  [scorer] ✓ relevance={judge.get('relevance','n/a')} heuristic={_heuristic(answer):.2f}")
```
*(`monitoring/langfuse_scorer.py`, lines 51–87)*

> **Note:** `monitoring/eval_runner.py` (§6 below) tries to import a
> `score_recent_traces` function from this module, but that function is **not defined**
> in `langfuse_scorer.py` — only `inline_score`, `_llm_judge`, and `_heuristic` exist. The
> import is wrapped in a `try/except`, so this fails silently in CI rather than crashing
> the pipeline; it's a latent dead code path worth cleaning up.

---

## 4. Prometheus exporter — `monitoring/langfuse_exporter.py`

This is a standalone process (started alongside `main.py` in the same container, see §8)
that polls Langfuse's public REST API for the latest scores and republishes them as
Prometheus gauges on port `8001`.

Basic-auth header construction from the public/secret key pair:

```python
def _auth_headers():
    credentials = f"{PUBLIC_KEY}:{SECRET_KEY}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/json",
    }
```
*(`monitoring/langfuse_exporter.py`, lines 15–21)*

Gauge definitions and the score-name → gauge mapping:

```python
g_relevance    = Gauge("llm_judge_relevance",    "LLM judge relevance score")
g_faithfulness = Gauge("llm_judge_faithfulness", "LLM judge faithfulness score")
g_completeness = Gauge("llm_judge_completeness", "LLM judge completeness score")
g_hallucination= Gauge("llm_judge_hallucination_risk", "LLM judge hallucination risk")
g_heuristic    = Gauge("llm_heuristic_quality",  "Heuristic quality score")
g_error_rate   = Gauge("llm_error_rate",         "Fraction of traces with errors")
g_scores_total = Gauge("llm_scores_fetched",     "Total scores fetched")

SCORE_MAP = {
    "llm-judge/relevance":         g_relevance,
    "llm-judge/faithfulness":      g_faithfulness,
    "llm-judge/completeness":      g_completeness,
    "llm-judge/hallucination_risk":g_hallucination,
    "heuristic/quality":           g_heuristic,
}
```
*(`monitoring/langfuse_exporter.py`, lines 23–37)*

The collection loop fetches up to 100 recent scores, buckets them by name, and sets gauge
values to the running average — polling every 30 seconds:

```python
def collect():
    headers = _auth_headers()
    scores_url = f"{BASE_URL}/api/public/scores"

    params = {"page": 1, "limit": 100}
    resp = requests.get(scores_url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    scores_list = resp.json().get("data", [])
    g_scores_total.set(len(scores_list))

    buckets = {k: [] for k in SCORE_MAP}
    errors = []

    for sc in scores_list:
        name = sc.get("name")
        value = sc.get("value")
        if name in buckets:
            buckets[name].append(value)
        if name == "has-error":
            errors.append(1 - value)

    for name, gauge in SCORE_MAP.items():
        vals = buckets[name]
        if vals:
            avg = sum(vals) / len(vals)
            gauge.set(avg)

    if errors:
        g_error_rate.set(sum(errors) / len(errors))

if __name__ == "__main__":
    start_http_server(8001)
    while True:
        try:
            collect()
        except Exception as e:
            print(f"[exporter] Collection error: {e}")
        time.sleep(30)
```
*(`monitoring/langfuse_exporter.py`, lines 39–81)*

---

## 5. CI/CD quality gate — `monitoring/eval_runner.py`

Run as job **8 · LLM Evaluate** in the pipeline (see §9). Fetches traces from the last
hour via the Langfuse REST API and fails the pipeline if the average
`heuristic/quality` score drops below a configurable threshold:

```python
QUALITY_THRESHOLD = float(os.environ.get("QUALITY_THRESHOLD", "0.5"))

def main():
    base_url = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
    headers = _make_langfuse_headers()

    since = datetime.now(timezone.utc) - timedelta(hours=1)
    since_ms = int(since.timestamp() * 1000)
    traces_url = f"{base_url}/api/public/traces"
    params = {
        "fromTimestamp": since_ms,
        "limit": 20,
        "orderBy": "createdAt",
        "order": "desc",
    }

    resp = requests.get(traces_url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    traces = resp.json().get("data", [])
    ...
    scores = []
    for t in traces:
        trace_id = t["id"]
        scores_url = f"{base_url}/api/public/traces/{trace_id}/scores"
        s_resp = requests.get(scores_url, headers=headers, timeout=10)
        s_resp.raise_for_status()
        s_data = s_resp.json()
        for sc in s_data.get("data", []):
            if sc.get("name") == "heuristic/quality":
                scores.append(sc["value"])

    avg = sum(scores) / len(scores)
    print(f"[eval] avg heuristic/quality = {avg:.2f} (threshold={QUALITY_THRESHOLD})")
    if avg < QUALITY_THRESHOLD:
        print("[eval] QUALITY GATE FAILED")
        sys.exit(1)
    print("[eval] QUALITY GATE PASSED")
    sys.exit(0)
```
*(`monitoring/eval_runner.py`, lines 9, 22–80, condensed)*

---

## 6. PR comment bot — `monitoring/pr_comment.py`

Posts a markdown table of the five most recent traces' `heuristic/quality` scores as a
GitHub PR comment, using the GitHub REST API and a `GITHUB_TOKEN`:

```python
lines = ["## 🤖 LLM Eval Scores", "| Trace | Score |", "|---|---|"]
for t in traces:
    trace_id = t["id"]
    scores_url = f"{base_url}/api/public/traces/{trace_id}/scores"
    ...
    for sc in scores:
        if sc.get("name") == "heuristic/quality":
            lines.append(f"| {trace_id[:8]} | {sc['value']:.2f} |")
            break

body = "\n".join(lines)
repo = os.environ["GITHUB_REPOSITORY"]
pr_number = os.environ["PR_NUMBER"]
token = os.environ["GITHUB_TOKEN"]

url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
resp = requests.post(
    url,
    headers={"Authorization": f"Bearer {token}"},
    json={"body": body},
    timeout=10,
)
```
*(`monitoring/pr_comment.py`, lines 42–69, condensed)*

---

## 7. Container wiring — `docker-compose.yml`

The agent container launches **both** the Prometheus exporter and the CLI agent as
background/foreground processes in the same container, and passes all three Langfuse
env vars through:

```yaml
services:
  agent:
    ...
    ports:
    - "8001:8001"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY}
      - LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY}
      - LANGFUSE_BASE_URL=${LANGFUSE_BASE_URL:-https://cloud.langfuse.com}
      - APP_ENV=${APP_ENV:-production}
      - APP_VERSION=${APP_VERSION:-0.1.0}
      - CHROMA_PATH=/app/data/chroma
    command: >
         sh -c "python /app/monitoring/langfuse_exporter.py & 
         python /app/main.py"
    healthcheck:
      test: ["CMD", "python", "scripts/healthcheck.py"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s
```
*(`docker-compose.yml`, lines 4–34, condensed)*

Prometheus and Grafana run as sibling services, scraping the exporter and visualizing it:

```yaml
  prometheus:
    image: prom/prometheus:v2.51.0
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command: ["--config.file=/etc/prometheus/prometheus.yml",
              "--storage.tsdb.retention.time=7d"]
    ports:
      - "9080:9090"

  grafana:
    image: grafana/grafana:10.4.0
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "3001:3000"
    depends_on:
      - prometheus
```
*(`docker-compose.yml`, lines 42–66, condensed)*

Prometheus's scrape target for the agent:

```yaml
scrape_configs:
  - job_name: "devops-agent"
    static_configs:
      - targets: ["agent:8001"]
    metrics_path: /metrics
```
*(`monitoring/prometheus.yml`, lines 4–8)*

---

## 8. Health check — `scripts/healthcheck.py`

Used both by the Docker `HEALTHCHECK` directive and as a CI smoke test. It instantiates a
Langfuse client as part of the health check, so a broken Langfuse configuration marks the
container unhealthy:

```python
def check():
    try:
        from config.settings import Settings
        Settings.validate()
        from langfuse import get_client
        _ = get_client()
        print("[healthcheck] OK — all settings valid")
        sys.exit(0)
    except ValueError as e:
        print(f"[healthcheck] FAIL — missing config: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[healthcheck] FAIL — unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
```
*(`scripts/healthcheck.py`, lines 9–20)*

---

## 9. CI/CD pipeline job — "8 · LLM Evaluate" — `.github/workflows/cicd.yml`

The pipeline's evaluation stage runs after deploy, executes the quality gate, and posts
the PR comment on `pull_request` events:

```yaml
  evaluate:
    name: "8 · LLM Evaluate"
    needs: deploy-docker
    runs-on: self-hosted
    container:
      image: python:3.11-slim
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Wait for agent to warm up
        run: sleep 30
      - name: Run LLM Evaluator + Quality Gate
        run: |
          export PYTHONPATH=$PYTHONPATH:$(pwd)
          python -m monitoring.eval_runner
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          LANGFUSE_SECRET_KEY: ${{ secrets.LANGFUSE_SECRET_KEY }}
          LANGFUSE_PUBLIC_KEY: ${{ secrets.LANGFUSE_PUBLIC_KEY }}
          LANGFUSE_BASE_URL: ${{ env.LANGFUSE_BASE_URL }}
      - name: Post score summary to PR
        if: github.event_name == 'pull_request'
        run: python monitoring/pr_comment.py
        env:
          LANGFUSE_SECRET_KEY: ${{ secrets.LANGFUSE_SECRET_KEY }}
          LANGFUSE_PUBLIC_KEY: ${{ secrets.LANGFUSE_PUBLIC_KEY }}
          LANGFUSE_BASE_URL: ${{ env.LANGFUSE_BASE_URL }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITHUB_REPOSITORY: ${{ github.repository }}
          PR_NUMBER: ${{ github.event.pull_request.number }}
```
*(`.github/workflows/cicd.yml`, lines 174–206)*

The pipeline env block that provides the Langfuse base URL to every job (self-hosted
Langfuse assumed at `localhost:3000` inside the runner):

```yaml
env:
  ...
  LANGFUSE_BASE_URL: "http://localhost:3000"
```
*(`.github/workflows/cicd.yml`, line 31)*

---

## 11. Dependency pin — `requirements.txt`

```text
# Observability
langfuse>=2.58.0
```
*(`requirements.txt`, observability section)*

> `monitoring/langfuse_exporter.py` also imports `prometheus_client` and `requests`, and
> `monitoring/eval_runner.py` / `pr_comment.py` import `requests` — none of these three
> appear as explicit top-level pins in `requirements.txt`. They currently work only if
> pulled in transitively (e.g. via `langfuse`, `chromadb`, or `tavily-python`), which is
> worth hardening with explicit pins.

---

## 10. Summary table

| Concern | File | Mechanism |
|---|---|---|
| Full-run trace | `main.py` | `@observe(name="agent-run")` + shared `CallbackHandler` |
| Per-node spans | `agent/graph.py` | `@observe` on `reasoning_node`, `tool_execution_node`, `answer_node` |
| Graceful shutdown | `main.py`, `agent/graph.py` | `atexit.register(...flush)` + `SIGTERM` handler |
| Inline LLM-as-judge scoring | `monitoring/langfuse_scorer.py` | Groq (OpenAI-compatible) + `lf_client.create_score()` |
| Heuristic scoring | `monitoring/langfuse_scorer.py` | `_heuristic()` regex/length rules |
| Infra metrics export | `monitoring/langfuse_exporter.py` | Polls `/api/public/scores`, exposes Prometheus `Gauge`s on `:8001` |
| Metrics scraping | `monitoring/prometheus.yml` + `docker-compose.yml` | Prometheus `devops-agent` job → Grafana |
| CI quality gate | `monitoring/eval_runner.py` | `/api/public/traces` + `/scores`, exits 1 below `QUALITY_THRESHOLD` |
| PR reporting | `monitoring/pr_comment.py` | Posts score table via GitHub REST API |
| Container health | `scripts/healthcheck.py` | Instantiates `langfuse.get_client()` as part of health check |
| CI wiring | `.github/workflows/cicd.yml` (job 8) | Runs evaluator + PR comment post-deploy |
