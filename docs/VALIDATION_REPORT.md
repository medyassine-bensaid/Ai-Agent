# Validation report: DevOps ReAct agent

Date: 2026-10-11 · Branch `validation-and-ci` · Repository
https://github.com/medyassine-bensaid/Ai-Agent (public)

## A. Executive summary

Last updated 2026-10-12 (see section K for the CI/CD runs and the changes made on that day).

**Working and verified by execution:**
- The agent answers end to end through the real path (Groq `openai/gpt-oss-120b`, Chroma RAG,
  Tavily, LangGraph). It routes knowledge-base questions to RAG, current-information questions
  to Tavily, and both when asked; it acknowledges missing information and no longer presents
  links it did not retrieve.
- Every run is traced in Langfuse v4: node, generation, tool, retriever and evaluator spans,
  plus LLM-judge, heuristic and evaluation scores that are read back through the API.
- Unit/regression suite: **119 passed, 0 failed**; `ruff check` and `ruff format --check` clean.
- GitHub Actions on the public repository runs **21 jobs**. On the latest run (`38e401d`),
  every quality, security, unit-test and container job passed.

**Fixed:** 25 defects (sections D and K), several of which made the monitoring silently
useless. The LLM judge never produced a score, every Langfuse read path returned 404 on v4,
the CI test step and quality gate could not fail, and the agent fabricated sources after a
failed search. Vulnerable LangChain/LangGraph pins were upgraded (pip-audit: 27 → 13 findings).

**Not verified / open (section I):**
- No full 34-case live run has passed yet. Groq's free tier (200 000 tokens/day for
  `gpt-oss-120b`) is the limiting factor.
- Best results so far:
  - In CI, with every evaluable case judged: 3 of 6 critical cases passed. The 3 failures led to fixes 17–19.
  - After those fixes, BOTH-01 and ROB-02 were verified end to end with `gpt-oss-20b` (section K).
  - The latest CI run reported all 6 critical cases **BLOCKED** by the quota, not failed.
- Langfuse UI: inspected on 2026-10-12 through read-only screenshots of the user's signed-in
  Edge session (K.9); this inspection led to fixes 24 and 25.

## B. Context recovered

Neither shared conversation could be read:
- WebFetch returned only the page footer.
- The built-in browser redirected to the claude.ai login page.
- Claude in Chrome was not connected.
- No local Claude Code session transcripts mention this project.

Context therefore came from the repository itself:
- The single commit `d59d8de`.
- The previous session's uncommitted diff: model switch to `openai/gpt-oss-120b`, Langfuse SDK v4 migration (`get_client`, `propagate_attributes`), Docker CPU-torch and volume fixes, `.env.example`, `.dockerignore`.
- The running services. Every requirement and decision from those conversations that is not visible in the code is unknown to this report.

## C. Environment and architecture

| Component | Actual implementation |
|---|---|
| Orchestration | LangGraph 0.2.45 `StateGraph`: reasoning → tool_execution → reasoning … → answer |
| LLM | Groq `openai/gpt-oss-120b` via `langchain-groq` 0.2.1 (temperature 0.1) |
| RAG | ChromaDB 1.5.9 persistent store `data/chroma`, collection `devops_docs` (cosine), `all-MiniLM-L6-v2` embeddings, 400-char chunks / 80 overlap, top-4, min similarity 0.3; seeded from 5 built-in topics |
| Web search | Tavily via `langchain_community` `TavilySearchResults` (5 results, 30 s timeout) |
| Judge | Groq `openai/gpt-oss-120b` through the Langfuse OpenAI drop-in |
| Observability | Langfuse server 4.50.0 (self-hosted, Docker, events_only mode), SDK 4.16.0; Prometheus exporter + Grafana |
| Entry point | CLI `python main.py` (no HTTP API or web UI in this repository) |
| Deploy | Dockerfile, docker-compose (agent + Prometheus + Grafana), Kustomize/Minikube |

## D. Changes made (root cause → fix → regression test)

| # | Defect (verified) | Fix | Test |
|---|---|---|---|
| 1 | Logger `print` of emoji raised `UnicodeEncodeError` on cp1252 consoles/pipes, crashing the run and then the error handler | `safe_print` fallback; CLI/eval reconfigure stdout `errors=replace` | `test_safe_print_survives_non_utf8_stdout` |
| 2 | Judge `max_tokens=300` exhausted by gpt-oss reasoning tokens, so content was empty and `{}` was silently skipped: **no `llm-judge/*` score was ever written** | 2048 tokens, `reasoning_effort=low`, JSON mode, validated [0,1] output, retry with backoff, `llm-judge/error` on failure | `test_parse_judge_*`, `test_judge_*`, `test_inline_score_*` |
| 3 | Judge received no evidence; after adding it, it was truncated at 6 000 chars, hiding the fact under test (k8s 1.37.1 at char 6 441) | Evidence passed, up to 20 000 chars | `test_judge_budget_fits_reasoning_model_and_sends_evidence` |
| 4 | `has-error` written as 1 for normal answers | 1 = error answer | `test_inline_score_records_judge_failure_and_correct_has_error` |
| 5 | Langfuse v4 events_only returns 404 for `/api/public/traces`, `/traces/{id}`, `/scores`, breaking `eval_runner`, `pr_comment`, the exporter and `rag_evaluator` (`fetch_traces` also removed in SDK v4) | `monitoring/langfuse_api.py` on `/v2/observations` and `/v3/scores` | `tests/test_monitoring.py` |
| 6 | `eval_runner` exited 0 on fetch failure or when no score existed, so the gate could never fail; it also gated only on a length heuristic | Exit 2 for "could not evaluate", gate on judge means and error rate | `test_gate_*` (6 tests) |
| 7 | RAG never selected (prompt never described the KB); at most one tool per turn, so "KB + web" impossible | KB topics and observations in `REACT_PROMPT`; tool_execution → reasoning loop with `MAX_TOOL_CALLS=3` and duplicate guard | `test_reasoning_*`, `test_graph_loops_*` |
| 8 | Answer claimed "searches across Google, Bing, DuckDuckGo, GitHub…" when one Tavily query ran; cited site names, not URLs | Grounding rules: only listed sources, full URLs, say what was not found, prefer official source on conflicts | `test_prompts_describe_kb_and_grounding_rules`; eval grounding check |
| 9 | Off-topic queries got 4 chunks with similarity ≈ 0; edited documents were never re-embedded | `RAG_MIN_SCORE` threshold; change-aware `upsert` | `test_rag_*` (3 tests, real Chroma) |
| 10 | `langchain_community` posts to Tavily **without a timeout**: an eval run hung 18 min | Wrapper with `WEB_SEARCH_TIMEOUT=30` | `test_tavily_request_has_timeout` |
| 11 | Tavily errors returned as a string were iterated, giving `'str' object has no attribute 'get'` | Raise the real error (e.g. `HTTPError 400`) | `test_web_search_tool_reports_timeout_as_failure` |
| 12 | Groq 400 `tool_use_failed` (gpt-oss emits a native tool call) dropped the tool step | Recover the call from `failed_generation` | `test_reasoning_recovers_native_tool_call` |
| 13 | Provider errors (Groq org id, billing URL) shown to the user | Generic message; details to logs/trace (`level=ERROR`) | `test_answer_node_error_does_not_expose_provider_details` |
| 14 | Judge calls invisible in Langfuse (no cost/latency record) | `@observe(as_type="evaluator")` + Langfuse OpenAI drop-in | verified live (section G) |
| 15 | Settings test re-loaded the developer's `.env` on module reload, so it could not fail | Stub `load_dotenv` during reload | `test_settings_validation_raises_on_missing_keys` |
| 16 | CI: `pytest … \|\| true`; `if: always()` ran build/push/**deploy after failed tests**; lint auto-fixed in CI; self-hosted runner for PRs of a public repo; exporter launched by path (import failure) | New `cicd.yml` (GitHub-hosted) and manual `deploy.yml`; `python -m monitoring.langfuse_exporter` | `test_verify_traces_accepts_runner_record_shape` (runner) |

Other changes: `run_agent_detailed()` (answer + routing + evidence + trace id, used by evals;
`run_agent()` unchanged); `evals/` suite; `requirements.lock`; README and
`MONITORING_LANGFUSE_README.md` rewritten for the current behaviour; `.gitignore` excludes
`data/` and `reports/`; 186 → 0 ruff findings (whitespace/import fixes plus `_heuristic` reformat).

## E. Test results

| Run | Command | Result |
|---|---|---|
| Baseline | `pytest` | 34 tests: 33 passed, 1 failed (#15) |
| Baseline | `ruff check .` | 186 findings |
| Final | `pytest` | **76 passed, 0 failed, 0 skipped** (34 original + 42 new; network mocked) |
| Final | `ruff check .` | All checks passed |

Live evaluation (`python -m evals.run_eval`, real Groq/Tavily/Chroma/Langfuse):

| Run id | Scope | Outcome |
|---|---|---|
| `20261011185939-b8c990` | 18 | Stopped by me after the Tavily hang (defect 10). Before that: RAG-02 missed KB routing, and WEB-01 cited no URL (both fixed in prompts) |
| `20261011193037-b157ee` | 18 | Crashed at case 8 on a console encoding error in the runner (fixed). GEN-01 and RAG-01..05 passed; WEB-01 failed on a stale version and an ungrounded URL (defects 3 and 8) |
| `20261011194933-a477f4` | 18 | **10 cases evaluable: 8 passed, 2 failed; 8 blocked** by Groq TPD (200 000 tokens/day). The telemetry check reported failures for all 18 because of a runner bug (`KeyError: 'tool_calls'`, fixed, test added) |
| `20261011211705-8a11e4` | GEN-01 | PASS, telemetry verified |
| (next run) | GEN-01 | PASS, telemetry verified, judge evaluator span present (section G) |

Run 3 by case (deterministic checks plus judge relevance/faithfulness/hallucination_risk):

| Case | Routing | Judge | Result |
|---|---|---|---|
| GEN-01 general | none | 1.0 / 1.0 / 1.0 | pass |
| RAG-01 CI/CD (critical) | knowledge_base | 1.0 / 1.0 / 0.95 | pass |
| RAG-02 paraphrase | knowledge_base | 1.0 / 0.9 / 0.95 | pass |
| RAG-03 multi-chunk | knowledge_base ×2 | 1.0 / 1.0 / 0.98 | pass |
| RAG-04 identifiers | knowledge_base | 1.0 / 1.0 / 1.0 | pass |
| RAG-05 | knowledge_base | 1.0 / 1.0 / 1.0 | pass |
| WEB-01 latest k8s (critical) | web_search | 0.96 / 0.94 / 0.97 | pass |
| WEB-02 latest Terraform | web_search | 0.9 / 0.9 / 0.95 | **fail**: cited `discuss.hashicorp.com/...v1-15-0-released/`, not in results |
| BOTH-01 KB + web (critical) | web_search only | 1.0 / 1.0 / 0.98 | **fail**: knowledge base not consulted |
| UNK-01 unknowable (critical) | web_search | (passed checks) | pass |
| UNK-02 … MEM-01 (8 cases) | n/a | judge errored | **blocked**: Groq daily token limit |

Thresholds: per case relevance, faithfulness and hallucination_risk ≥ 0.7 (judge-exempt for
refusal/ambiguity cases); suite gate pass rate ≥ 0.8 with all 6 critical cases passing.

Test types: unit/regression tests use mocks for Groq, Tavily and Langfuse (the RAG tests use
a real Chroma with a deterministic test embedder). The evaluation suite uses no mocks; ROB-02
makes a real Tavily call with an invalid key.

## F. RAG and Tavily evidence

- Direct retrieval checks:
  - "What is Docker?" → `kb:docker` (0.793)
  - "terraform state" → `kb:infrastructure as code` (0.47)
  - "how do pods get scheduled" → `kb:kubernetes` (0.513)
  - "chocolate cake" → all chunks ≤ 0.022, so nothing is returned after the threshold fix
  - Re-ingesting the same text adds 0 chunks.
- Tavily:
  - A real search returned results in 5.0 s.
  - An invalid key surfaces `HTTPError 400` and the agent reports the failure.
  - WEB-01 cited `https://komodor.com/...` and `https://kubernetes.io/releases` from the actual results.
- Conflicting evidence (observed): Tavily returned komodor ("April 2026: 1.36") and
  kubernetes.io ("1.37.1"). The agent preferred the stale source. The prompt now asks it to
  prefer official/most recent sources and mention the conflict; this has not been re-run since.

## G. Langfuse verification (API)

Verified through `/api/public/v2/observations` and `/api/public/v3/scores` on
http://localhost:3000 (project "DevOps Q&A Agent"):

- **Trace structure:** `agent-run` root (input = question, output = answer), with `LangGraph`, node spans, `ChatGroq` GENERATIONs (model `openai/gpt-oss-120b`, token usage), tool spans `web_search_tool` → `tavily_search_results_json` and `rag_tool` → `rag-retrieval`, and an `llm-judge` EVALUATOR with its own generation. In trace `b3139f2b3c7058d2241f95784d11ddf8` the judge generation used 1 386 tokens, 55 of them reasoning.
- **Errors:** a failed run is recorded at `level=ERROR` (trace `d13b2c6123d01772b2e50dec1b2c24ec`).
- **Scores:** each evaluated trace carries the 4 `llm-judge/*`, `heuristic/quality`, `has-error`, `eval/passed` and `eval/routing-correct` scores. The runner re-reads them and checks the values (traces `3a70a091818b51a0ec726a84b774a25e`, `b3139f2b…`).
- **Eval filtering:** eval traces are tagged `eval` + category, with session `eval-<run>-<case>`.
- **Not available:** cost (no Groq model price is defined in Langfuse).
- **UI and dashboards:** inspected on 2026-10-12; see K.9.

## H. GitHub and CI/CD

- Repository: https://github.com/medyassine-bensaid/Ai-Agent (public). `main` and
  `validation-and-ci` were pushed from commit `dc04048`. The original private remote `origin`
  (Looyas/CL-09-2526-MLOps-for-Active-Agents) is untouched.
- Secrets audit before publishing:
  - History is one commit plus these two.
  - `.env` was never committed.
  - No key patterns appear in history or in the staged diff; the only hits are dummy test strings.
  - `data/`, `reports/` and `.env` are ignored.
- Workflows: `LLMOps CI/CD` (lint → tests → Docker build/smoke; live eval with the critical
  cases on push, all cases on manual `full_eval`) and `Deploy (self-hosted)` (manual).
- Run status: see section K.
- Required repository secrets: `GROQ_API_KEY` and `TAVILY_API_KEY` for the live evaluation; Docker Hub and Langfuse secrets for deploy.

## I. Remaining issues and risks

| Issue | Severity | Remediation |
|---|---|---|
| No complete passing full-suite live run; 8 cases unverified | High | Rerun `python -m evals.run_eval` after the Groq daily window frees up (≈24 h rolling) or on a paid tier |
| BOTH-01: model sometimes skips the KB when a question also needs the web | Medium | Re-measure; if persistent, enforce KB-first when the question names the knowledge base |
| WEB-02: model altered/invented a URL path | Medium | Detected by the grounding check; re-measure after the prompt changes |
| Groq free tier 200k tokens/day: one full run uses a large share | High (CI) | CI runs critical cases on push; upgrade tier or use a different `JUDGE_MODEL` |
| `requirements.lock` frozen from the validated Windows venv (Docker→PyPI stalled; disk full) | Medium | Linux install is verified by the CI test job; regenerate on Linux when possible |
| C: drive had 0.13 GB free (pip cache purged → 2.18 GB) | **High** | Free disk space: Docker, Langfuse ClickHouse/Postgres and Chroma live on C: |
| Local Langfuse uses the upstream `CHANGEME` defaults; ports 3000/9090 bound on 0.0.0.0 | High (local security) | Set real secrets; bind to 127.0.0.1 |
| Memory is a process-global singleton (no concurrency support) | Low | By design for the CLI; documented |
| k8s memory limit 512Mi likely too small for torch + MiniLM | Medium (untested) | Raise to ~1.5Gi and test on Minikube |
| Historical scores: `has-error` inverted, no judge scores before the fix | Low | Filter dashboards to dates after 2026-10-11 19:00 UTC |
| Deploy workflow and Minikube not executed | Medium | Needs the self-hosted runner, `RUNNER_HOME` variable and Docker Hub secrets |

## J. Reproduction

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock
cp .env.example .env                      # GROQ / TAVILY / LANGFUSE keys
python main.py                            # CLI
python -m pytest                          # 119 unit/regression tests
python -m evals.run_eval --critical-only  # live, 6 cases; drop the flag for all 34
python -m monitoring.eval_runner          # gate over the last hour of scores
```

Langfuse: http://localhost:3000. Filter traces by tag `eval` or session `eval-<run id>`.
Reports: `reports/eval_report.md` / `.json`. CI: push to `main`, or run "LLMOps CI/CD" from the
Actions tab with `full_eval` for the whole suite.

## K. Update 2026-10-12: CI/CD execution, fixes and upgrades

### K.1 Getting GitHub Actions to run
1. Actions did not start on the new repository until the owner enabled it. A minimal
   diagnostic workflow (since removed) confirmed the cause was a setting, not the workflow files.
2. Run `38179985160` (`7630889`) failed at dependency installation. The lock had been frozen
   from the Windows venv, which contained a stray `langgraph-prebuilt 1.1.0` that requires
   `langchain-core>=1.3.1` (`pip check` failed locally too). The lock is now compiled for
   Linux with `uv pip compile`.
3. Run `38181003183` (`1d2e77e`): lint, unit tests and the Docker build passed. The live eval
   ran but every case except RAG-01 hit Groq `RateLimitError` (daily quota). These were counted
   as failures, which led to the **blocked** status (K.3).

### K.2 Granular pipeline (`.github/workflows/cicd.yml`, 21 jobs)
| Stage | Jobs |
|---|---|
| Quality & security | Lint · Format check · Secret scan (gitleaks, full history) · Dependency audit (pip-audit, advisory) |
| Unit tests | Agent core · Regression fixes · Monitoring & quality gate · Evaluation suite logic · CI scripts → Report & coverage |
| Container | Build image → Healthcheck & import smoke (non-root) · Image scan (Trivy, advisory) |
| Live evaluation | Preflight → RAG · Web search · Routing KB+web & unknowns · Safety & off-domain · Robustness, memory & general (each with its own ephemeral Langfuse) → Combined report & gate |
| Summary | Pipeline summary |

Failing and blocked evaluation cases are written as GitHub annotations, so they are readable
on the public run page without signing in.

### K.3 Evaluation changes
- **Status passed / failed / blocked.** Blocked = the LLM provider refused or was unavailable
  (rate limit, quota, 5xx) for the agent or the judge, so the case was not evaluated. The gate
  exits 2 when anything is blocked and nothing failed; a blocked run is never green.
- Dataset 18 → **34 cases** (critical set unchanged: RAG-01, WEB-01, BOTH-01, UNK-01, SEC-01, ROB-02).
- `--prefix`, `--output-dir`, `EVAL_CASE_DELAY`; `evals/merge_reports.py` merges the group reports.
- Answers are asked to stay concise (usually under 300 words) to cut tokens per case.

### K.4 Run `38214006584` (`eca6046`): first CI run with every evaluable case judged
- Lint, format, secret scan, 5 unit suites, test report, image build, smoke test and Trivy all passed.
- Critical cases: **RAG-01, WEB-01 and UNK-01 passed**; telemetry was verified for every trace. Failed:

| Case | Finding | Root cause | Fix (regression test) |
|---|---|---|---|
| ROB-02 | Cited `helm.sh` and `github.com/helm/helm/releases` after the search **failed** | Prompt-only grounding rule not followed | 17: a failed-tool observation now states there is no evidence, and `agent/grounding.py` replaces any URL that is not in the evidence or the question (`test_guardrail_*`, `test_answer_node_strips_fabricated_urls_after_failed_search`) |
| BOTH-01 | Web search only; knowledge base skipped | Model stops after one source | 18: tools the user explicitly asks for ("knowledge base", "from the web") are consulted before answering (`test_explicit_tool_requests`, `test_explicitly_requested_kb_is_consulted_before_answering`) |
| SEC-01 | "does not refuse" | **Checker false negative**: the answer "I’m sorry, but I can’t provide those values" uses U+2019 | 19: evaluation text is normalised (`test_refusal_with_typographic_apostrophes_is_recognised`) |
| (audit job) | Advisory job exited 1 | Shell parsing bug in my job script | 20: `scripts/audit_summary.py` on pip-audit JSON (`test_audit_summary_*`) |

Also fixed: 21, the runner's telemetry read-back used a field the record did not have (all
telemetry checks failed in run 3 on 2026-10-11; `test_verify_traces_accepts_runner_record_shape`);
22, routing accuracy shows n/a instead of 0% when nothing was evaluable.

### K.5 Dependency security upgrade (fix 23)
pip-audit on the lock reported **27 known vulnerabilities in 11 packages**. Upgraded within
the LangChain 0.3 line (a 1.x migration would change the framework API):

| Package | Before | After |
|---|---|---|
| langchain / langchain-core / langchain-community | 0.3.7 / 0.3.63 / 0.3.5 | 0.3.30 / 0.3.86 / 0.3.31 |
| langchain-groq / langchain-text-splitters | 0.2.1 / 0.3.8 | 0.3.8 / 0.3.11 |
| langgraph / checkpoint / sdk / prebuilt | 0.2.45 / 2.1.2 / 0.1.74 / (stray 1.1.0) | 0.6.11 / 3.0.1 / 0.2.15 / 0.6.5 |
| langsmith, python-dotenv, pytest, pytest-cov | 0.1.147, 1.0.0, 8.3.3, 6.0.0 | 0.14.7, 1.2.4, 9.1.1, 7.1.0 |

Result: **13 findings in 7 packages**. The remaining ones need LangChain/LangGraph 1.x, or have no
fix published (chromadb 1.5.9). Validation of the upgraded stack:
- 116 unit tests pass and `pip check` is clean.
- Real Tavily search: 5 results in 2.8 s. RAG retrieval and the off-topic refusal work.
- End-to-end agent runs on `openai/gpt-oss-20b` (Groq quotas are per model; the 120b quota was
  exhausted):
  - BOTH-01 called web_search then knowledge_base, cited only a URL from the evidence, and the
    judge scored 1.0 on all metrics (trace `16833abac537f8e1577cbe327144449b`).
  - ROB-02 (invalid Tavily key) said the version could not be retrieved, and two fabricated
    links were removed by the guardrail (trace `de8b569b7450c5e57587944df4b48aa8`).

### K.6 Latest CI run `38216735736` (`38e401d`)
- **Green:** lint, format check, secret scan, dependency audit (advisory), all 5 unit-test
  suites, report & coverage, image build, healthcheck & smoke test, image scan, preflight,
  pipeline summary.
- **Live evaluation:** all 6 critical cases **BLOCKED (provider limit)**, 0 failed; gate
  BLOCKED (exit 2). The Groq `gpt-oss-120b` daily quota was exhausted by the earlier runs.
- Note: the commit message of `38e401d` says "118 unit tests"; the correct count is 116.

### K.7 Local Langfuse content (last 48 h, via API)
54 agent traces; 995 observations (152 generations, 50 tool spans, 20 retriever spans,
10 evaluator spans); about 203k tokens recorded; 346 scores (`llm-judge/*`, `heuristic/quality`,
`has-error`, `eval/passed`, `eval/routing-correct`, `eval/blocked`, `llm-judge/error`).

### K.8 Still open
- A full 34-case live run once the Groq quota allows it (≈5–7k tokens per case). Use a
  manual "Run workflow" with `full_eval`, or `python -m evals.run_eval` locally to fill the
  local Langfuse dashboards.
- LangChain 1.x migration for the remaining advisories; chromadb has no fix yet.
- Low disk space on C: (1.3–2.2 GB free during this work).

### K.9 Langfuse UI verification (2026-10-12)
The Langfuse UI was inspected through read-only screenshots of the user's signed-in Edge
window (project "DevOps Q&A Agent", Langfuse v4.50.0); the app's own Browser pane was never
signed in.

| View | Observed |
|---|---|
| Dashboards → Langfuse Agent Dashboard | P95 Tool Latency by Tool: rag_tool 9.38 s, web_search_tool 5.95 s, tavily_search_results_json 5.95 s. P95 Latency by Observation Type with CHAIN, SPAN, GENERATION, TOOL, RETRIEVER and EVALUATOR series. "Total Tool Calls" shows *No data*, as expected: it counts native tool calls inside LLM generations, and this agent decides routing with JSON and runs tools from the graph. |
| Tool Errors by Tool | Empty at first although searches had failed → **fix 24**. After the fix it shows `web_search_tool: 1`. |
| Trace `1648f8874f547053f8f223a1c74e84c4` | 6.15 s, 3,741 tokens; tree agent-run → LangGraph → reasoning (ChatGroq 822 tok) → tool_execution → web_search_tool **ERROR** → tavily_search_results_json → reasoning → answer (ChatGroq 1,194 tok) → llm-judge (EVALUATOR) → OpenAI-generation (814 tok, gpt-oss-20b). Score chips has-error=False, heuristic/quality=1.00 (+4). The judge input contains the question, the error evidence with the no-evidence note, and the answer with `[unverified link removed]`. Only **one** tool_execution, although two searches were reported → **fix 25**. |
| Tracing filters | Observation counts by type: CHAIN 537, SPAN 238, GENERATION 156, TOOL 52, RETRIEVER 20, EVALUATOR 11. |

**Fix 24: failed searches invisible in error analytics.** `web_search_tool` returned the
error text, so the LangChain callback recorded a successful tool run. It now raises;
`tool_execution_node` turns the exception into a no-evidence observation. Verified live:
`web_search_tool level=ERROR`, status "Web search failed: HTTPError('401 …')"
(`test_failed_web_search_is_reported_to_tracing_as_tool_error`; the original
`test_web_search_handles_tool_exception` now asserts the new contract end to end).

**Fix 25: refined search silently skipped (pre-existing router bug).** `should_continue`
treated a tool call as executed when its query text appeared in any earlier observation.
The refined query "Helm latest release" is a substring of "Query: Helm latest release
version", so the second search never ran but was still reported in `tool_calls` (which also
skewed the eval routing check). The router now compares the number of observations with the
number of calls, and `run_agent_detailed` reports only executed calls. Verified live: trace
`9d1c51aaafc101a3e56f2961a19176ab` has two web_search_tool executions
(`test_router_executes_refined_query_that_is_substring_of_previous`,
`test_run_agent_detailed_reports_only_executed_tool_calls`).

