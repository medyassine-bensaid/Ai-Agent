# Validation report: DevOps ReAct agent

Date: 2026-10-11 · Branch `validation-and-ci` · Repository
https://github.com/medyassine-bensaid/Ai-Agent (public)

## A. Executive summary

**Working and verified by execution:** the agent answers end to end through the real path
(Groq `openai/gpt-oss-120b`, Chroma RAG, Tavily, LangGraph), routes knowledge-base questions
to RAG and current-information questions to Tavily, acknowledges missing information, and
every run is traced in Langfuse v4 with node, generation, tool and evaluator spans, plus
LLM-judge, heuristic and evaluation scores that can be read back through the API.
Unit/regression suite: **76 passed, 0 failed**; ruff clean.

**Fixed:** 16 defects (section D), several of which made the monitoring silently useless.
The LLM judge never produced a score, every Langfuse read path returned 404 on v4, and both
the CI test step and the CI quality gate could not fail.

**Not fully verified (section I):**
- No complete 18-case live run has passed. The last full run was cut off by Groq's free-tier
  daily token limit after 10 cases; of those, 8 passed and 2 failed (WEB-02 grounding, BOTH-01 routing).
- The Langfuse web UI was not inspected: the app's Browser pane was never signed in.
- No GitHub Actions run was triggered on the new public repository (section H).

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
- **UI and dashboards:** **not inspected**. The Browser pane of the desktop app showed the sign-in page every time; signing in must happen inside that pane (or connect Claude in Chrome).

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
- **Run status: no workflow run was triggered.** Both workflows are registered and `active`
  (`GET /repos/.../actions/workflows`), but `GET /repos/.../actions/runs` stayed at
  `total_count: 0` for 15 minutes after the push of `main` and again after a second push.
  The usual cause is GitHub Actions being disabled or awaiting confirmation for the
  repository/account (Settings → Actions → General, or the Actions tab). **CI/CD is
  therefore not verified remotely**; only the local equivalents ran (ruff, pytest, live eval).
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
| Langfuse UI/dashboards not inspected | Medium | Sign in inside the app's Browser pane |
| Memory is a process-global singleton (no concurrency support) | Low | By design for the CLI; documented |
| k8s memory limit 512Mi likely too small for torch + MiniLM | Medium (untested) | Raise to ~1.5Gi and test on Minikube |
| Historical scores: `has-error` inverted, no judge scores before the fix | Low | Filter dashboards to dates after 2026-10-11 19:00 UTC |
| Deploy workflow and Minikube not executed | Medium | Needs the self-hosted runner, `RUNNER_HOME` variable and Docker Hub secrets |

## J. Reproduction

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock
cp .env.example .env                      # GROQ / TAVILY / LANGFUSE keys
python main.py                            # CLI
python -m pytest                          # 76 unit/regression tests
python -m evals.run_eval --critical-only  # live, 6 cases; drop the flag for all 18
python -m monitoring.eval_runner          # gate over the last hour of scores
```

Langfuse: http://localhost:3000. Filter traces by tag `eval` or session `eval-<run id>`.
Reports: `reports/eval_report.md` / `.json`. CI: push to `main`, or run "LLMOps CI/CD" from the
Actions tab with `full_eval` for the whole suite.
