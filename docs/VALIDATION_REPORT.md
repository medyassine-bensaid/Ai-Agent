# Validation report (INTERIM, work in progress, 2026-10-11)

Status: **incomplete**. Work stopped partway because the session's usage limit was reached. Everything below
was observed directly; items under "Not done" are not verified.

## Context recovery
- Both claude.ai share links were inaccessible (WebFetch returned only the page footer; the
  browser redirected to a login page; Claude in Chrome was not connected; no local session
  transcripts). The repository, its uncommitted diff (model switch to `openai/gpt-oss-120b`,
  Langfuse SDK v4 migration, Docker fixes) and the running services were used as the source of truth.

## Baseline (before changes)
- `pytest`: 34 tests, 33 passed, 1 failed (`test_settings_validation_raises_on_missing_keys`).
- `ruff check .`: 186 findings.
- Langfuse server 4.50.0 healthy at http://localhost:3000; SDK 4.16.0.

## Defects found and fixed (each has a regression test in `tests/test_regressions.py` / `tests/test_monitoring.py`)
1. Logger crashed the agent on non-UTF-8 consoles (`UnicodeEncodeError` on emoji); scorer/eval prints too.
2. LLM judge never produced scores: `max_tokens=300` exhausted by gpt-oss reasoning tokens, so content was empty and `{}` was silently skipped. Fixed with budget 2048, `reasoning_effort=low`, JSON mode, validation, retry with backoff, and an explicit `llm-judge/error` score.
3. Judge saw no evidence, and when evidence was added it was truncated at 6k chars (hid the facts it was judging). Evidence is now passed, up to 20k chars.
4. `has-error` score was inverted (1 = no error).
5. Langfuse v4 events_only mode returns 404 for `/api/public/traces` and `/api/public/scores`, so `eval_runner`, `pr_comment`, the Prometheus exporter and `rag_evaluator` were broken. All now use the v2 observations / v3 scores APIs (`monitoring/langfuse_api.py`).
6. `eval_runner` exited 0 when Langfuse was unreachable or nothing was scored (gate always green). Now exit 2.
7. RAG was never routed to; no ReAct loop (max one tool, so KB plus web was impossible). Fixed with a routing prompt that lists KB topics, observation→reasoning loop, `MAX_TOOL_CALLS`, and duplicate-call guard.
8. Answers claimed searches that never ran and cited site names only. Fixed with grounding rules in `FINAL_ANSWER_PROMPT` (full URLs, no invented sources, conflict rule).
9. RAG had no relevance threshold (off-topic queries got irrelevant chunks), and changed documents were never re-embedded. Fixed with `RAG_MIN_SCORE` and change-aware upsert.
10. Tavily calls had no timeout (a live eval run hung for over 18 minutes). Fixed with a 30 s timeout. Tavily errors were also masked as `'str' object has no attribute 'get'`; the real error is now surfaced.
11. Groq 400 `tool_use_failed` (gpt-oss emitting native tool calls) dropped the tool step; the call is now recovered from `failed_generation`.
12. Settings test isolation: a local `.env` was re-loaded on module reload.
13. CI: `pytest || true`, `if: always()` let build/deploy run after failed tests, lint auto-fixed in CI, self-hosted runner for PRs. Replaced by `cicd.yml` (GitHub-hosted, ephemeral Langfuse) and a manual `deploy.yml`.
14. docker-compose ran the exporter by path (import failure); it now runs as `python -m monitoring.langfuse_exporter`.

## Verified results
- Unit/regression tests: **74 passed, 0 failed** (local, mocked network). `ruff check .`: clean.
- Langfuse API (live): traces contain the `agent-run` root, node spans, `ChatGroq` generations with model `openai/gpt-oss-120b` and token usage, Tavily/RAG tool spans, and ERROR level on failures. Scores are readable via `/api/public/v3/scores`.
- Live judge: a grounded answer scored about 0.9; a fabricated version and URL scored faithfulness 0.0.
- Live eval run 2 (`20261011193037-b157ee`, before fixes 3, 10 and 11): GEN-01 and RAG-01..05 passed; WEB-01 failed (judge, correctly: a stale version plus an ungrounded docs URL); the run then crashed on a console encoding error (fixed).
- Live eval run 3 (`20261011194933-a477f4`) was still running when work stopped (first 4 cases passed).

## Not done / open
- Full eval suite pass/fail totals (run 3 unfinished; see `reports/eval_report.md` if it completes).
- `requirements.lock` **not yet generated**: Docker→PyPI network stalled. README, `cicd.yml` and the planned Dockerfile change reference it, so CI will fail until it is created (a host-side pip resolution was started).
- Langfuse UI / dashboard inspection: not performed (the user had not signed in yet).
- Nothing committed or pushed; no new public repo URL received; no GitHub Actions run executed.
- Risks: local Langfuse uses the upstream default `CHANGEME` secrets and binds ports 3000/9090 on 0.0.0.0; k8s memory limit 512Mi is likely too small for torch plus sentence-transformers (untested); conversation memory is process-global.
