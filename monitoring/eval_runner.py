#!/usr/bin/env python3
"""
eval_runner.py - run by CI/CD after deploy to gate on recent production scores.

Reads the agent-run traces of the last EVAL_WINDOW_HOURS from Langfuse and the
scores attached to them (written inline by monitoring/langfuse_scorer.py), then
applies a quality gate. Exit codes:
  0 = gate passed
  1 = gate failed (scores below threshold)
  2 = could not evaluate (Langfuse unreachable, or no scored traces) - never a pass
"""

import os
import sys
from datetime import datetime, timedelta, timezone

from monitoring import langfuse_api

QUALITY_THRESHOLD = float(
    os.environ.get("QUALITY_THRESHOLD", "0.5")
)  # heuristic/quality mean
JUDGE_THRESHOLD = float(
    os.environ.get("JUDGE_THRESHOLD", "0.7")
)  # each llm-judge/* mean
MAX_ERROR_RATE = float(os.environ.get("MAX_ERROR_RATE", "0.2"))  # has-error mean
WINDOW_HOURS = float(os.environ.get("EVAL_WINDOW_HOURS", "1"))
GATED_JUDGE_METRICS = (
    "llm-judge/relevance",
    "llm-judge/faithfulness",
    "llm-judge/hallucination_risk",
)


def summarize(scores: list) -> dict:
    """Group numeric/boolean score values by name -> {name: [values]}."""
    buckets = {}
    for sc in scores:
        v = sc.get("value")
        if isinstance(v, bool):
            v = 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            buckets.setdefault(sc["name"], []).append(float(v))
    return buckets


def gate(buckets: dict) -> list:
    """Return the list of gate failures (empty list = pass)."""
    failures = []

    def mean(name):
        vals = buckets.get(name) or []
        return sum(vals) / len(vals) if vals else None

    q = mean("heuristic/quality")
    if q is None:
        failures.append("no heuristic/quality scores")
    elif q < QUALITY_THRESHOLD:
        failures.append(f"heuristic/quality mean {q:.2f} < {QUALITY_THRESHOLD}")

    for name in GATED_JUDGE_METRICS:
        m = mean(name)
        if m is None:
            failures.append(f"no {name} scores (judge not running?)")
        elif m < JUDGE_THRESHOLD:
            failures.append(f"{name} mean {m:.2f} < {JUDGE_THRESHOLD}")

    err = mean("has-error")
    if err is not None and err > MAX_ERROR_RATE:
        failures.append(f"error rate {err:.2f} > {MAX_ERROR_RATE}")
    return failures


def main() -> int:
    since = datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)
    try:
        runs = langfuse_api.list_agent_runs(since, max_items=50)
        print(f"[eval] Found {len(runs)} agent-run traces in last {WINDOW_HOURS}h")
        trace_ids = [r["traceId"] for r in runs]
        scores = langfuse_api.list_scores(trace_ids=trace_ids) if trace_ids else []
    except Exception as e:
        print(
            f"[eval] ERROR: could not read Langfuse at {langfuse_api.base_url()}: {e}"
        )
        return 2

    buckets = summarize(scores)
    if not buckets:
        print("[eval] ERROR: no scored traces found - nothing was evaluated")
        return 2

    for name in sorted(buckets):
        vals = buckets[name]
        print(f"[eval] {name:32} mean={sum(vals)/len(vals):.3f} n={len(vals)}")

    failures = gate(buckets)
    if failures:
        for f in failures:
            print(f"[eval] FAIL: {f}")
        print("[eval] QUALITY GATE FAILED")
        return 1
    print("[eval] QUALITY GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
