#!/usr/bin/env python3
"""
Post a PR comment summarising the latest Langfuse quality scores.
Uses the Langfuse v4 read APIs (see monitoring/langfuse_api.py).
"""
import os
from datetime import datetime, timedelta, timezone

import requests

from monitoring import langfuse_api

SHOWN = ("heuristic/quality", "llm-judge/relevance", "llm-judge/faithfulness")


def main():
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    runs = langfuse_api.list_agent_runs(since, max_items=5)
    if not runs:
        print("No recent traces found, skipping PR comment.")
        return

    trace_ids = [r["traceId"] for r in runs]
    by_trace = {}
    for sc in langfuse_api.list_scores(trace_ids=trace_ids):
        by_trace.setdefault(langfuse_api.score_trace_id(sc), {})[sc["name"]] = sc["value"]

    lines = ["## LLM Eval Scores", "| Trace | " + " | ".join(SHOWN) + " |",
             "|---|" + "---|" * len(SHOWN)]
    for tid in trace_ids:
        vals = by_trace.get(tid, {})
        cells = [f"{vals[n]:.2f}" if isinstance(vals.get(n), (int, float)) else "n/a" for n in SHOWN]
        lines.append(f"| {tid[:8]} | " + " | ".join(cells) + " |")

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
    resp.raise_for_status()
    print(f"PR comment posted, status: {resp.status_code}")

if __name__ == "__main__":
    main()
