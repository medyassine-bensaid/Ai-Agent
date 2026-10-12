#!/usr/bin/env python3
"""
langfuse_exporter.py - Prometheus exporter for Langfuse scores.
Reads recent scores via the v3 scores API (Langfuse v4; the legacy
/api/public/scores endpoint is not available in v4 events_only mode)
and exposes their averages.
"""

import os
import time
from datetime import datetime, timedelta, timezone

from prometheus_client import Gauge, start_http_server

from monitoring import langfuse_api

WINDOW_HOURS = float(os.environ.get("EXPORTER_WINDOW_HOURS", "24"))

g_relevance = Gauge("llm_judge_relevance", "LLM judge relevance score")
g_faithfulness = Gauge("llm_judge_faithfulness", "LLM judge faithfulness score")
g_completeness = Gauge("llm_judge_completeness", "LLM judge completeness score")
g_hallucination = Gauge("llm_judge_hallucination_risk", "LLM judge hallucination risk")
g_heuristic = Gauge("llm_heuristic_quality", "Heuristic quality score")
g_error_rate = Gauge("llm_error_rate", "Fraction of traces with errors")
g_scores_total = Gauge("llm_scores_fetched", "Total scores fetched")

SCORE_MAP = {
    "llm-judge/relevance": g_relevance,
    "llm-judge/faithfulness": g_faithfulness,
    "llm-judge/completeness": g_completeness,
    "llm-judge/hallucination_risk": g_hallucination,
    "heuristic/quality": g_heuristic,
}


def collect():
    """Average the scores of the last EXPORTER_WINDOW_HOURS (v3 scores API)."""
    since = datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)
    print("[exporter] Fetching scores...")
    scores_list = langfuse_api.list_scores(since=since, max_items=5000)
    print(f"[exporter] Fetched {len(scores_list)} scores")
    g_scores_total.set(len(scores_list))

    buckets = {k: [] for k in SCORE_MAP}
    errors = []

    for sc in scores_list:
        name = sc.get("name")
        value = sc.get("value")
        if isinstance(value, bool):
            value = 1.0 if value else 0.0
        if name in buckets:
            buckets[name].append(value)
        if name == "has-error":
            errors.append(value)  # 1 = error answer

    for name, gauge in SCORE_MAP.items():
        vals = buckets[name]
        if vals:
            avg = sum(vals) / len(vals)
            gauge.set(avg)
            print(f"[exporter] {name}: {avg:.3f} (from {len(vals)} values)")

    if errors:
        g_error_rate.set(sum(errors) / len(errors))


if __name__ == "__main__":
    start_http_server(8001)
    print("[exporter] Prometheus metrics at http://localhost:8001/metrics")
    while True:
        try:
            collect()
        except Exception as e:
            print(f"[exporter] Collection error: {e}")
        time.sleep(30)
