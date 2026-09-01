#!/usr/bin/env python3
"""
langfuse_exporter.py — Prometheus exporter for Langfuse scores.
Fetches the most recent scores via /api/public/scores and exposes averages.
Compatible with self-hosted Langfuse v4.5.1.
"""
import os, time, base64
import requests
from prometheus_client import start_http_server, Gauge

BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
PUBLIC_KEY = os.environ["LANGFUSE_PUBLIC_KEY"]
SECRET_KEY = os.environ["LANGFUSE_SECRET_KEY"]

def _auth_headers():
    credentials = f"{PUBLIC_KEY}:{SECRET_KEY}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/json",
    }

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

def collect():
    headers = _auth_headers()
    scores_url = f"{BASE_URL}/api/public/scores"

    # Fetch first page with maximum allowed limit (test showed limit=100 works)
    params = {"page": 1, "limit": 100}
    print("[exporter] Fetching scores...")
    resp = requests.get(scores_url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    scores_list = resp.json().get("data", [])
    print(f"[exporter] Fetched {len(scores_list)} scores")
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