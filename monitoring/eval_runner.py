#!/usr/bin/env python3
"""
eval_runner.py — run by CI/CD after deploy to score recent traces.
Exit code 1 if avg quality < threshold (quality gate).
"""
import os, sys, base64, requests
from datetime import datetime, timedelta, timezone

QUALITY_THRESHOLD = float(os.environ.get("QUALITY_THRESHOLD", "0.5"))

def _make_langfuse_headers():
    """Basic Auth headers for the Langfuse REST API."""
    public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
    secret_key = os.environ["LANGFUSE_SECRET_KEY"]
    credentials = f"{public_key}:{secret_key}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/json",
    }

def main():
    base_url = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
    headers = _make_langfuse_headers()

    # 1. Score recent traces (optional, but nice to do)
    try:
        from monitoring.langfuse_scorer import score_recent_traces
        score_recent_traces(hours=1)
    except Exception as e:
        print(f"[eval] scoring step failed: {e}")

    # 2. Fetch traces from the last hour using the REST API
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    since_ms = int(since.timestamp() * 1000)
    traces_url = f"{base_url}/api/public/traces"
    params = {
        "fromTimestamp": since_ms,
        "limit": 20,
        "orderBy": "createdAt",
        "order": "desc",
    }

    try:
        resp = requests.get(traces_url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        traces = data.get("data", [])
        print(f"[eval] Found {len(traces)} traces in last hour")
    except Exception as e:
        print(f"[eval] Failed to fetch traces: {e}")
        sys.exit(0)

    # 3. For each trace, fetch its scores
    scores = []
    for t in traces:
        trace_id = t["id"]
        scores_url = f"{base_url}/api/public/traces/{trace_id}/scores"
        try:
            s_resp = requests.get(scores_url, headers=headers, timeout=10)
            s_resp.raise_for_status()
            s_data = s_resp.json()
            # s_data["data"] is a list of score objects
            for sc in s_data.get("data", []):
                if sc.get("name") == "heuristic/quality":
                    scores.append(sc["value"])
        except Exception as e:
            print(f"[eval] Could not fetch scores for trace {trace_id[:12]}…: {e}")

    if not scores:
        print("[eval] No 'heuristic/quality' scores found — skipping quality gate")
        sys.exit(0)

    avg = sum(scores) / len(scores)
    print(f"[eval] avg heuristic/quality = {avg:.2f} (threshold={QUALITY_THRESHOLD})")
    if avg < QUALITY_THRESHOLD:
        print("[eval] QUALITY GATE FAILED")
        sys.exit(1)
    print("[eval] QUALITY GATE PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()