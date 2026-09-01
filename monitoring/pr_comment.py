#!/usr/bin/env python3
"""
Post a PR comment summarising the latest Langfuse quality scores.
Uses the Langfuse REST API (compatible with self-hosted v4.5.1).
"""
import os, sys, base64, requests
from datetime import datetime, timedelta, timezone

def _auth_headers():
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
    headers = _auth_headers()

    # Fetch recent traces
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    since_ms = int(since.timestamp() * 1000)
    traces_url = f"{base_url}/api/public/traces"
    params = {
        "fromTimestamp": since_ms,
        "limit": 5,
        "orderBy": "createdAt",
        "order": "desc",
    }
    resp = requests.get(traces_url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    traces = resp.json().get("data", [])

    if not traces:
        print("No recent traces found, skipping PR comment.")
        return

    # Build comment table
    lines = ["## 🤖 LLM Eval Scores", "| Trace | Score |", "|---|---|"]
    for t in traces:
        trace_id = t["id"]
        scores_url = f"{base_url}/api/public/traces/{trace_id}/scores"
        try:
            s_resp = requests.get(scores_url, headers=headers, timeout=10)
            s_resp.raise_for_status()
            scores = s_resp.json().get("data", [])
        except Exception:
            continue

        for sc in scores:
            if sc.get("name") == "heuristic/quality":
                lines.append(f"| {trace_id[:8]} | {sc['value']:.2f} |")
                break   # only one row per trace

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
    print(f"PR comment posted, status: {resp.status_code}")

if __name__ == "__main__":
    main()