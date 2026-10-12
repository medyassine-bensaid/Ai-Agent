"""
langfuse_api.py - Read-side helpers for the Langfuse public REST API.

Self-hosted Langfuse v4 runs in "events_only" mode, where the legacy read
endpoints (GET /api/public/traces, /api/public/traces/{id}, /api/public/scores)
return 404. These helpers use their replacements:

- GET /api/public/v2/observations  (cursor pagination, fromStartTime/toStartTime)
- GET /api/public/v3/scores        (cursor pagination, fromTimestamp/toTimestamp)

A "trace" in v4 is the set of observations sharing a traceId; the root
observation of an agent run is the "agent-run" span created in main.py.
"""

import base64
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional

import requests

TIMEOUT = 20


def base_url() -> str:
    return os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000").rstrip("/")


def auth_headers() -> dict:
    """Basic Auth headers for the Langfuse REST API."""
    credentials = (
        f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}"
    )
    encoded = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}", "Content-Type": "application/json"}


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _paginate(path: str, params: dict, max_items: int) -> Iterator[dict]:
    """Yield items across cursor pages. HTTP errors propagate (no silent empty result)."""
    params = dict(params)
    fetched = 0
    while True:
        resp = requests.get(
            f"{base_url()}{path}",
            params=params,
            headers=auth_headers(),
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        body = resp.json()
        for item in body.get("data", []):
            yield item
            fetched += 1
            if fetched >= max_items:
                return
        cursor = (body.get("meta") or {}).get("cursor")
        if not cursor or not body.get("data"):
            return
        params["cursor"] = cursor


def list_observations(
    since: datetime,
    until: Optional[datetime] = None,
    *,
    trace_id: Optional[str] = None,
    name: Optional[str] = None,
    fields: str = "core,basic",
    max_items: int = 1000,
) -> list:
    until = until or datetime.now(timezone.utc) + timedelta(minutes=5)
    params = {
        "fromStartTime": iso(since),
        "toStartTime": iso(until),
        "fields": fields,
        "limit": 200,
    }
    if trace_id:
        params["traceId"] = trace_id
    if name:
        params["filter"] = json.dumps(
            [{"type": "string", "column": "name", "operator": "=", "value": name}]
        )
    return list(_paginate("/api/public/v2/observations", params, max_items))


def list_agent_runs(since: datetime, max_items: int = 100) -> list:
    """Root 'agent-run' observations (one per agent trace), newest first."""
    runs = list_observations(
        since, name="agent-run", fields="core,basic,io", max_items=max_items
    )
    return sorted(runs, key=lambda o: o.get("startTime", ""), reverse=True)


def list_scores(
    *,
    trace_ids: Optional[list] = None,
    since: Optional[datetime] = None,
    name: Optional[str] = None,
    max_items: int = 5000,
) -> list:
    """Scores with their subject (trace/observation) attached."""
    params = {"fields": "details,subject", "limit": 100}
    if trace_ids:
        params["traceId"] = ",".join(trace_ids)
    if since:
        params["fromTimestamp"] = iso(since)
    if name:
        params["name"] = name
    return list(_paginate("/api/public/v3/scores", params, max_items))


def score_trace_id(score: dict) -> Optional[str]:
    subject = score.get("subject") or {}
    return subject.get("id") if subject.get("kind") == "trace" else score.get("traceId")
