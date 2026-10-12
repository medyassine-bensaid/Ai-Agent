"""
rag_evaluator.py - RAG evaluation using Groq LLM-as-a-Judge.

Metrics tracked (all pushed to Langfuse):
- context_relevance   : do retrieved chunks relate to the question?
- faithfulness        : is the answer grounded in the retrieved context?
- answer_relevance    : does the answer address the question?

Run from cron/CI or after agent sessions:
    python -m tools.rag_evaluator          # evaluates RAG traces of the last hour
"""

import json
import os
import re
from datetime import datetime, timedelta, timezone

from openai import OpenAI

MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b")
RAG_METRICS = ("context_relevance", "faithfulness", "answer_relevance")

_groq = None


def _get_groq() -> OpenAI:
    global _groq
    if _groq is None:
        _groq = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
    return _groq


def _judge(prompt: str, metric: str) -> float | None:
    """Return the metric value in [0, 1], or None if the judge failed."""
    try:
        r = _get_groq().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            # reasoning model: leave room for hidden reasoning tokens
            max_tokens=2048,
            temperature=0.0,
            reasoning_effort="low",
            response_format={"type": "json_object"},
        )
        raw = re.sub(r"```json|```", "", r.choices[0].message.content or "").strip()
        v = json.loads(raw).get(metric)
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not 0.0 <= v <= 1.0
        ):
            raise ValueError(f"{metric} missing or out of range: {v!r}")
        return float(v)
    except Exception as e:
        print(f"  [rag_eval] judge error ({metric}): {e}")
        return None


CONTEXT_RELEVANCE_PROMPT = """\
Does this context contain information relevant to the question?
Score context_relevance 0.0-1.0.
Respond ONLY with JSON: {{"context_relevance":0.0,"reasoning":"one line"}}

Question: {question}
Context: {context}"""

FAITHFULNESS_PROMPT = """\
Is this answer fully supported by the provided context (no hallucinations)?
Score faithfulness 0.0-1.0.
Respond ONLY with JSON: {{"faithfulness":0.0,"reasoning":"one line"}}

Context: {context}
Answer: {answer}"""

ANSWER_RELEVANCE_PROMPT = """\
Does this answer directly address the question?
Score answer_relevance 0.0-1.0.
Respond ONLY with JSON: {{"answer_relevance":0.0,"reasoning":"one line"}}

Question: {question}
Answer: {answer}"""


def score_rag(question: str, context: str, answer: str) -> dict:
    """Judge one RAG interaction. Failed metrics are omitted (never defaulted to 0/1)."""
    prompts = {
        "context_relevance": CONTEXT_RELEVANCE_PROMPT.format(
            question=question[:1000], context=context[:4000]
        ),
        "faithfulness": FAITHFULNESS_PROMPT.format(
            context=context[:4000], answer=answer[:6000]
        ),
        "answer_relevance": ANSWER_RELEVANCE_PROMPT.format(
            question=question[:1000], answer=answer[:6000]
        ),
    }
    scores = {}
    for metric, prompt in prompts.items():
        v = _judge(prompt, metric)
        if v is not None:
            scores[metric] = v
    return scores


def evaluate_rag_trace(trace_id: str, question: str, context: str, answer: str) -> dict:
    """Score a single RAG trace and push all scores to Langfuse."""
    from langfuse import get_client

    langfuse = get_client()
    scores = score_rag(question, context, answer)
    for name, value in scores.items():
        langfuse.create_score(
            trace_id=trace_id,
            name=f"rag/{name}",
            value=value,
            data_type="NUMERIC",
        )
        print(f"  [rag_eval] {name}={value:.2f} -> trace {trace_id[:12]}...")
    if len(scores) < len(RAG_METRICS):
        langfuse.create_score(
            trace_id=trace_id,
            name="rag/judge-error",
            value=1,
            data_type="BOOLEAN",
            comment=f"missing: {set(RAG_METRICS) - set(scores)}",
        )
    langfuse.flush()
    return scores


def evaluate_recent_rag_traces(hours: float = 1) -> int:
    """Find recent agent runs that called the RAG tool and evaluate them.

    Uses the Langfuse v4 read APIs: the 'agent-run' root observation holds the
    question/answer, the 'rag_tool' tool observation holds the retrieved context.
    """
    from monitoring import langfuse_api

    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    rag_calls = langfuse_api.list_observations(
        since, name="rag_tool", fields="core,basic,io", max_items=200
    )
    if not rag_calls:
        print("[rag_eval] No RAG tool calls found.")
        return 0

    already = {
        langfuse_api.score_trace_id(s)
        for s in langfuse_api.list_scores(since=since, name="rag/faithfulness")
    }
    runs = {r["traceId"]: r for r in langfuse_api.list_agent_runs(since, max_items=200)}

    evaluated = 0
    contexts = {}
    for obs in rag_calls:
        contexts.setdefault(obs["traceId"], []).append(str(obs.get("output") or ""))
    for trace_id, ctx in contexts.items():
        run = runs.get(trace_id)
        if not run or trace_id in already:
            continue
        inp = run.get("input")
        if isinstance(inp, str):
            try:
                inp = json.loads(inp)
            except ValueError:
                pass
        question = (
            (inp.get("args") or [""])[0] if isinstance(inp, dict) else str(inp or "")
        )
        answer = str(run.get("output") or "")
        if question and answer:
            evaluate_rag_trace(trace_id, question, "\n\n".join(ctx), answer)
            evaluated += 1

    print(f"[rag_eval] Evaluated {evaluated} RAG traces.")
    return evaluated


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    evaluate_recent_rag_traces(hours=1)
