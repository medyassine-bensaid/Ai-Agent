import os, re, json, base64
from datetime import datetime, timedelta, timezone
from openai import OpenAI
from langfuse import Langfuse

# ── always use env var, fallback to localhost (OSS) not cloud ────────────────
def _make_lf():
    return Langfuse(
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        host=os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000"),
    )

groq_judge = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)
JUDGE_MODEL = "llama-3.3-70b-versatile"

JUDGE_PROMPT = """\
You are an expert evaluator for a DevOps AI assistant.
Reply ONLY with raw JSON, no markdown:
{{"relevance":0.0,"faithfulness":0.0,"completeness":0.0,"hallucination_risk":0.0,"reasoning":"one line"}}
hallucination_risk: 1.0=grounded, 0.0=hallucinated
Question: {question}
Answer: {answer}"""

def _llm_judge(question: str, answer: str) -> dict:
    try:
        r = groq_judge.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role":"user","content":JUDGE_PROMPT.format(
                question=question[:500], answer=answer[:1500])}],
            max_tokens=300, temperature=0.0)
        raw = re.sub(r"```json|```","",r.choices[0].message.content).strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(m.group()) if m else {}
    except Exception as e:
        print(f"  [scorer] judge error: {e}")
        return {}

def _heuristic(answer: str) -> float:
    if not answer: return 0.0
    s = 0.4 if len(answer)>500 else (0.3 if len(answer)>200 else 0.1)
    if re.search(r"i apologize|error generating", answer, re.I): s -= 0.3
    if re.search(r"\d+\.", answer): s += 0.2
    if re.search(r"^[-*]", answer, re.M): s += 0.2
    if re.search(r"```", answer): s += 0.2
    return max(0.0, min(1.0, s))

def inline_score(trace_id: str, question: str, answer: str):
    """Called from main.py with the current trace_id, query, and answer.
    Creates a Langfuse client and pushes evaluation scores."""
    if not trace_id or not answer:
        print(f"  [scorer] SKIP — trace_id={trace_id!r} answer_len={len(answer or '')}")
        return

    lf_client = _make_lf()
    print(f"  [scorer] scoring trace {trace_id[:12]}…")
    judge = _llm_judge(question, answer)

    for metric in ("relevance","faithfulness","completeness","hallucination_risk"):
        v = judge.get(metric)
        if isinstance(v, (int,float)):
            lf_client.create_score(
                trace_id=trace_id,
                name=f"llm-judge/{metric}",
                value=float(v),
                data_type="NUMERIC",
                comment=judge.get("reasoning","")
            )

    lf_client.create_score(
        trace_id=trace_id,
        name="heuristic/quality",
        value=_heuristic(answer),
        data_type="NUMERIC"
    )
    lf_client.create_score(
        trace_id=trace_id,
        name="has-error",
        value=0 if re.search(r"i apologize|error generating",answer,re.I) else 1,
        data_type="BOOLEAN"
    )

    lf_client.flush()
    print(f"  [scorer] ✓ relevance={judge.get('relevance','n/a')} heuristic={_heuristic(answer):.2f}")

if __name__ == "__main__":
    print("[scorer] This module provides inline scoring for the agent.")
    print("[scorer] It is not meant to be run standalone (batch scoring via REST is disabled).")
    print("[scorer] Scoring happens automatically inside `main.py` after each query.")