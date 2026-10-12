import json
import os
import re
import time

from langfuse import get_client, observe

# Drop-in OpenAI client: each judge call is recorded as a generation (model, tokens)
from langfuse.openai import OpenAI

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b")
# gpt-oss is a reasoning model: hidden reasoning tokens count against max_tokens.
# 300 tokens (the old budget) was regularly exhausted before any JSON was emitted.
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "2048"))
# Search results are often >6k chars; truncating them hid the very facts being judged
JUDGE_EVIDENCE_CHARS = int(os.environ.get("JUDGE_EVIDENCE_CHARS", "20000"))
JUDGE_METRICS = ("relevance", "faithfulness", "completeness", "hallucination_risk")

_groq_judge = None


def _get_judge_client() -> OpenAI:
    """Created lazily so importing this module never requires GROQ_API_KEY."""
    global _groq_judge
    if _groq_judge is None:
        _groq_judge = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
            max_retries=5,  # SDK backs off on 429 (Groq tokens-per-minute limit)
        )
    return _groq_judge


JUDGE_PROMPT = """\
You are an expert evaluator for a DevOps AI assistant.
Score the answer on each metric from 0.0 to 1.0:
- relevance: does the answer address the question that was asked?
- faithfulness: are the answer's claims supported by the evidence below or by
  well-established, non-time-sensitive DevOps knowledge?
- completeness: does it cover what the question needs?
- hallucination_risk: 1.0 = fully grounded, 0.0 = contains invented facts
  (e.g. versions, dates, URLs or tool runs not supported by the evidence)
Reply ONLY with raw JSON, no markdown:
{{"relevance":0.0,"faithfulness":0.0,"completeness":0.0,"hallucination_risk":0.0,"reasoning":"one line"}}

Question: {question}

Evidence retrieved by the agent's tools (empty if it answered from its own knowledge):
{evidence}

Answer: {answer}"""


def _parse_judge(raw: str) -> dict:
    """Extract and validate the judge JSON. Raises ValueError if unusable."""
    raw = re.sub(r"```json|```", "", raw or "").strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in judge output: {raw[:120]!r}")
    data = json.loads(m.group())
    out = {}
    for metric in JUDGE_METRICS:
        v = data.get(metric)
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not 0.0 <= v <= 1.0
        ):
            raise ValueError(f"judge metric {metric!r} missing or out of range: {v!r}")
        out[metric] = float(v)
    out["reasoning"] = str(data.get("reasoning", ""))[:500]
    return out


@observe(name="llm-judge", as_type="evaluator")
def _llm_judge(
    question: str, answer: str, evidence: str = "", retries: int = 1
) -> dict:
    """Return validated judge scores, or {"error": "..."} if the judge failed."""
    prompt = JUDGE_PROMPT.format(
        question=question[:1000],
        evidence=(evidence or "(none)")[:JUDGE_EVIDENCE_CHARS],
        answer=answer[:6000],
    )
    last_error = "unknown"
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(5 * attempt)  # back off before re-asking the judge
        try:
            r = _get_judge_client().chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=JUDGE_MAX_TOKENS,
                temperature=0.0,
                reasoning_effort="low",
                response_format={"type": "json_object"},
            )
            choice = r.choices[0]
            if choice.finish_reason == "length":
                raise ValueError("judge hit max_tokens before finishing")
            return _parse_judge(choice.message.content)
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            print(f"  [scorer] judge error: {last_error[:200]}")
    return {"error": last_error}


def _heuristic(answer: str) -> float:
    if not answer:
        return 0.0
    s = 0.4 if len(answer) > 500 else (0.3 if len(answer) > 200 else 0.1)
    if re.search(r"i apologize|error generating", answer, re.I):
        s -= 0.3
    if re.search(r"\d+\.", answer):
        s += 0.2
    if re.search(r"^[-*]", answer, re.M):
        s += 0.2
    if re.search(r"```", answer):
        s += 0.2
    return max(0.0, min(1.0, s))


# Public name used by tests / external callers
score_answer_quality = _heuristic


def is_error_answer(answer: str) -> bool:
    return bool(re.search(r"i apologize|error generating", answer or "", re.I))


def inline_score(
    trace_id: str,
    question: str,
    answer: str,
    evidence: str = "",
    details: dict | None = None,
) -> dict:
    """Called from main.py with the current trace_id, query, answer and tool evidence.
    Pushes evaluation scores to Langfuse and returns {score_name: value}.
    If `details` is given, a judge failure reason is stored in details["judge_error"]."""
    if not trace_id or not answer:
        print(f"  [scorer] SKIP - trace_id={trace_id!r} answer_len={len(answer or '')}")
        return {}

    lf_client = get_client()
    print(f"  [scorer] scoring trace {trace_id[:12]}...")
    judge = _llm_judge(question, answer, evidence)
    submitted = {}

    def _score(name, value, data_type, comment=None):
        lf_client.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            data_type=data_type,
            comment=comment,
        )
        submitted[name] = value

    if "error" in judge:
        # Make judge failures visible instead of silently producing no scores
        _score("llm-judge/error", 1, "BOOLEAN", comment=judge["error"][:500])
        if details is not None:
            details["judge_error"] = judge["error"]
    else:
        for metric in JUDGE_METRICS:
            _score(
                f"llm-judge/{metric}",
                judge[metric],
                "NUMERIC",
                comment=judge["reasoning"],
            )

    _score("heuristic/quality", _heuristic(answer), "NUMERIC")
    # 1 = the answer is an error message, 0 = normal answer
    _score("has-error", 1 if is_error_answer(answer) else 0, "BOOLEAN")

    lf_client.flush()
    print(
        f"  [scorer] OK relevance={judge.get('relevance','n/a')} heuristic={_heuristic(answer):.2f}"
    )
    return submitted


if __name__ == "__main__":
    print("[scorer] This module provides inline scoring for the agent.")
    print("[scorer] It is not meant to be run standalone.")
    print("[scorer] Scoring happens automatically inside `main.py` after each query.")
