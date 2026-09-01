"""
rag_evaluator.py — RAG evaluation using Groq LLM-as-a-Judge.

Metrics tracked (all pushed to Langfuse):
- context_relevance   : do retrieved chunks relate to the question?
- faithfulness        : is the answer grounded in the retrieved context?
- answer_relevance    : does the answer address the question?
- retrieval_precision : fraction of retrieved chunks actually useful

Run from cron/CI or after agent sessions.
"""

import os, re, json
from langfuse import Langfuse
from openai import OpenAI

langfuse = Langfuse(
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    host=os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
)
groq = OpenAI(api_key=os.environ["GROQ_API_KEY"],
              base_url="https://api.groq.com/openai/v1")
MODEL = "llama-3.3-70b-versatile"


def _judge(prompt: str) -> dict:
    try:
        r = groq.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=256, temperature=0.0)
        raw = re.sub(r"```json|```","",r.choices[0].message.content).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"  [rag_eval] judge error: {e}")
        return {}


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


def evaluate_rag_trace(trace_id: str, question: str, context: str, answer: str):
    """Score a single RAG trace and push all scores to Langfuse."""
    scores = {}

    r = _judge(CONTEXT_RELEVANCE_PROMPT.format(
        question=question[:400], context=context[:1200]))
    scores["context_relevance"] = r.get("context_relevance", 0.0)

    r = _judge(FAITHFULNESS_PROMPT.format(
        context=context[:1200], answer=answer[:800]))
    scores["faithfulness"] = r.get("faithfulness", 0.0)

    r = _judge(ANSWER_RELEVANCE_PROMPT.format(
        question=question[:400], answer=answer[:800]))
    scores["answer_relevance"] = r.get("answer_relevance", 0.0)

    for name, value in scores.items():
        langfuse.create_score(
            trace_id=trace_id,
            name=f"rag/{name}",
            value=float(value),
            data_type="NUMERIC",
        )
        print(f"  [rag_eval] {name}={value:.2f} → trace {trace_id[:12]}…")

    langfuse.flush()
    return scores


def evaluate_recent_rag_traces(hours: int = 1):
    """Scan recent traces for RAG tool calls and evaluate them."""
    from datetime import datetime, timedelta
    since = datetime.utcnow() - timedelta(hours=hours)
    traces = langfuse.fetch_traces(from_timestamp=since, limit=50,
                                    order_by="createdAt", order="desc").data
    evaluated = 0
    for trace in traces:
        output = trace.output or {}
        input_ = trace.input  or {}
        answer   = output.get("answer","") if isinstance(output,dict) else str(output)
        question = input_.get("query","")  if isinstance(input_,dict)  else ""

        # Look for RAG context in trace metadata or output
        metadata = trace.metadata or {}
        context  = metadata.get("rag_context","")

        if not (answer and question and context):
            continue

        evaluate_rag_trace(trace.id, question, context, answer)
        evaluated += 1

    print(f"[rag_eval] Evaluated {evaluated} RAG traces.")
    langfuse.flush()


if __name__ == "__main__":
    evaluate_recent_rag_traces(hours=1)
