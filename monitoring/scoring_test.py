"""
scoring_test.py - Run ONCE to verify scorer + Langfuse connection work.

Usage:
    python scoring_test.py

If a score appears in Langfuse → SDK connectivity is fine.
If scores tab stays empty → check your env vars / network.
"""
import os
from pathlib import Path
from langfuse import Langfuse
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

lf = Langfuse(
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    host=os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000"),
)

# 1. Create a test trace using v4.5.1 API
with lf.start_as_current_observation(
    name="score-connectivity-test",
    input={"query": "test question"},
    output={"answer": "test answer - this is a connectivity check"},
) as obs:
    trace_id = obs.trace_id
    print(f"[+] Trace created: {trace_id}")

    # 2. Attach a score directly — this is what langfuse_scorer.py does
    lf.create_score(
        trace_id=trace_id,
        name="test/manual-score",
        value=0.9,
        data_type="NUMERIC",
        comment="connectivity test",
    )

lf.flush()
print("[+] Score pushed. Check Langfuse → Scores tab for 'test/manual-score'.")
print(f"[+] Direct trace URL: {os.environ.get('LANGFUSE_BASE_URL','http://localhost:3000')}/traces/{trace_id}")