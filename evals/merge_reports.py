#!/usr/bin/env python3
"""
merge_reports.py - Combine eval_report.json files (e.g. one per CI matrix job) into a
single report and gate decision.

Usage:
    python -m evals.merge_reports <dir-or-json> [...] [--output-dir reports/combined]
Exit codes follow evals/run_eval.py: 0 passed, 1 failed, 2 blocked / nothing evaluated.
"""

import argparse
import json
import sys
from pathlib import Path

from evals.run_eval import render_markdown, summarize_records


def load_reports(paths: list) -> list:
    reports = []
    for p in map(Path, paths):
        files = sorted(p.rglob("eval_report.json")) if p.is_dir() else [p]
        for f in files:
            reports.append(json.loads(f.read_text(encoding="utf-8")))
    return reports


def merge(reports: list) -> tuple:
    records = [case for rep in reports for case in rep.get("cases", [])]
    summary = summarize_records(records)
    first = reports[0]["summary"] if reports else {}
    summary.update(
        {
            "run_id": ", ".join(r["summary"].get("run_id", "?") for r in reports),
            "model": first.get("model"),
            "judge_model": first.get("judge_model"),
            "groups": [
                {
                    "run_id": r["summary"].get("run_id"),
                    "selection": r["summary"].get("selection"),
                    "gate": r["summary"].get("gate"),
                    "total": r["summary"].get("total"),
                }
                for r in reports
            ],
        }
    )
    return summary, records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Merge evaluation reports")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output-dir", default="reports/combined")
    args = parser.parse_args(argv)

    reports = load_reports(args.paths)
    if not reports:
        print("[merge] ERROR: no eval_report.json found - evaluation did not run")
        return 2
    summary, records = merge(reports)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_report.json").write_text(
        json.dumps(
            {"summary": summary, "cases": records}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    md = render_markdown(
        summary, records, title=f"Combined agent evaluation ({len(reports)} groups)"
    )
    (out / "eval_report.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"[merge] GATE {summary['gate']}")
    return summary["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
