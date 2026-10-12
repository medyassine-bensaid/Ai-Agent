#!/usr/bin/env python3
"""
audit_summary.py - Turn `pip-audit -f json` output into a Markdown table and GitHub
warning annotations (one per vulnerable package).

Usage: python scripts/audit_summary.py audit.json [--output audit.md] [--annotate]
Always exits 0: the dependency audit is advisory (reported, not blocking).
"""

import argparse
import json
import sys
from pathlib import Path


def vulnerable(report: dict) -> list:
    deps = report.get("dependencies", report if isinstance(report, list) else [])
    return [d for d in deps if d.get("vulns")]


def render(deps: list) -> str:
    total = sum(len(d["vulns"]) for d in deps)
    lines = ["## Dependency audit (advisory)", ""]
    if not deps:
        return (
            "\n".join(lines + ["No known vulnerabilities in requirements.lock."]) + "\n"
        )
    lines += [
        f"**{total} known vulnerabilities in {len(deps)} packages.**",
        "",
        "| Package | Version | Advisories | Fixed in |",
        "|---|---|---|---|",
    ]
    for d in sorted(deps, key=lambda d: d["name"]):
        ids = ", ".join(v["id"] for v in d["vulns"])
        fixes = sorted({f for v in d["vulns"] for f in v.get("fix_versions", [])})
        lines.append(
            f"| {d['name']} | {d['version']} | {ids} | {', '.join(fixes) or 'no fix'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--output")
    parser.add_argument("--annotate", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"::warning title=Dependency audit::could not read pip-audit report: {e}")
        return 0
    deps = vulnerable(report)
    md = render(deps)
    print(md)
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
    if args.annotate:
        for d in deps:
            fixes = sorted({f for v in d["vulns"] for f in v.get("fix_versions", [])})
            print(
                f"::warning title=Vulnerable dependency {d['name']} {d['version']}::"
                f"{len(d['vulns'])} advisories; fixed in: {', '.join(fixes) or 'no fix yet'}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
