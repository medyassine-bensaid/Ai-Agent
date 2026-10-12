#!/usr/bin/env python3
"""
junit_summary.py - Summarise JUnit XML files (one per test suite) as a Markdown table.

Usage: python scripts/junit_summary.py <dir> [--output summary.md]
Exit code 1 if any test failed or errored, or if no JUnit file was found.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def suite_counts(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0}
    failed_names = []
    for s in suites:
        for key in ("tests", "failures", "errors", "skipped"):
            counts[key] += int(s.get(key, 0))
        counts["time"] += float(s.get("time", 0))
        for case in s.iter("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                failed_names.append(f"{case.get('classname')}::{case.get('name')}")
    counts["passed"] = (
        counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    )
    counts["failed_names"] = failed_names
    return counts


def render(rows: dict) -> str:
    lines = [
        "## Unit test results",
        "",
        "| Suite | Tests | Passed | Failed | Errors | Skipped | Time s |",
        "|---|---|---|---|---|---|---|",
    ]
    total = {
        "tests": 0,
        "passed": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "time": 0.0,
    }
    for name, c in sorted(rows.items()):
        lines.append(
            f"| {name} | {c['tests']} | {c['passed']} | {c['failures']} | {c['errors']} "
            f"| {c['skipped']} | {c['time']:.1f} |"
        )
        for k in total:
            total[k] += c[k]
    lines.append(
        f"| **Total** | **{total['tests']}** | **{total['passed']}** | **{total['failures']}** "
        f"| **{total['errors']}** | **{total['skipped']}** | {total['time']:.1f} |"
    )
    failed = [n for c in rows.values() for n in c["failed_names"]]
    if failed:
        lines += ["", "### Failed tests", ""] + [f"- `{n}`" for n in failed]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    files = sorted(Path(args.directory).rglob("junit*.xml"))
    if not files:
        print("No JUnit files found")
        return 1
    rows = {f.stem.replace("junit-", ""): suite_counts(f) for f in files}
    md = render(rows)
    print(md)
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
    bad = sum(c["failures"] + c["errors"] for c in rows.values())
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
