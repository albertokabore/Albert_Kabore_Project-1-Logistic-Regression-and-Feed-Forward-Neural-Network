"""Append the AI disclosure to the computed report in every submission format."""

from pathlib import Path

from sentiment.reporting import build_report

ROOT = Path(__file__).resolve().parent


def build_submission_report(result, destination):
    report = build_report(result, destination)
    disclosure = (ROOT / "AI_USE_DISCLOSURE.md").read_text(encoding="utf-8")
    report = report.rstrip() + "\n\n" + disclosure.strip() + "\n"
    Path(destination).write_text(report, encoding="utf-8")
    return report
