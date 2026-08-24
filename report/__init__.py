"""Reporting package."""
from report.html_report import render_report
from report.metrics import TRADING_DAYS, compute_metrics

__all__ = ["render_report", "compute_metrics", "TRADING_DAYS"]
