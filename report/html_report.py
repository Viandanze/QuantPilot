"""Self-contained HTML report rendering.

The report embeds everything (CSS, SVG charts) in a single file so it can be
archived or shared without assets. Charts are rendered server-side as SVG.
"""
from __future__ import annotations

import datetime as dt
import html
import math
import pathlib

import pandas as pd

VERSION = "0.1.0"


# --------------------------------------------------------------------- format
def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def fmt_pct(x, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}%" if _finite(x) else "n/a"


def fmt_num(x, digits: int = 2) -> str:
    return f"{x:.{digits}f}" if _finite(x) else "n/a"


def fmt_money(x) -> str:
    return f"{x:,.0f}" if _finite(x) else "n/a"


def fmt_date(x) -> str:
    return x.strftime("%Y-%m-%d") if hasattr(x, "strftime") else str(x)


def _cls(x) -> str:
    if not _finite(x):
        return ""
    return "pos" if x > 0 else ("neg" if x < 0 else "")


# ----------------------------------------------------------------------- svg
def _svg_line_chart(
    series_map: dict[str, tuple[int, list[float]]],
    dates: list,
    colors: dict[str, str],
    width: int = 940,
    height: int = 300,
    y_fmt=fmt_num,
    title: str = "",
) -> str:
    pad_l, pad_r, pad_t, pad_b = 76, 18, 30, 34
    iw, ih = width - pad_l - pad_r, height - pad_t - pad_b
    values = [v for _, vals in series_map.values() for v in vals if v == v]
    if not values or len(dates) < 2:
        return "<p class='sub'>not enough data to plot</p>"
    lo, hi = min(values), max(values)
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad
    n = len(dates)

    def x_at(i: int) -> float:
        return pad_l + iw * i / (n - 1)

    def y_at(v: float) -> float:
        return pad_t + ih - (v - lo) / (hi - lo) * ih

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" style="width:100%;height:auto;background:#fff;border:1px solid #e2e8f0;border-radius:8px">'
    ]
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = y_at(v)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + iw}" y2="{y:.1f}" stroke="#eef2f7"/>'
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{html.escape(y_fmt(v))}</text>'
        )
    for k in range(5):
        i = int((n - 1) * k / 4)
        parts.append(
            f'<text x="{x_at(i):.1f}" y="{height - 10}" text-anchor="middle" class="tick">{fmt_date(dates[i])}</text>'
        )
    for label, (start, vals) in series_map.items():
        pts = " ".join(
            f"{x_at(start + i):.1f},{y_at(v):.1f}" for i, v in enumerate(vals) if v == v
        )
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{colors[label]}" stroke-width="1.8"/>'
        )
    lx = pad_l + 8
    for label in series_map:
        parts.append(
            f'<rect x="{lx}" y="8" width="18" height="4" rx="2" fill="{colors[label]}"/>'
            f'<text x="{lx + 24}" y="13" class="tick" fill="#334155">{html.escape(label)}</text>'
        )
        lx += 24 + 8 * len(label) + 18
    if title:
        parts.append(
            f'<text x="{pad_l}" y="20" class="tick" font-size="12">{html.escape(title)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _svg_drawdown_chart(dd_values: list[float], dates: list, width: int = 940, height: int = 200) -> str:
    pad_l, pad_r, pad_t, pad_b = 76, 18, 26, 34
    iw, ih = width - pad_l - pad_r, height - pad_t - pad_b
    vals = [v if v == v else 0.0 for v in dd_values]
    lo = min(min(vals), -0.001)
    n = len(dates)

    def x_at(i: int) -> float:
        return pad_l + iw * i / (n - 1)

    def y_at(v: float) -> float:
        return pad_t + ih - (v - lo) / (0.0 - lo) * ih

    zero_y = y_at(0.0)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" style="width:100%;height:auto;background:#fff;border:1px solid #e2e8f0;border-radius:8px">'
    ]
    for k in range(4):
        v = lo * k / 3
        y = y_at(v)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + iw}" y2="{y:.1f}" stroke="#eef2f7"/>'
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{fmt_pct(v, 1)}</text>'
        )
    for k in range(5):
        i = int((n - 1) * k / 4)
        parts.append(
            f'<text x="{x_at(i):.1f}" y="{height - 10}" text-anchor="middle" class="tick">{fmt_date(dates[i])}</text>'
        )
    area = f"M {x_at(0):.1f},{zero_y:.1f} " + " ".join(
        f"L {x_at(i):.1f},{y_at(v):.1f}" for i, v in enumerate(vals)
    ) + f" L {x_at(n - 1):.1f},{zero_y:.1f} Z"
    parts.append(f'<path d="{area}" fill="rgba(220,38,38,0.14)" stroke="#dc2626" stroke-width="1.4"/>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------- html
_CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
 margin:0;background:#f8fafc;color:#0f172a}
.wrap{max-width:1020px;margin:0 auto;padding:36px 22px 48px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:18px;margin:34px 0 10px}
.sub{color:#475569;font-size:14px;margin:2px 0 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:12px;margin-top:18px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px}
.card .k{font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:21px;font-weight:650;margin-top:6px;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;background:#fff;font-size:13px;margin-top:8px}
th,td{border:1px solid #e2e8f0;padding:6px 10px;text-align:right;font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
thead th{background:#f1f5f9;font-weight:600}
.pos{color:#15803d}.neg{color:#b91c1c}
details{margin-top:8px} summary{cursor:pointer;font-size:14px;color:#334155}
.foot{margin-top:40px;font-size:12px;color:#64748b;border-top:1px solid #e2e8f0;padding-top:14px;line-height:1.6}
.tick{font-size:11px;fill:#64748b;font-family:inherit}
.note{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:10px 14px;font-size:13px;color:#713f12;margin-top:10px}
"""


def _metric_rows(m: dict, bench: bool) -> list[tuple[str, str, str]]:
    """(metric, strategy display, benchmark display) rows."""
    rows = [
        ("Total return", f"<span class='{_cls(m['total_return'])}'>{fmt_pct(m['total_return'])}</span>", fmt_pct(m.get("bench_total"))),
        ("Annualized return (CAGR)", f"<span class='{_cls(m['cagr'])}'>{fmt_pct(m['cagr'])}</span>", fmt_pct(m.get("bench_cagr"))),
        ("Excess CAGR vs benchmark", f"<span class='{_cls(m.get('excess_cagr'))}'>{fmt_pct(m.get('excess_cagr'))}</span>", "-"),
        ("Annualized volatility", fmt_pct(m["ann_vol"]), "-"),
        ("Sharpe ratio (rf=0)", fmt_num(m["sharpe"]), "-"),
        ("Max drawdown", f"<span class='neg'>{fmt_pct(m['max_drawdown'])}</span>", fmt_pct(m.get("bench_max_drawdown"))),
        ("Max DD duration (sessions)", str(m["max_dd_sessions"]), "-"),
        ("Calmar ratio", fmt_num(m["calmar"]), "-"),
        ("Average exposure", fmt_pct(m["avg_exposure"]), "-"),
        ("Round-trip trades", str(m["round_trips"]), "-"),
        ("Win rate (round trips)", fmt_pct(m["win_rate"]) if m["win_rate"] is not None else "n/a (never flat)", "-"),
        ("Profit factor", fmt_num(m["profit_factor"]) if m["profit_factor"] is not None else "n/a", "-"),
        ("Fills (buy/sell)", f"{m['n_buys']} / {m['n_sells']}", "-"),
        ("Traded turnover", fmt_money(m["turnover_total"]) + " CNY", "-"),
        ("Commissions paid", fmt_money(m["commission_total"]) + " CNY", "-"),
        ("Stamp tax paid", fmt_money(m["stamp_total"]) + " CNY", "-"),
        ("Stop-loss exits", str(m["stop_loss_exits"]), "-"),
        ("Drawdown liquidations", str(m["drawdown_flats"]), "-"),
    ]
    if not bench:
        rows = [(a, b, "-") for a, b, _ in rows]
    return rows


def _metrics_table(m: dict, bench: bool = True) -> str:
    head = "<tr><th>Metric</th><th>Strategy</th><th>CSI 300</th></tr>"
    body = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in _metric_rows(m, bench))
    return f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"


def _split_table(m: dict) -> str:
    rows = [
        ("Period", f"{fmt_date(m['start'])} .. {fmt_date(m['end'])}", f"{m['sessions']} sessions"),
        ("Total return", fmt_pct(m["total_return"]), ""),
        ("CAGR", fmt_pct(m["cagr"]), ""),
        ("Max drawdown", fmt_pct(m["max_drawdown"]), ""),
        ("Sharpe", fmt_num(m["sharpe"]), ""),
        ("Win rate", fmt_pct(m["win_rate"]) if m["win_rate"] is not None else "n/a", ""),
        ("Excess CAGR", fmt_pct(m.get("excess_cagr")), ""),
    ]
    body = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)
    return f"<table><thead><tr><th></th><th>Value</th><th></th></tr></thead><tbody>{body}</tbody></table>"


def _trades_table(trades) -> str:
    head = ("<tr><th>Date</th><th>Symbol</th><th>Side</th><th>Shares</th><th>Price</th>"
            "<th>Turnover</th><th>Fee</th><th>Realized PnL</th><th>Cash after</th><th>Reason</th></tr>")
    rows = []
    for t in trades:
        pnl = fmt_money(t.realized_pnl) if t.side == "SELL" else "-"
        cls = _cls(t.realized_pnl) if t.side == "SELL" else ""
        rows.append(
            f"<tr><td>{fmt_date(t.date)}</td><td>{html.escape(t.symbol)}</td><td>{t.side}</td>"
            f"<td>{t.shares}</td><td>{fmt_num(t.price)}</td><td>{fmt_money(t.turnover)}</td>"
            f"<td>{fmt_money(t.fee)}</td><td class='{cls}'>{pnl}</td>"
            f"<td>{fmt_money(t.cash_after)}</td><td>{t.reason}</td></tr>"
        )
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _symbol_table(result) -> str:
    trips_by_symbol: dict[str, list] = {}
    for trip in result.round_trips:
        trips_by_symbol.setdefault(trip.symbol, []).append(trip)
    head = "<tr><th>Symbol</th><th>Round trips</th><th>Wins</th><th>Win rate</th><th>Total PnL (CNY)</th><th>Open shares</th></tr>"
    rows = []
    for symbol in sorted(set(list(trips_by_symbol) + list(result.final_positions))):
        trips = trips_by_symbol.get(symbol, [])
        wins = sum(1 for t in trips if t.pnl > 0)
        pnl = sum(t.pnl for t in trips)
        shares, _ = result.final_positions.get(symbol, (0, 0.0))
        wr = fmt_pct(wins / len(trips)) if trips else "-"
        rows.append(
            f"<tr><td>{html.escape(symbol)}</td><td>{len(trips)}</td><td>{wins}</td>"
            f"<td>{wr}</td><td class='{_cls(pnl)}'>{fmt_money(pnl)}</td><td>{shares}</td></tr>"
        )
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def render_report(
    result,
    metrics: dict,
    out_path,
    split_metrics: dict | None = None,
    extra_meta: dict | None = None,
) -> pathlib.Path:
    """Render one self-contained HTML report; returns the written path."""
    meta = dict(result.meta)
    meta.update(extra_meta or {})
    eq = result.equity
    dates = list(eq.index)

    series_map = {"Strategy": (0, [float(v) for v in eq / eq.iloc[0] * 100.0])}
    if result.benchmark is not None:
        b = result.benchmark.dropna()
        if len(b) >= 2:
            start_idx = 0
            for i, d in enumerate(dates):
                if d in b.index:
                    start_idx = i
                    break
            series_map["CSI 300"] = (
                start_idx,
                [float(v) for v in (b / b.iloc[0] * 100.0).reindex(dates).ffill()],
            )
    dd_values = [float(v) for v in (eq / eq.cummax() - 1.0)]

    data_source = str(meta.get("data_source", "unknown"))
    sample_note = ""
    if data_source != "akshare":
        sample_note = (
            "<div class='note'><b>Demo data:</b> this run used the built-in synthetic sample "
            "series (akshare unavailable). Numbers demonstrate the pipeline end-to-end and are "
            "NOT market results.</div>"
        )

    cards = [
        ("Total return", f"<span class='{_cls(metrics['total_return'])}'>{fmt_pct(metrics['total_return'])}</span>"),
        ("CAGR", f"<span class='{_cls(metrics['cagr'])}'>{fmt_pct(metrics['cagr'])}</span>"),
        ("Max drawdown", f"<span class='neg'>{fmt_pct(metrics['max_drawdown'])}</span>"),
        ("Sharpe", fmt_num(metrics["sharpe"])),
        ("Win rate", fmt_pct(metrics["win_rate"]) if metrics["win_rate"] is not None else "n/a"),
        ("Excess CAGR", f"<span class='{_cls(metrics.get('excess_cagr'))}'>{fmt_pct(metrics.get('excess_cagr'))}</span>"),
    ]
    cards_html = "".join(f"<div class='card'><div class='k'>{k}</div><div class='v'>{v}</div></div>" for k, v in cards)

    cfg_rows = [
        ("Strategy", html.escape(str(meta.get("strategy", "")))),
        ("Symbols", ", ".join(html.escape(s) for s in meta.get("symbols", []))),
        ("Period", f"{fmt_date(eq.index[0])} .. {fmt_date(eq.index[-1])} ({len(eq)} sessions)"),
        ("Initial capital", fmt_money(meta.get("capital")) + " CNY"),
        ("Data source", data_source),
        ("Cost model", "commission 0.025%/side (min 5 CNY), stamp tax 0.1% sell, slippage 0.1%/side"),
        ("Execution", "close-of-day signals filled at next open, 100-share lots, T+1"),
        ("Risk rules", "20% single-name cap, -8% stop loss, -15% drawdown -> flat + cooldown"),
    ]
    if meta.get("split"):
        cfg_rows.append(("IS/OOS split", f"{fmt_pct(meta['split'], 0)} in-sample / {fmt_pct(1 - meta['split'], 0)} out-of-sample"))
    cfg_html = "<table><tbody>" + "".join(
        f"<tr><td style='width:220px'>{a}</td><td style='text-align:left'>{b}</td></tr>" for a, b in cfg_rows
    ) + "</tbody></table>"

    split_html = ""
    if split_metrics:
        is_m = split_metrics.get("in_sample")
        oos_m = split_metrics.get("out_of_sample")
        if is_m and oos_m:
            split_html = (
                "<h2>In-sample / out-of-sample check</h2>"
                "<p class='sub'>Independent engine runs on each slice (fresh capital). "
                "Diverging out-of-sample numbers are a normal overfitting warning, not a bug.</p>"
                "<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px'>"
                f"<div><h3 style='margin:6px 0'>In-sample</h3>{_split_table(is_m)}</div>"
                f"<div><h3 style='margin:6px 0'>Out-of-sample</h3>{_split_table(oos_m)}</div>"
                "</div>"
            )

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuantPilot report - {html.escape(str(meta.get('strategy', '')))}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<h1>QuantPilot backtest report</h1>
<p class="sub">Strategy <b>{html.escape(str(meta.get('strategy', '')))}</b> -
{fmt_date(eq.index[0])} to {fmt_date(eq.index[-1])} - net of all frictions</p>
{sample_note}
<div class="cards">{cards_html}</div>

<h2>Run configuration</h2>{cfg_html}

<h2>Equity curve vs CSI 300</h2>
{_svg_line_chart(series_map, dates, {"Strategy": "#2563eb", "CSI 300": "#94a3b8"},
                 y_fmt=lambda v: f"{v:,.0f}", title="indexed, start = 100")}
<p class="sub">Strategy equity is marked daily after costs; benchmark is the raw CSI 300 index
(price return, no costs).</p>

<h2>Drawdown</h2>
{_svg_drawdown_chart(dd_values, dates)}

<h2>Metrics</h2>
{_metrics_table(metrics, bench=result.benchmark is not None)}
{split_html}

<h2>Risk events</h2>
<p class="sub">Stop-loss exits: {metrics['stop_loss_exits']} -
drawdown liquidations: {metrics['drawdown_flats']} -
fees+tax paid: {fmt_money(metrics['fees_total'])} CNY -
slippage estimate: {fmt_money(result.fee_summary.get('slippage_est', float('nan')))} CNY</p>

<h2>Per-symbol summary</h2>
{_symbol_table(result)}

<h2>Trade log</h2>
<details><summary>Show all {len(result.trades)} fills</summary>{_trades_table(result.trades)}</details>

<div class="foot">Generated by QuantPilot v{VERSION} on {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} -
Metrics are historical simulations net of estimated costs and do not predict future results.
This report is for education and engineering review only; it is not investment advice.</div>
</div></body></html>"""

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path
