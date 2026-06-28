"""Utility functions: charting, summary tables, exports."""

import os
import json

import numpy as np
import pandas as pd


def equity_curve_to_df(results):
    """Convert equity curve from results dict to DataFrame."""
    records = results.get("equity_curve", [])
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


def compute_drawdowns(equity_df):
    """Compute drawdown series from equity curve."""
    eq = equity_df["equity"]
    rolling_max = eq.expanding().max()
    dd = (eq - rolling_max) / rolling_max * 100
    return dd


def generate_report(results, output_dir="reports"):
    """Generate a text report file from backtest results."""
    os.makedirs(output_dir, exist_ok=True)
    symbol = results.get("symbol", "unknown")
    path = os.path.join(output_dir, f"report_{symbol}.txt")

    lines = []
    lines.append("=" * 60)
    lines.append(f"  Phase Trading System Report")
    lines.append(f"  Symbol: {symbol}")
    lines.append(f"  Period: {results.get('date_range', 'N/A')}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("  PERFORMANCE METRICS")
    lines.append(f"    Total Return:      {results.get('total_return_pct', 0):>8.2f}%")
    lines.append(f"    Annual Return:     {results.get('annual_return_pct', 0):>8.2f}%")
    lines.append(f"    Sharpe Ratio:      {results.get('sharpe_ratio', 0):>8.2f}")
    lines.append(f"    Max Drawdown:      {results.get('max_drawdown_pct', 0):>8.2f}%")
    lines.append(f"    Final Capital:     {results.get('final_capital', 0):>10.2f}")
    lines.append("")
    lines.append("  TRADE STATISTICS")
    lines.append(f"    Total Trades:      {results.get('total_trades', 0):>8d}")
    lines.append(f"    Win Rate:          {results.get('win_rate_pct', 0):>8.1f}%")
    lines.append(f"    Profit Factor:     {results.get('profit_factor', 0):>8.2f}")
    lines.append(f"    Avg Profit:        {results.get('avg_profit', 0):>8.2f}")
    lines.append(f"    Avg Loss:          {results.get('avg_loss', 0):>8.2f}")
    lines.append(f"    Expectancy:        {results.get('expectancy', 0):>8.2f}")
    lines.append("")
    lines.append("  PHASE DISTRIBUTION")
    for phase, count in results.get("phase_distribution", {}).items():
        pct = count / sum(results.get("phase_distribution", {}).values()) * 100
        lines.append(f"    {phase}: {count:>6d} bars ({pct:.1f}%)")
    lines.append("")

    # Trade list
    trades = results.get("trades", [])
    if trades:
        lines.append("  TRADE LOG")
        lines.append(f"    {'#':>3s} {'Entry':>12s} {'Exit':>12s} {'Return':>8s} {'Phase':>6s} {'PnL':>10s}")
        lines.append(f"    {'-'*55}")
        for i, t in enumerate(trades[-20:], 1):
            entry = str(t.get("entry_date", "?")[:10])
            exit_d = str(t.get("exit_date", "?")[:10])
            ret = t.get("return_pct", 0)
            phase = t.get("phase_name", "?")
            pnl = t.get("gross_pnl", 0)
            lines.append(f"    {i:3d} {entry:>12s} {exit_d:>12s} {ret:>7.2f}% {phase:>6s} {pnl:>9.2f}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved: {path}")
    return path


def print_phase_summary(phase_history, phase_detector):
    """Print a summary of phase distribution."""
    total = len(phase_history)
    if total == 0:
        return
    counts = pd.Series(phase_history).value_counts().sort_index()
    print("\nPhase Summary:")
    for phase_val, count in counts.items():
        name = phase_detector.get_phase_name(phase_val)
        pct = count / total * 100
        print(f"  {name}: {count} bars ({pct:.1f}%)")
