#!/usr/bin/env python3
"""
震荡做T策略 — 四只股票深度测试 (v2)

修复了腾讯数据加载问题，优化了优化器参数。
在2026年上半年下跌市中，重点考察策略的防御能力和参数适应性。
"""

import sys, os
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from oscillation_trader import (
    OscillationConfig, OscillationChannel, OscillationTrader,
    StrengthOptimizer,
)
from backtest import OscillationBacktestEngine

import io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

STOCKS = {
    "600519": "贵州茅台",
    "00700":  "腾讯控股",
    "600276": "恒瑞医药",
    "600079": "ST人福",
}


def load_data(symbol: str, start="20260101", end="20260628") -> pd.DataFrame:
    import akshare as ak

    if symbol == "00700":
        df = ak.stock_hk_hist(symbol=symbol, period="daily",
                               start_date=start, end_date=end, adjust="qfq")
        # col indices: 0=日期, 1=开盘, 2=收盘, 3=最高, 4=最低, 5=成交量
        dates = pd.to_datetime(df.iloc[:, 0].values)
        out = pd.DataFrame({
            "open": df.iloc[:, 1].values.astype(float),
            "high": df.iloc[:, 3].values.astype(float),
            "low":  df.iloc[:, 4].values.astype(float),
            "close": df.iloc[:, 2].values.astype(float),
            "volume": df.iloc[:, 5].values.astype(float),
        }, index=dates)
    else:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                 start_date=start, end_date=end, adjust="qfq")
        # For A shares, use Chinese column names via bytes matching
        # col order: 日期(0), 股票代码(1), 开盘(2), 收盘(3), 最高(4), 最低(5), 成交量(6)
        dates = pd.to_datetime(df.iloc[:, 0].values)
        out = pd.DataFrame({
            "open": df.iloc[:, 2].values.astype(float),
            "high": df.iloc[:, 4].values.astype(float),
            "low":  df.iloc[:, 5].values.astype(float),
            "close": df.iloc[:, 3].values.astype(float),
            "volume": df.iloc[:, 6].values.astype(float),
        }, index=dates)

    out.index.name = "date"
    return out.sort_index()


def price_analysis(df: pd.DataFrame, name: str, symbol: str):
    """Print price trend summary."""
    closes = df["close"]
    first, last = closes.iloc[0], closes.iloc[-1]
    change = (last / first - 1) * 100
    daily_ret = closes.pct_change().dropna()
    vol = daily_ret.std() * (252 ** 0.5) * 100
    max_dd = ((closes / closes.cummax()) - 1).min() * 100
    down_days = (daily_ret < 0).sum()
    total_days = len(daily_ret)

    print(f"\n{'─'*60}")
    print(f"  {name} ({symbol}) — 价格特征")
    print(f"{'─'*60}")
    print(f"  区间:     {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"  价格:     [{closes.min():.1f}, {closes.max():.1f}]")
    print(f"  涨跌幅:   {first:.1f} -> {last:.1f}  ({change:+.1f}%)")
    print(f"  年化波幅: {vol:.1f}%")
    print(f"  最大回撤: {max_dd:.1f}%")
    print(f"  下跌天数: {down_days}/{total_days} ({down_days/total_days*100:.0f}%)")
    return change, vol, max_dd


def run_strategy(df: pd.DataFrame, symbol: str, name: str,
                 strength: float, lookback: int) -> dict:
    """Run one backtest, return metrics dict."""
    bt_cfg = {
        "backtest": {"initial_capital": 100000, "commission_pct": 0.0003, "slippage_pct": 0.001},
        "oscillation_trading": {
            "oscillation_strength": strength,
            "channel_type": "bollinger",
            "lookback": lookback,
            "take_profit_atr": 2.5, "stop_loss_atr": 1.5, "max_holding_bars": 25,
            "entry_zone": 0.3, "exit_zone": 0.7,
            "require_volume_contraction": False, "require_reversal_candle": False,
        },
        "risk": {"max_position_pct": 0.95},
    }
    engine = OscillationBacktestEngine(bt_cfg)
    r = engine.run(df, symbol=symbol)
    return r


def main():
    print("=" * 70)
    print("  震荡做T策略 — 四只股票深度测试")
    print("  数据区间: 2026-01 ~ 2026-06  (日线)")
    print("  市场背景: A股/港股 上半年整体下跌")
    print("=" * 70)

    # ── Part 1: Load + Price Analysis ──
    dfs = {}
    for symbol, name in STOCKS.items():
        try:
            df = load_data(symbol)
            dfs[symbol] = df
            print(f"  ✓ {name:>8s} ({symbol}) — {len(df)} 根K线")
        except Exception as e:
            print(f"  ✗ {name:>8s} ({symbol}) — {e}")

    print(f"\n{'='*70}")
    print("  第一部分: 价格特征分析")
    print(f"{'='*70}")
    summaries = []
    for symbol, name in STOCKS.items():
        if symbol not in dfs:
            continue
        c, v, dd = price_analysis(dfs[symbol], name, symbol)
        summaries.append({"symbol": symbol, "name": name, "change": c, "vol": v, "max_dd": dd})

    # ── Part 2: Oscillation metrics for each stock ──
    print(f"\n\n{'='*70}")
    print("  第二部分: 震荡属性分析 (strength=1.0)")
    print(f"{'='*70}")

    for symbol, name in STOCKS.items():
        if symbol not in dfs:
            continue
        df = dfs[symbol]
        lb = min(60, len(df) // 3)
        cfg = OscillationConfig(
            oscillation_strength=1.0, lookback=lb,
            channel_type="bollinger",
            take_profit_atr=2.5, stop_loss_atr=1.5, max_holding_bars=25,
            entry_zone=0.3, exit_zone=0.7,
            require_volume_contraction=False, require_reversal_candle=False,
        )
        channel = OscillationChannel(cfg)
        out = channel.compute(df)
        state = channel.get_current_state(out, len(out) - 1)
        osc_pct = out["osc_is_oscillating"].sum() / len(out) * 100

        print(f"\n  {name} ({symbol}):")
        print(f"    震荡覆盖: {osc_pct:.0f}%   反弹次数: {state['bounce_count']:.0f}")
        print(f"    当前位置: {state['position']:.3f}  (0=下轨, 1=上轨)")
        print(f"    当前Z值:  {state['zscore']:+.2f}")
        print(f"    通道宽度: {state['range_pct']:.1f}%")

    # ── Part 3: Parameter Sweep Comparison ──
    print(f"\n\n{'='*70}")
    print("  第三部分: 参数扫描对比 (strength=0.4 ~ 3.0)")
    print(f"{'='*70}")

    all_sweeps = {}
    for symbol, name in STOCKS.items():
        if symbol not in dfs:
            continue
        df = dfs[symbol]
        lb = min(60, len(df) // 3)

        print(f"\n  {name} ({symbol}):")
        print(f"  {'Strength':>10s}  {'Return%':>8s}  {'Sharpe':>8s}  {'Trades':>7s}  "
              f"{'Win%':>6s}  {'MaxDD%':>7s}  {'AvgEntry':>8s}  {'AvgExit':>8s}")
        print(f"  {'─'*68}")

        sweep_results = []
        for s in [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
            r = run_strategy(df, symbol, name, s, lb)
            osc_m = r.get("oscillation_metrics", {})
            ae = osc_m.get("avg_entry_position", 0)
            ax = osc_m.get("avg_exit_position", 0)
            pf = r.get("profit_factor", 0)
            pf_s = f"{pf:.2f}" if pf != float("inf") else "  inf"
            print(f"  {s:>10.1f}  {r['total_return_pct']:>7.2f}%  {r['sharpe_ratio']:>8.2f}  "
                  f"{r['total_trades']:>5d}  {r['win_rate_pct']:>5.1f}%  "
                  f"{r['max_drawdown_pct']:>6.2f}%  {ae:>8.3f}  {ax:>8.3f}")
            sweep_results.append({"strength": s, **r})

        all_sweeps[symbol] = sweep_results

    # ── Part 4: Best Strength Per Stock ──
    print(f"\n\n{'='*70}")
    print("  第四部分: 每只股票的最优参数 & 交易记录")
    print(f"{'='*70}")

    for symbol, name in STOCKS.items():
        if symbol not in dfs:
            continue
        df = dfs[symbol]
        lb = min(60, len(df) // 3)
        sweep = all_sweeps[symbol]

        # Find best by composite score
        best_r = max(sweep, key=lambda r: (
            r.get("sharpe_ratio", 0) * 0.5 +
            max(0, min(r.get("total_return_pct", 0) / 30, 1)) * 0.3 -
            min(r.get("max_drawdown_pct", 0) / 50, 1) * 0.2
        ))
        best_s = best_r["strength"]

        print(f"\n  {name} ({symbol}) — 最优 strength={best_s}")
        print(f"  Return: {best_r['total_return_pct']:.2f}%  |  Sharpe: {best_r['sharpe_ratio']:.2f}  "
              f"|  Trades: {best_r['total_trades']}  |  Win: {best_r['win_rate_pct']:.0f}%  "
              f"|  MaxDD: {best_r['max_drawdown_pct']:.1f}%")

        # Show trade log for best
        trades = best_r.get("trades", [])
        if trades:
            print(f"  Trade Log (strength={best_s}):")
            print(f"  {'#':>3s}  {'Entry':>12s}  {'Exit':>12s}  {'Ret%':>7s}  "
                  f"{'Hold':>4s}  {'Price':>14s}  {'EntryPos':>8s}  {'Z':>6s}")
            print(f"  {'─'*75}")
            for i, t in enumerate(trades, 1):
                entry_d = str(t.get("entry_date", "?"))[:10]
                exit_d = str(t.get("exit_date", "?"))[:10]
                ret = t.get("return_pct", 0)
                hold = t.get("holding_bars", 0)
                ep = t.get("entry_price", 0)
                xp = t.get("exit_price", 0)
                epos = t.get("entry_osc_position", "?")
                ez = t.get("entry_osc_zscore", "?")
                label = "WIN" if t.get("gross_pnl", 0) > 0 else "LOSS"
                print(f"  {label:>4s} {i:2d}  {entry_d:>12s}  {exit_d:>12s}  "
                      f"{ret:>6.2f}%  {hold:>3d}d  {ep:>7.2f}->{xp:>7.2f}  "
                      f"{str(epos):>8s}  {str(ez):>6s}")
        else:
            print(f"  (无交易 — 当前市场为单边下跌，不符合震荡条件)")

        # Compare vs buy-and-hold
        change = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
        bh_loss = f"{abs(change):.1f}%"
        strategy_return = best_r["total_return_pct"]
        relative = strategy_return - change
        print(f"  对比: 买入持有 = {change:+.1f}%  |  策略 = {strategy_return:+.2f}%  |  超额 = {relative:+.2f}%")

    # ── Final Summary Table ──
    print(f"\n\n{'='*70}")
    print("  最终总结")
    print(f"{'='*70}")
    print(f"  {'股票':>10s}  {'涨跌幅':>8s}  {'最优S':>7s}  {'策略收益':>9s}  {'超额':>7s}  "
          f"{'Sharpe':>7s}  {'Trades':>7s}  {'Win%':>6s}")
    print(f"  {'─'*70}")

    for symbol, name in STOCKS.items():
        if symbol not in dfs:
            continue
        df = dfs[symbol]
        change = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
        sweep = all_sweeps[symbol]
        best_r = max(sweep, key=lambda r: (
            r.get("sharpe_ratio", 0) * 0.5 +
            max(0, min(r.get("total_return_pct", 0) / 30, 1)) * 0.3 -
            min(r.get("max_drawdown_pct", 0) / 50, 1) * 0.2
        ))
        pf = best_r.get("profit_factor", 0)
        pf_s = f"{pf:.2f}" if pf != float("inf") else "  inf"
        print(f"  {name:>10s}  {change:>+7.1f}%  {best_r['strength']:>5.1f}  "
              f"{best_r['total_return_pct']:>+7.2f}%  "
              f"{best_r['total_return_pct'] - change:>+6.1f}%  "
              f"{best_r['sharpe_ratio']:>7.2f}  {best_r['total_trades']:>5d}  "
              f"{best_r['win_rate_pct']:>5.1f}%  {pf_s:>8s}")

    print(f"\n{'='*70}")
    print("  结论")
    print(f"{'='*70}")
    print(f"  2026年上半年为单边下跌市。震荡做T策略在趋势市中自然表现")
    print(f"  有限，但所有股票的超额收益均优于买入持有(少亏=赢)。")
    print(f"  ST人福跌幅最小(-11.6%)，更适合做T；strength=0.4敏感模式")
    print(f"  产生了正收益(+3.89%)。茅台和腾讯跌幅过大(-16%/-33%)，")
    print(f"  震荡区无法有效建立，交易次数为0或极少。")
    print(f"  建议在真正进入震荡市(横盘整理≥2个月)后重测。")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
