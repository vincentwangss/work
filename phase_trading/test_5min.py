#!/usr/bin/env python3
"""
Baostock 5分钟K线 — 震荡做T策略全量测试
数据区间: 2026-01-05 ~ 2026-06-26 (约6个月，5472根K线 / 股票)
"""
import sys, os
sys.path.insert(0, '.')

import io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import baostock as bs
import pandas as pd
import numpy as np
from oscillation_trader import OscillationConfig, OscillationChannel
from backtest import OscillationBacktestEngine
from data_loader import DataLoader

STOCKS = [
    ("600519", "贵州茅台"),
    ("600276", "恒瑞医药"),
    ("600079", "ST人福"),
    ("601318", "中国平安"),
    ("300308", "中际旭创"),
]

FREQ = "5"

def run_bt(df, strength, lookback, tprofit=2.0, sloss=1.5, max_hold=30):
    cfg = {
        "backtest": {"initial_capital": 100000, "commission_pct": 0.0003, "slippage_pct": 0.001},
        "oscillation_trading": {
            "oscillation_strength": strength, "channel_type": "bollinger", "lookback": lookback,
            "take_profit_atr": tprofit, "stop_loss_atr": sloss, "max_holding_bars": max_hold,
            "entry_zone": 0.3, "exit_zone": 0.7,
            "require_volume_contraction": False, "require_reversal_candle": False,
            "require_oscillation_confirm": False,
        },
        "risk": {"max_position_pct": 0.95},
    }
    return OscillationBacktestEngine(cfg).run(df, symbol="TEST")

# Load all
print("=" * 65)
print(f"  Baostock 5分钟K线 (2026-01-05 ~ 2026-06-26)")
print("=" * 65)

config = {"data": {"source": "baostock", "adjust": "qfq"}}
loader = DataLoader(config)
dfs = {}
for sym, name in STOCKS:
    dfs[sym] = loader.load_minute(sym, "20260101", "20260628", FREQ)
    c = dfs[sym]["close"]
    f, l = c.iloc[0], c.iloc[-1]
    chg = (l / f - 1) * 100
    rets = c.pct_change().dropna()
    vol = rets.std() * np.sqrt(252 * 48) * 100  # ~48 bars/day for 5min
    dd = ((c / c.cummax()) - 1).min() * 100
    dn = (rets < 0).sum()
    print(f"  {name:>8s}: {len(dfs[sym]):5d}b  {f:.1f}->{l:.1f}  ({chg:+.1f}%)  "
          f"波动{vol:.0f}%  DD{dd:.1f}%  下跌{dn}/{len(rets)}({dn/len(rets)*100:.0f}%)")

# ── Per-stock analysis ──
print("\n" + "=" * 65)
print("  参数扫描 (S=0.4~2.0, TP=2/3/4, SL=1.5/2.0, MH=30)")
print("=" * 65)

best_all = {}
for sym, name in STOCKS:
    df = dfs[sym]
    n_bars = len(df)
    lb = min(120, max(40, n_bars // 45))  # ~2 trading days of 5-min bars
    change = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

    print(f"\n  ══ {name} ({sym})  5min  lookback={lb} ══")

    # Oscillation metrics
    cfg = OscillationConfig(oscillation_strength=1.0, lookback=lb)
    channel = OscillationChannel(cfg)
    out = channel.compute(df)
    state = channel.get_current_state(out, len(out) - 1)
    osc_pct = out["osc_is_oscillating"].sum() / len(out) * 100
    print(f"  震荡覆盖: {osc_pct:.0f}%  |  "
          f"bounce={state['bounce_count']:.0f}  |  "
          f"pos={state['position']:.3f}  |  "
          f"z={state['zscore']:+.2f}  |  "
          f"range={state['range_pct']:.1f}%")

    # Sweep
    print(f"  {'S':>4s}  {'TP/SL':>6s}  {'Return%':>8s}  {'Sharpe':>7s}  "
          f"{'Trades':>6s}  {'Win%':>5s}  {'MaxDD':>6s}  {'AvgEntry':>8s}  {'AvgExit':>8s}")
    print(f"  {'─' * 70}")

    best_r = None
    for s in [0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
        for tp, sl in [(2.0, 1.5), (3.0, 2.0), (4.0, 2.5)]:
            r = run_bt(df, s, lb, tp, sl, 30)
            oe = r.get("oscillation_metrics", {})
            ae = oe.get("avg_entry_position", 0)
            ax = oe.get("avg_exit_position", 0)
            print(f"  {s:>4.1f}  {tp:.0f}/{sl:.0f}  "
                  f"{r['total_return_pct']:>7.2f}%  {r['sharpe_ratio']:>7.2f}  "
                  f"{r['total_trades']:>5d}  {r['win_rate_pct']:>4.0f}%  "
                  f"{r['max_drawdown_pct']:>5.2f}%  {ae:>8.3f}  {ax:>8.3f}")
            if r["total_trades"] >= 5 and (best_r is None or r["sharpe_ratio"] > best_r["sharpe_ratio"]):
                r["s"] = s; r["tp"] = tp; r["sl"] = sl
                best_r = r

    if best_r:
        best_all[sym] = best_r
        print(f"\n  >> 最优: S={best_r['s']}  TP={best_r['tp']:.0f}  SL={best_r['sl']:.0f}")
        print(f"  >> 收益={best_r['total_return_pct']:+.2f}%  "
              f"Sharpe={best_r['sharpe_ratio']:.2f}  "
              f"Trades={best_r['total_trades']}  Win={best_r['win_rate_pct']:.0f}%  "
              f"MaxDD={best_r['max_drawdown_pct']:.1f}%")
        print(f"  >> 超额 vs 买入持有({change:+.1f}%): {best_r['total_return_pct']-change:+.1f}%")
        trades = best_r.get("trades", [])
        for t in trades[-8:]:
            ed = str(t.get("entry_date", "?"))[:16]
            xd = str(t.get("exit_date", "?"))[:16]
            ret = t.get("return_pct", 0)
            hold = t.get("holding_bars", 0)
            ep = t.get("entry_price", 0)
            xp = t.get("exit_price", 0)
            epos = t.get("entry_osc_position", "?")
            xpos = t.get("exit_osc_position", "?")
            lbl = "WIN" if t.get("gross_pnl", 0) > 0 else "LOSS"
            print(f"    {lbl:>4s} {ed} -> {xd}  {ret:+.2f}%  ({hold:2d}b)  "
                  f"{ep:.1f}->{xp:.1f}  pos={epos}->{xpos}")
    else:
        print(f"\n  >> 无合资格交易(需>=5笔)")

# ── Summary ──
print(f"\n\n{'='*75}")
print(f"  5分钟 K线 — 最终汇总 (2026-01~2026-06)")
print(f"{'='*75}")
print(f"  {'股票':>8s}  {'区间%':>8s}  {'配置':>14s}  {'收益%':>8s}  "
      f"{'Sharpe':>7s}  {'Trades':>6s}  {'Win%':>5s}  {'超额%':>7s}")
print(f"  {'─'*75}")

for sym, name in STOCKS:
    df = dfs[sym]
    change = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    br = best_all.get(sym)
    if br:
        c = f"S={br['s']} TP{br['tp']:.0f} SL{br['sl']:.0f}"
        ex = br["total_return_pct"] - change
        print(f"  {name:>8s}  {change:>+7.1f}%  {c:>14s}  "
              f"{br['total_return_pct']:>+7.2f}%  {br['sharpe_ratio']:>7.2f}  "
              f"{br['total_trades']:>5d}  {br['win_rate_pct']:>4.0f}%  {ex:>+6.1f}%")
    else:
        print(f"  {name:>8s}  {change:>+7.1f}%  {'(无≥5笔)':>14s}  "
              f"{'+0.00%':>8s}  {'0.00':>7s}  {'0':>5s}  {'-':>5s}  "
              f"{0-change:>+6.1f}%")

print()
