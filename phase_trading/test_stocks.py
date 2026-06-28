#!/usr/bin/env python3
"""
测试震荡做T策略在四只股票上的表现：
  600519  — 茅台 (贵州茅台)
  00700   — 腾讯控股 (港股)
  600276  — 恒瑞医药
  600079  — ST人福 (人福医药)

测试内容：
  1. 各股票震荡属性分析
  2. 默认 strength=1.0 回测
  3. 滚动优化 strength 回测
  4. 各 strength 参数对比扫描
  5. 交易记录分析
"""

import sys, os
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from oscillation_trader import (
    OscillationConfig, OscillationChannel, OscillationTrader,
    StrengthOptimizer, find_oscillation_stocks,
)
from backtest import OscillationBacktestEngine

# ── fix Windows encoding ──
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
    """Load data from akshare, return standard OHLCV DataFrame."""
    import akshare as ak

    if symbol == "00700":
        # HK stock
        df = ak.stock_hk_hist(symbol=symbol, period="daily",
                               start_date=start, end_date=end, adjust="qfq")
        col_map = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "成交量",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
    else:
        # A-share
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                 start_date=start, end_date=end, adjust="qfq")
        col_map = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

    return df


def run_single_test(df: pd.DataFrame, symbol: str, name: str, strength: float,
                    do_optimize: bool = False):
    """Run the full test on one stock with a given strength (or optimize)."""
    print(f"\n{'='*70}")
    print(f"  {name} ({symbol})")
    print(f"{'='*70}")

    # ── Config ──
    cfg = OscillationConfig(
        oscillation_strength=strength,
        lookback=min(60, len(df) // 3),
        channel_type="bollinger",
        take_profit_atr=2.5,
        stop_loss_atr=1.5,
        max_holding_bars=25,
        entry_zone=0.3,
        exit_zone=0.7,
        require_volume_contraction=False,
        require_reversal_candle=False,
    )

    print(f"  Strength: {cfg.oscillation_strength}")
    print(f"  Lookback: {cfg.lookback}")
    print(f"  Channel:  {cfg.channel_type} (effective std={cfg.effective_std:.2f})")
    print(f"  Data:     {df.index[0].date()} ~ {df.index[-1].date()}  ({len(df)} bars)")
    print(f"  Price:    [{df['low'].min():.2f}, {df['high'].max():.2f}]")

    # ── Optimize if requested ──
    if do_optimize:
        print(f"\n  ── Rolling Optimization ──")
        optimizer = StrengthOptimizer(cfg)
        best_s, opt_results = optimizer.optimize(
            df, validation_bars=min(80, len(df) // 2), metric="composite",
        )
        if opt_results:
            print(optimizer.summary(opt_results, best_s))
            print(f"    >> Best: strength={best_s}")
            cfg.oscillation_strength = best_s
        else:
            print(f"    (no valid candidates)")

    # ── Compute channel ──
    channel = OscillationChannel(cfg)
    df_out = channel.compute(df)

    # ── Current state ──
    state = channel.get_current_state(df_out, len(df_out) - 1)
    print(f"\n  Current State:")
    print(f"    Oscillating:    {state['is_oscillating']}")
    print(f"    Position:       {state['position']:.3f} (0=lower, 1=upper)")
    print(f"    Z-score:        {state['zscore']:.2f}")
    print(f"    Range:          {state['range_pct']:.1f}%")
    print(f"    Bounces:        {state['bounce_count']:.0f}")

    # Oscillation coverage
    osc_pct = df_out["osc_is_oscillating"].sum() / len(df_out) * 100
    print(f"    OSC coverage:   {osc_pct:.1f}% of bars")

    # ── Backtest ──
    bt_cfg = {
        "backtest": {"initial_capital": 100000, "commission_pct": 0.0003, "slippage_pct": 0.001},
        "oscillation_trading": {
            "oscillation_strength": cfg.oscillation_strength,
            "channel_type": "bollinger",
            "lookback": cfg.lookback,
            "take_profit_atr": 2.5, "stop_loss_atr": 1.5, "max_holding_bars": 25,
            "entry_zone": 0.3, "exit_zone": 0.7,
            "require_volume_contraction": False, "require_reversal_candle": False,
        },
        "risk": {"max_position_pct": 0.95},
    }
    engine = OscillationBacktestEngine(bt_cfg)
    result = engine.run(df_out, symbol=symbol)
    engine.summary(result)

    # ── Trade log ──
    trades = result.get("trades", [])
    if trades:
        print(f"\n  Trade Log:")
        print(f"  {'#':>3s}  {'Entry':>12s}  {'Exit':>12s}  {'Ret%':>7s}  "
              f"{'Hold':>4s}  {'Price':>14s}  {'EntryPos':>8s}")
        print(f"  {'-'*67}")
        for i, t in enumerate(trades, 1):
            entry_d = str(t.get("entry_date", "?"))[:10]
            exit_d = str(t.get("exit_date", "?"))[:10]
            ret = t.get("return_pct", 0)
            hold = t.get("holding_bars", 0)
            ep = t.get("entry_price", 0)
            xp = t.get("exit_price", 0)
            epos = t.get("entry_osc_position", "?")
            label = "WIN" if t.get("gross_pnl", 0) > 0 else "LOSS"
            print(f"  {label:>4s} {i:2d}  {entry_d:>12s}  {exit_d:>12s}  "
                  f"{ret:>6.2f}%  {hold:>3d}d  {ep:>7.2f}->{xp:>7.2f}  {str(epos):>8s}")

    return result


def run_param_sweep(df: pd.DataFrame, symbol: str, name: str):
    """Run a parameter sweep across strength values."""
    print(f"\n  ── Parameter Sweep ──")
    print(f"  {'Strength':>10s}  {'Return%':>8s}  {'Sharpe':>7s}  {'Trades':>7s}  "
          f"{'Win%':>6s}  {'MaxDD%':>7s}  {'PFactor':>8s}")
    print(f"  {'-'*65}")

    results = []
    for s in [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
        cfg = OscillationConfig(
            oscillation_strength=s,
            lookback=min(60, len(df) // 3),
            take_profit_atr=2.5, stop_loss_atr=1.5, max_holding_bars=25,
            entry_zone=0.3, exit_zone=0.7,
            require_volume_contraction=False, require_reversal_candle=False,
        )
        bt_cfg = {
            "backtest": {"initial_capital": 100000, "commission_pct": 0.0003,
                         "slippage_pct": 0.001},
            "oscillation_trading": {
                "oscillation_strength": s, "channel_type": "bollinger",
                "lookback": cfg.lookback,
                "take_profit_atr": 2.5, "stop_loss_atr": 1.5, "max_holding_bars": 25,
                "entry_zone": 0.3, "exit_zone": 0.7,
                "require_volume_contraction": False, "require_reversal_candle": False,
            },
            "risk": {"max_position_pct": 0.95},
        }
        engine = OscillationBacktestEngine(bt_cfg)
        r = engine.run(df, symbol=symbol)
        pf = r.get("profit_factor", 0)
        pf_str = f"{pf:.2f}" if pf != float("inf") else "  inf"
        print(f"  {s:>10.1f}  {r['total_return_pct']:>7.2f}%  {r['sharpe_ratio']:>7.2f}  "
              f"{r['total_trades']:>5d}  {r['win_rate_pct']:>5.1f}%  "
              f"{r['max_drawdown_pct']:>6.2f}%  {pf_str:>8s}")
        results.append({"strength": s, **r})

    return results


def main():
    print(f"\n{'='*70}")
    print(f"  震荡做T策略 — 四只股票综合测试")
    print(f"  日期: 2026-01 ~ 2026-06")
    print(f"{'='*70}\n")

    all_results = {}

    for symbol, name in STOCKS.items():
        print(f"\n{'#'*70}")
        print(f"#  Loading {name} ({symbol})...")
        try:
            df = load_data(symbol)
            print(f"#  OK — {len(df)} bars loaded")
        except Exception as e:
            print(f"#  FAILED — {e}")
            continue

        # 1. Run default strength=1.0 test
        print(f"\n{'─'*70}")
        r1 = run_single_test(df, symbol, name, strength=1.0)

        # 2. Parameter sweep
        sweep = run_param_sweep(df, symbol, name)

        # 3. Run with optimization
        print(f"\n{'─'*70}")
        print(f"  ── Optimized Test ──")
        r2 = run_single_test(df, symbol, name, strength=1.0, do_optimize=True)

        all_results[symbol] = {
            "name": name,
            "df": df,
            "default": r1,
            "optimized": r2,
            "sweep": sweep,
        }

    # ── 汇总对比 ──
    print(f"\n\n{'='*70}")
    print(f"  最终对比汇总")
    print(f"{'='*70}")
    print(f"  {'股票':>10s}  {'策略':>12s}  {'Return%':>8s}  {'Sharpe':>7s}  "
          f"{'Trades':>6s}  {'Win%':>6s}  {'MaxDD%':>7s}  {'PFactor':>8s}")
    print(f"  {'-'*70}")

    for symbol, data in all_results.items():
        r1 = data["default"]
        r2 = data["optimized"]
        name = data["name"]

        pf1 = r1.get("profit_factor", 0)
        pf1_s = f"{pf1:.2f}" if pf1 != float("inf") else "  inf"
        pf2 = r2.get("profit_factor", 0)
        pf2_s = f"{pf2:.2f}" if pf2 != float("inf") else "  inf"

        print(f"  {name:>10s}  {'默认(1.0)':>12s}  {r1['total_return_pct']:>7.2f}%  "
              f"{r1['sharpe_ratio']:>7.2f}  {r1['total_trades']:>4d}  "
              f"{r1['win_rate_pct']:>5.1f}%  {r1['max_drawdown_pct']:>6.2f}%  {pf1_s:>8s}")
        print(f"  {'':>10s}  {'优化后':>12s}  {r2['total_return_pct']:>7.2f}%  "
              f"{r2['sharpe_ratio']:>7.2f}  {r2['total_trades']:>4d}  "
              f"{r2['win_rate_pct']:>5.1f}%  {r2['max_drawdown_pct']:>6.2f}%  {pf2_s:>8s}")

    print(f"\n{'='*70}")
    print(f"  测试完毕")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
