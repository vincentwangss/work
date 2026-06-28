#!/usr/bin/env python3
"""
震荡做T策略 — 快速演示脚本

运行方式：
    python oscillation_demo.py                          # 用合成数据演示
    python oscillation_demo.py --symbol 000001          # 用真实股票数据 (需 akshare)
    python oscillation_demo.py --symbol 600519 --strength 1.5 --channel percentile

功能：
    1. 下载股票数据或使用合成数据
    2. 计算震荡通道
    3. 生成做T信号和回测
    4. 输出分析报告和交易记录
"""

import argparse
import sys
import numpy as np
import pandas as pd

from oscillation_trader import (
    OscillationConfig, OscillationChannel, OscillationTrader,
    find_oscillation_stocks, compute_oscillation_metrics,
)
from backtest import OscillationBacktestEngine


def generate_synthetic_data(n=300, amp=2.0, base=10.0, seed=42):
    """Generate synthetic oscillating price data."""
    np.random.seed(seed)
    t = np.linspace(0, 10 * np.pi, n)
    prices = base + amp * np.sin(t) + np.random.randn(n) * 0.15

    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": prices + np.random.randn(n) * 0.1,
        "high": prices + abs(np.random.randn(n)) * 0.2 + 0.1,
        "low": prices - abs(np.random.randn(n)) * 0.2 - 0.1,
        "close": prices,
        "volume": np.random.randint(100000, 500000, n),
    })
    df = df.set_index("date")
    print(f"  Generated synthetic data: {len(df)} bars, "
          f"price range [{df['low'].min():.2f}, {df['high'].max():.2f}], "
          f"amplitude={amp}")
    return df


def load_real_data(symbol, start="20240101", end="20260101"):
    """Load real stock data via akshare."""
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=start, end_date=end, adjust="qfq",
        )
        col_map = {"日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume"}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        print(f"  Loaded {symbol}: {len(df)} bars, "
              f"price range [{df['low'].min():.2f}, {df['high'].max():.2f}]")
        return df
    except ImportError:
        print("  akshare not installed. Install: pip install akshare")
        sys.exit(1)
    except Exception as e:
        print(f"  Error loading {symbol}: {e}")
        sys.exit(1)


def main():
    # Fix Windows GBK encoding issues
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
    parser = argparse.ArgumentParser(
        description="Oscillation 做T Strategy Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--symbol", default=None, help="Stock symbol (omit for synthetic data)")
    parser.add_argument("--start", default="20240101", help="Start date (real data only)")
    parser.add_argument("--end", default="20260101", help="End date (real data only)")
    parser.add_argument("--strength", type=float, default=1.0, help="Oscillation strength (0.5~5.0)")
    parser.add_argument("--channel", default="bollinger", choices=["bollinger", "percentile", "zscore"],
                        help="Channel type")
    parser.add_argument("--lookback", type=int, default=60, help="Oscillation lookback period")
    parser.add_argument("--no-backtest", action="store_true", help="Skip backtest, only show signals")
    args = parser.parse_args()

    # ── 1. Load data ──
    print(f"\n{'='*60}")
    print(f"  震荡做T策略 Demo")
    print(f"{'='*60}")

    if args.symbol:
        df = load_real_data(args.symbol, args.start, args.end)
    else:
        df = generate_synthetic_data()

    # ── 2. Config ──
    config = OscillationConfig(
        oscillation_strength=args.strength,
        channel_type=args.channel,
        lookback=args.lookback,
        take_profit_atr=2.5,
        stop_loss_atr=1.5,
        max_holding_bars=25,
        entry_zone=0.3,
        exit_zone=0.7,
        require_volume_contraction=False,
        require_reversal_candle=False,
    )

    print(f"\n  Config:")
    print(f"    Strength:       {config.oscillation_strength}")
    print(f"    Channel:        {config.channel_type} (effective std={config.effective_std:.2f})")
    print(f"    Lookback:       {config.lookback}")
    print(f"    Entry zone:     position < {config.entry_zone}  (z < {config.effective_entry_zscore})")
    print(f"    Exit zone:      position > {config.exit_zone}  (z > {config.effective_exit_zscore})")
    print(f"    Min bounces:    {config.effective_min_bounces}")

    # ── 3. Compute channel ──
    channel = OscillationChannel(config)
    df_out = channel.compute(df)

    # ── 4. Current state ──
    state = channel.get_current_state(df_out, len(df_out) - 1)
    print(f"\n  Current State:")
    print(f"    Oscillating:    {state['is_oscillating']}")
    print(f"    Position:       {state['position']:.3f} (0=lower, 1=upper)")
    print(f"    Z-score:        {state['zscore']:.2f}")
    print(f"    Range:          {state['range_pct']:.1f}%")
    print(f"    Bounces:        {state['bounce_count']:.0f}")

    # ── 5. Entry signals ──
    trader = OscillationTrader(config)
    signals = []
    for i in range(config.lookback, len(df_out)):
        sig = trader.evaluate_entry(df_out, i, phase=df_out.iloc[i].get("phase", 0))
        if sig != 0:
            state_i = channel.get_current_state(df_out, i)
            signals.append({
                "date": df_out.index[i],
                "close": round(df_out["close"].iloc[i], 2),
                "position": round(state_i["position"], 3),
                "zscore": round(state_i["zscore"], 2),
            })

    print(f"\n  Entry Signals (last 5 of {len(signals)}):")
    if signals:
        for s in signals[-5:]:
            print(f"    {s['date'].date()}  close={s['close']:>8.2f}  "
                  f"pos={s['position']:.3f}  z={s['zscore']:+5.2f}")
    else:
        print("    (none — try lower strength or check channel type)")

    # ── 6. Oscillation stats ──
    osc_pct = df_out["osc_is_oscillating"].sum() / len(df_out) * 100
    print(f"\n  Oscillation Coverage:")
    print(f"    Bars oscillating: {osc_pct:.1f}%")
    print(f"    Avg bounce count: {df_out['osc_bounce_count'].mean():.1f}")

    # ── 7. Backtest ──
    if not args.no_backtest:
        bt_config = {
            "backtest": {
                "initial_capital": 100000,
                "commission_pct": 0.0003,
                "slippage_pct": 0.001,
            },
            "oscillation_trading": {
                "oscillation_strength": args.strength,
                "channel_type": args.channel,
                "lookback": args.lookback,
                "take_profit_atr": 2.5,
                "stop_loss_atr": 1.5,
                "max_holding_bars": 25,
                "entry_zone": 0.3,
                "exit_zone": 0.7,
                "require_volume_contraction": False,
                "require_reversal_candle": False,
            },
            "risk": {"max_position_pct": 0.95},
        }
        engine = OscillationBacktestEngine(bt_config)
        result = engine.run(df_out, symbol=args.symbol or "SYNTH")
        engine.summary(result)

        # Print trade log
        trades = result.get("trades", [])
        if trades:
            print(f"\n  Trade Log:")
            print(f"  {'#':>3s}  {'Entry':>12s}  {'Exit':>12s}  {'Ret%':>7s}  "
                  f"{'Hold':>4s}  {'Price':>10s}  {'EntryPos':>8s}")
            print(f"  {'-'*60}")
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
                      f"{ret:>6.2f}%  {hold:>3d}d  {ep:>5.2f}->{xp:>5.2f}  {str(epos):>8s}")
    else:
        print("\n  (backtest skipped with --no-backtest)")

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
