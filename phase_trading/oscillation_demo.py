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
    StrengthOptimizer,
)
from backtest import OscillationBacktestEngine
from data_loader import DataLoader


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


def load_real_data(symbol, start="20240101", end="20260101", source="akshare", freq=None):
    """Load real stock data from specified source."""
    config = {"data": {"source": source, "adjust": "qfq", "cache_dir": "data/cache"}}
    loader = DataLoader(config)

    try:
        if freq:
            # Minute-level data
            df = loader.load_minute(symbol, start, end, freq)
            print(f"  Loaded {symbol}: {len(df)} bars ({freq}min), "
                  f"{df.index[0]} ~ {df.index[-1]}, "
                  f"price [{df['low'].min():.2f}, {df['high'].max():.2f}]")
        else:
            # Daily data
            df = loader.load(symbol, start, end)
            print(f"  Loaded {symbol}: {len(df)} daily bars, "
                  f"{df.index[0].date()} ~ {df.index[-1].date()}, "
                  f"price [{df['low'].min():.2f}, {df['high'].max():.2f}]")
        return df
    except ImportError as e:
        print(f"  {e}")
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
    parser.add_argument("--start", default="20260101", help="Start date (YYYYMMDD)")
    parser.add_argument("--end", default="20260628", help="End date (YYYYMMDD)")
    parser.add_argument("--source", default="akshare",
                        choices=["akshare", "baostock"],
                        help="Data source (akshare=1.5mo min data, baostock=6mo+ min data)")
    parser.add_argument("--freq", default=None,
                        choices=["5", "15", "30", "60"],
                        help="Minute frequency (omit for daily data)")
    parser.add_argument("--strength", type=float, default=1.0, help="Oscillation strength (0.5~5.0)")
    parser.add_argument("--channel", default="bollinger", choices=["bollinger", "percentile", "zscore"],
                        help="Channel type")
    parser.add_argument("--lookback", type=int, default=60, help="Oscillation lookback period")
    parser.add_argument("--optimize", action="store_true",
                        help="Auto-optimize oscillation_strength on recent data")
    parser.add_argument("--optimize-bars", type=int, default=120,
                        help="Validation window for optimizer (bars)")
    parser.add_argument("--optimize-metric", default="sharpe_ratio",
                        choices=["sharpe_ratio", "total_return_pct", "profit_factor", "composite"],
                        help="Metric to maximize in optimization")
    parser.add_argument("--no-backtest", action="store_true", help="Skip backtest, only show signals")
    args = parser.parse_args()

    # ── 1. Load data ──
    print(f"\n{'='*60}")
    print(f"  震荡做T策略 Demo")
    print(f"{'='*60}")

    if args.symbol:
        source_name = f"{args.source} {'/' + args.freq + 'min' if args.freq else 'daily'}"
        print(f"  Source: {source_name}")
        df = load_real_data(args.symbol, args.start, args.end,
                             source=args.source, freq=args.freq)
    else:
        df = generate_synthetic_data()

    # Smart lookback: minute data needs shorter lookback
    auto_lookback = args.lookback
    if args.freq and args.lookback == 60:
        auto_lookback = min(45, len(df) // 12)  # ~3-4 trading days for minute data

    # ── 2. Config ──
    config = OscillationConfig(
        oscillation_strength=args.strength,
        channel_type=args.channel,
        lookback=auto_lookback,
        take_profit_atr=2.5,
        stop_loss_atr=1.5,
        max_holding_bars=25,
        entry_zone=0.3,
        exit_zone=0.7,
        require_volume_contraction=False,
        require_reversal_candle=False,
    )

    print(f"\n  Config{' (optimized)' if args.optimize else ''}:")
    print(f"    Strength:       {config.oscillation_strength}")
    print(f"    Channel:        {config.channel_type} (effective std={config.effective_std:.2f})")
    print(f"    Lookback:       {config.lookback}")
    print(f"    Entry zone:     position < {config.entry_zone}  (z < {config.effective_entry_zscore})")
    print(f"    Exit zone:      position > {config.exit_zone}  (z > {config.effective_exit_zscore})")
    print(f"    Min bounces:    {config.effective_min_bounces}")

    # ── 3. Optional: Optimize strength ──
    if args.optimize:
        print(f"\n  ── Rolling Optimization ──")
        print(f"    Window:         last {args.optimize_bars} bars")
        print(f"    Metric:         {args.optimize_metric}")
        print(f"    Candidates:     [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]")

        optimizer = StrengthOptimizer(config)
        best_strength, opt_results = optimizer.optimize(
            df,
            validation_bars=args.optimize_bars,
            metric=args.optimize_metric,
        )

        if opt_results:
            print(f"\n  Optimization Results:")
            print(optimizer.summary(opt_results, best_strength))
            print(f"\n    >> Best strength: {best_strength}")

            # Update config with optimized value
            from oscillation_trader import copy_config_with_strength
            config = copy_config_with_strength(config, best_strength)
            print(f"    >> Updated config strength to {config.oscillation_strength}")
        else:
            print(f"    (optimization returned no valid candidates, using default)")

    # ── 4. Compute channel ──
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
                "oscillation_strength": config.oscillation_strength,
                "channel_type": args.channel,
                "lookback": auto_lookback,
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
        hold_unit = "b" if args.freq else "d"
        trades = result.get("trades", [])
        if trades:
            print(f"\n  Trade Log:")
            print(f"  {'#':>3s}  {'Entry':>16s}  {'Exit':>16s}  {'Ret%':>7s}  "
                  f"{'Hold':>4s}  {'Price':>14s}  {'EntryPos':>8s}")
            print(f"  {'-'*74}")
            for i, t in enumerate(trades, 1):
                entry_d = str(t.get("entry_date", "?"))[:16]
                exit_d = str(t.get("exit_date", "?"))[:16]
                ret = t.get("return_pct", 0)
                hold = t.get("holding_bars", 0)
                ep = t.get("entry_price", 0)
                xp = t.get("exit_price", 0)
                epos = t.get("entry_osc_position", "?")
                label = "WIN" if t.get("gross_pnl", 0) > 0 else "LOSS"
                print(f"  {label:>4s} {i:2d}  {entry_d:>16s}  {exit_d:>16s}  "
                      f"{ret:>6.2f}%  {hold:>3d}{hold_unit}  "
                      f"{ep:>7.2f}->{xp:>7.2f}  {str(epos):>8s}")
    else:
        print("\n  (backtest skipped with --no-backtest)")

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
