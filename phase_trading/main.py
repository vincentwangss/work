#!/usr/bin/env python3
"""Phase Trading System — CLI entry point.

A market-phase-aware trading system that identifies three phases
(主升 / 震荡 / 主跌) and applies phase-specific strategies.

Usage:
    python main.py backtest --symbol 000001 --start 20230101 --end 20240101
    python main.py scan --symbols 000001,600000 --start 20230101
    python main.py analyze --results reports/results_000001.json
    python main.py list
"""

import argparse
import json
import os
import sys

import pandas as pd
import yaml

from data_loader import DataLoader
from backtest import BacktestEngine, OscillationBacktestEngine
from oscillation_trader import OscillationConfig, find_oscillation_stocks


def load_config(path="config.yaml"):
    """Load YAML configuration."""
    if not os.path.exists(path):
        print(f"Config not found: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_backtest(config, args):
    """Run a single backtest."""
    loader = DataLoader(config)
    engine = BacktestEngine(config)

    print(f"\nLoading data: {args.symbol} [{args.start} ~ {args.end}]")
    df = loader.load(args.symbol, args.start, args.end)
    print(f"  Loaded {len(df)} bars ({df.index[0].date()} ~ {df.index[-1].date()})")

    print(f"Running backtest...")
    results = engine.run(df, symbol=args.symbol)

    engine.summary(results)

    # Save results
    if args.output:
        os.makedirs(config.get("backtest", {}).get("output_dir", "reports"), exist_ok=True)
        path = args.output
    else:
        out_dir = config.get("backtest", {}).get("output_dir", "reports")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"results_{args.symbol}.json")

    # Convert dates to strings for JSON serialization
    _make_json_serializable(results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Results saved to: {path}")

    return results


def cmd_scan(config, args):
    """Scan multiple symbols."""
    symbols = [s.strip() for s in args.symbols.split(",")]
    loader = DataLoader(config)
    engine = BacktestEngine(config)
    all_results = []

    for sym in symbols:
        try:
            print(f"\n{'─'*50}")
            print(f"Processing: {sym}")
            df = loader.load(sym, args.start, args.end or "20260101")
            results = engine.run(df, symbol=sym)
            engine.summary(results)
            _make_json_serializable(results)
            all_results.append(results)
        except Exception as e:
            print(f"  Error processing {sym}: {e}")
            continue

    # Save combined results
    out_dir = config.get("backtest", {}).get("output_dir", "reports")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "scan_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nScan results saved to: {path}")

    # Summary ranking
    if all_results:
        print(f"\n{'='*60}")
        print(f"  RANKING (by total return)")
        print(f"{'='*60}")
        ranked = sorted(all_results, key=lambda r: r.get("total_return_pct", 0), reverse=True)
        for i, r in enumerate(ranked, 1):
            print(f"  {i:2d}. {r['symbol']:>8s}  "
                  f"Return: {r.get('total_return_pct', 0):>7.2f}%  "
                  f"Sharpe: {r.get('sharpe_ratio', 0):>5.2f}  "
                  f"DD: {r.get('max_drawdown_pct', 0):>5.2f}%  "
                  f"Trades: {r.get('total_trades', 0):>3d}  "
                  f"Win: {r.get('win_rate_pct', 0):>4.1f}%")

    return all_results


def cmd_analyze(config, args):
    """Analyze a saved backtest result."""
    path = args.results
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for r in data:
            _print_analysis(r, path)
    else:
        _print_analysis(data, path)


def cmd_list(config, args):
    """List available symbols from data source."""
    loader = DataLoader(config)
    available = loader.list_available()
    if available is None:
        print("akshare data source: all A-share stocks available")
        print("  Example: python main.py backtest --symbol 000001")
    else:
        print("Available symbols:")
        for s in available:
            print(f"  {s}")


def _print_analysis(results, path):
    """Print detailed analysis of results."""
    print(f"\n{'='*60}")
    print(f"  Analysis: {results.get('symbol', 'unknown')} (from {path})")
    print(f"{'='*60}")
    print(f"  Period:        {results.get('date_range', 'N/A')}")
    print(f"  Total Return:  {results.get('total_return_pct', 0):>7.2f}%")
    print(f"  Annual Return: {results.get('annual_return_pct', 0):>7.2f}%")
    print(f"  Sharpe Ratio:  {results.get('sharpe_ratio', 0):>7.2f}")
    print(f"  Max Drawdown:  {results.get('max_drawdown_pct', 0):>7.2f}%")
    print(f"  Win Rate:      {results.get('win_rate_pct', 0):>7.1f}%")
    print(f"  Profit Factor: {results.get('profit_factor', 0):>7.2f}")
    print(f"  Total Trades:  {results.get('total_trades', 0):>7d}")
    print(f"  Phase Dist:    {results.get('phase_distribution', {})}")

    trades = results.get("trades", [])
    print(f"\n  Recent Trades (last {min(10, len(trades))}):")
    for t in trades[-10:]:
        entry = t.get("entry_date", "?")[:10]
        exit_d = t.get("exit_date", "?")[:10]
        ret = t.get("return_pct", 0)
        phase = t.get("phase_name", "?")
        label = "WIN" if t.get("gross_pnl", 0) > 0 else "LOSS"
        print(f"    {label:5s} | {entry} → {exit_d} | "
              f"Return: {ret:>6.2f}% | Entry phase: {phase}")


def cmd_osc_backtest(config, args):
    """Run oscillation 做T backtest on a single symbol."""
    # Override oscillation_strength from CLI if specified
    if args.strength is not None:
        if "oscillation_trading" not in config:
            config["oscillation_trading"] = {}
        config["oscillation_trading"]["oscillation_strength"] = args.strength

    if args.channel_type:
        if "oscillation_trading" not in config:
            config["oscillation_trading"] = {}
        config["oscillation_trading"]["channel_type"] = args.channel_type

    # Override source from CLI
    if args.source and "data" not in config:
        config["data"] = {"source": "akshare"}
    if args.source:
        config["data"]["source"] = args.source

    loader = DataLoader(config)

    print(f"\n{'='*60}")
    source_info = f"{args.source or config['data']['source']}"
    if args.freq:
        source_info += f" / {args.freq}min"
    print(f"  Source: {source_info}")
    print(f"  震荡做T Backtest: {args.symbol}")
    strength = config.get("oscillation_trading", {}).get("oscillation_strength", 1.0)
    channel = config.get("oscillation_trading", {}).get("channel_type", "bollinger")
    print(f"  Strength: {strength}  |  Channel: {channel}")
    if args.optimize:
        print(f"  Optimize:  ON (window={args.optimize_bars}b, metric={args.optimize_metric})")
    print(f"{'='*60}")

    print(f"Loading data: {args.symbol} [{args.start} ~ {args.end}]")
    if args.freq:
        df = loader.load_minute(args.symbol, args.start, args.end, args.freq)
    else:
        df = loader.load(args.symbol, args.start, args.end)
    print(f"  Loaded {len(df)} bars ({df.index[0]} ~ {df.index[-1]})")

    # ── Optional: Rolling optimization ──
    if args.optimize:
        from oscillation_trader import OscillationConfig, StrengthOptimizer, copy_config_with_strength
        osc_cfg = OscillationConfig.from_dict(config.get("oscillation_trading", {}))
        optimizer = StrengthOptimizer(osc_cfg)

        print(f"\n  Optimizing oscillation_strength on last {args.optimize_bars} bars...")
        best_strength, opt_results = optimizer.optimize(
            df,
            validation_bars=args.optimize_bars,
            metric=args.optimize_metric,
        )

        if opt_results:
            print(optimizer.summary(opt_results, best_strength))
            print(f"\n    >> Best strength: {best_strength}")
            config["oscillation_trading"]["oscillation_strength"] = best_strength
            strength = best_strength
        else:
            print(f"    (no valid candidates, using configured strength={strength})")

    # Re-create engine with (possibly optimized) config
    engine = OscillationBacktestEngine(config)
    print(f"  Loaded {len(df)} bars ({df.index[0].date()} ~ {df.index[-1].date()})")

    print(f"Running backtest...")
    results = engine.run(df, symbol=args.symbol)
    engine.summary(results)

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        path = args.output
    else:
        out_dir = config.get("backtest", {}).get("output_dir", "reports")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"osc_results_{args.symbol}.json")

    _make_json_serializable(results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Results saved to: {path}")

    return results


def cmd_osc_scan(config, args):
    """Scan multiple symbols for oscillation suitability."""
    symbols = [s.strip() for s in args.symbols.split(",")]

    if args.strength is not None:
        if "oscillation_trading" not in config:
            config["oscillation_trading"] = {}
        config["oscillation_trading"]["oscillation_strength"] = args.strength

    loader = DataLoader(config)
    df_dict = {}

    print(f"\nLoading {len(symbols)} symbols...")
    for sym in symbols:
        try:
            df = loader.load(sym, args.start, args.end or "20260101")
            df_dict[sym] = df
            print(f"  {sym}: {len(df)} bars")
        except Exception as e:
            print(f"  {sym}: ERROR — {e}")

    if not df_dict:
        print("No data loaded.")
        return

    print(f"\nScanning for oscillation patterns (strength={args.strength or 'default'})...")
    results = find_oscillation_stocks(df_dict, config)

    # Print results table
    print(f"\n{'='*80}")
    print(f"  Oscillation Stock Scan Results")
    print(f"{'='*80}")
    print(f"  {'Symbol':>8s}  {'Score':>6s}  {'Range%':>7s}  {'Z-Score':>8s}  {'Pos':>5s}  "
          f"{'Bounces':>7s}  {'Bias':>10s}  {'Vol':>5s}")
    print(f"  {'-'*66}")
    for r in results:
        print(f"  {r['symbol']:>8s}  {r['score']:>6.3f}  {r['range_pct']:>6.2f}%  "
              f"{r['zscore']:>+8.2f}  {r['position']:>5.3f}  {r['bounce_count']:>3d}/{r.get('bounce_required', '?'):>3s}  "
              f"{r['recent_bias']:>10s}  {r['volume_ratio']:>5.2f}")
    print(f"{'='*80}")
    print(f"  Found {len(results)} oscillating stocks out of {len(symbols)} scanned.")

    # Save results
    out_dir = config.get("backtest", {}).get("output_dir", "reports")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "osc_scan_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Scan results saved to: {path}")


def _make_json_serializable(obj):
    """Convert non-serializable types in-place for JSON export."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, pd.Timestamp):
                obj[k] = str(v)
            else:
                _make_json_serializable(v)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _make_json_serializable(item)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase Trading System — market-phase-aware trading strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py backtest --symbol 000001 --start 20230101 --end 20240101
  python main.py scan --symbols 000001,600000,300750 --start 20230101
  python main.py analyze --results reports/results_000001.json
  python main.py list
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # backtest
    bt = subparsers.add_parser("backtest", help="Run backtest on a single symbol")
    bt.add_argument("--symbol", required=True, help="Stock symbol (e.g., 000001)")
    bt.add_argument("--start", default="20200101", help="Start date (YYYYMMDD)")
    bt.add_argument("--end", default="20260101", help="End date (YYYYMMDD)")
    bt.add_argument("--output", help="Output JSON path")
    bt.add_argument("--config", default="config.yaml", help="Config file path")

    # scan
    sc = subparsers.add_parser("scan", help="Scan multiple symbols")
    sc.add_argument("--symbols", required=True, help="Comma-separated symbols")
    sc.add_argument("--start", default="20200101", help="Start date (YYYYMMDD)")
    sc.add_argument("--end", default="20260101", help="End date (YYYYMMDD)")
    sc.add_argument("--config", default="config.yaml", help="Config file path")

    # analyze
    an = subparsers.add_parser("analyze", help="Analyze saved backtest results")
    an.add_argument("--results", required=True, help="Path to results JSON file")

    # list
    subparsers.add_parser("list", help="List available symbols")

    # ── Oscillation 做T ──
    osc = subparsers.add_parser(
        "osc", help="Oscillation 做T backtest on a single symbol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py osc --symbol 000001 --start 20230101 --end 20240101
  python main.py osc --symbol 600519 --strength 1.5 --channel zscore
  python main.py osc --symbol 000001 --strength 0.8 --output my_results.json
        """,
    )
    osc.add_argument("--symbol", required=True, help="Stock symbol")
    osc.add_argument("--start", default="20260101", help="Start date (YYYYMMDD)")
    osc.add_argument("--end", default="20260628", help="End date (YYYYMMDD)")
    osc.add_argument("--source", default=None, choices=["akshare", "baostock"],
                     help="Data source (baostock has longer minute history)")
    osc.add_argument("--freq", default=None, choices=["5", "15", "30", "60"],
                     help="Use minute K-line data instead of daily")
    osc.add_argument("--strength", type=float, default=None,
                     help="Oscillation strength (0.5~5.0, overrides config)")
    osc.add_argument("--channel", dest="channel_type", default=None,
                     choices=["bollinger", "percentile", "zscore"],
                     help="Channel type (overrides config)")
    osc.add_argument("--output", help="Output JSON path")
    osc.add_argument("--optimize", action="store_true",
                     help="Auto-optimize oscillation_strength on recent data")
    osc.add_argument("--optimize-bars", type=int, default=120,
                     help="Validation window for optimizer (bars)")
    osc.add_argument("--optimize-metric", default="sharpe_ratio",
                     choices=["sharpe_ratio", "total_return_pct", "profit_factor", "composite"],
                     help="Metric to maximize in optimization")
    osc.add_argument("--config", default="config.yaml", help="Config file path")

    # ── Oscillation Scan ──
    oscan = subparsers.add_parser(
        "osc-scan", help="Scan multiple symbols for oscillation suitability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py osc-scan --symbols 000001,600519,300750 --start 20240101
  python main.py osc-scan --symbols 000001,600000 --strength 1.2
        """,
    )
    oscan.add_argument("--symbols", required=True, help="Comma-separated symbols")
    oscan.add_argument("--start", default="20240101", help="Start date")
    oscan.add_argument("--end", default="20260101", help="End date")
    oscan.add_argument("--strength", type=float, default=None,
                       help="Oscillation strength (0.5~5.0, overrides config)")
    oscan.add_argument("--config", default="config.yaml", help="Config file path")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "backtest":
        config = load_config(args.config)
        cmd_backtest(config, args)
    elif args.command == "scan":
        config = load_config(args.config)
        cmd_scan(config, args)
    elif args.command == "analyze":
        cmd_analyze(None, args)
    elif args.command == "list":
        config = load_config()
        cmd_list(config, args)
    elif args.command == "osc":
        config = load_config(args.config)
        cmd_osc_backtest(config, args)
    elif args.command == "osc-scan":
        config = load_config(args.config)
        cmd_osc_scan(config, args)
    else:
        print("Unknown command. Use -h for help.")
        sys.exit(1)


if __name__ == "__main__":
    main()
