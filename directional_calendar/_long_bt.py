"""Run long-history backtest with 4 scenarios on synthetic long data."""
import subprocess
import sys, os, re

PYTHON = r"D:\veighna_studio\python.exe"
SCRIPT = r"c:\Users\wang\WorkBuddy\20260425111208\directional_calendar\quick_backtest.py"
DATA_DIR = r"c:\Users\wang\WorkBuddy\20260425111208\directional_calendar\data"
REPORT_DIR = DATA_DIR
PRODUCTS = "IF IH IC IM"

scenarios = [
    ("Long_IF_baseline", "IF:+1"),
    ("Long_IM_strong", "IM:+2"),
    ("Long_mixed_long", "IM:+2,IC:+1,IF:+1"),
    ("Long_short_bias", "IC:-1,IH:-1"),
]

results = []

for name, signal in scenarios:
    report_path = os.path.join(REPORT_DIR, f"bt_{name}.html")
    cmd = [
        PYTHON, SCRIPT,
        "--data-dir", DATA_DIR,
        "--products", "IF", "IH", "IC", "IM",
        "--fixed-signal", signal,
        "--capital", "1000000",
        "--volume", "1",
        "--sigma-entry", "1.0",
        "--sigma-exit", "0.3",
        "--cooldown", "6",
        "--html-report", report_path,
    ]
    print(f"\n{'='*70}")
    print(f"  Scenario: {name} | Signal: {signal}")
    print(f"{'='*70}")
    
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    
    ret_match = re.search(r"Total Return:\s*\+?([\-\d.]+)%", output)
    trades_match = re.search(r"Total Trades:\s*(\d+)", output)
    roll_match = re.search(r"Rollover:\s*(\d+)", output)
    switch_match = re.search(r"Product Switch:\s*(\d+)", output)
    wr_match = re.search(r"Win Rate:\s*([\d.]+)%", output)
    pf_match = re.search(r"Profit Factor:\s*([\d.]+)", output)
    dd_match = re.search(r"Max Drawdown:\s*([\-\d.]+)%", output)
    ann_match = re.search(r"Annualized:\s*\+?([\-\d.]+)%", output)
    
    ret = float(ret_match.group(1)) if ret_match else 0
    n_trades = int(trades_match.group(1)) if trades_match else 0
    n_roll = int(roll_match.group(1)) if roll_match else 0
    n_switch = int(switch_match.group(1)) if switch_match else 0
    wr = float(wr_match.group(1)) if wr_match else 0
    pf = float(pf_match.group(1)) if pf_match else 0
    dd = float(dd_match.group(1)) if dd_match else 0
    ann = float(ann_match.group(1)) if ann_match else 0
    
    results.append({
        "name": name, "signal": signal,
        "return_pct": ret, "annualized": ann,
        "trades": n_trades, "rollover": n_roll, "product_switch": n_switch,
        "win_rate": wr, "profit_factor": pf, "max_drawdown": dd,
        "report": report_path,
    })
    
    # Print key lines
    for line in output.split("\n"):
        if any(kw in line for kw in [
            "Total Return", "Annualized", "Max Drawdown",
            "Total Trades", "Win Rate", "Profit Factor",
            "Products Used",
            "OPEN ", "SWITCH_", "CLOSE ", "ROLLOVER",
        ]):
            print(f"  {line.strip()}")

# Summary table
print("\n\n")
print("=" * 95)
print("  LONG-HISTORY BACKTEST RESULTS (Synthetic 5min from Daily, up to 9 years)")
print("=" * 95)
print(f"{'Scenario':<24} {'Signal':<22} {'Ret%':>7} {'Ann%':>7} {'Trd':>5} {'Roll':>5} {'Sw':>4} {'WR%':>6} {'PF':>6} {'DD%':>7}")
print("-" * 95)
for r in results:
    print(f"{r['name']:<24} {r['signal']:<22} {r['return_pct']:>+6.2f}% {r['annualized']:>+6.2f}% "
          f"{r['trades']:>5} {r['rollover']:>5} {r['product_switch']:>4} {r['win_rate']:>5.1f}% "
          f"{r['profit_factor']:>6.2f} {r['max_drawdown']:>+6.2f}%")

print()
print("Reports:")
for r in results:
    print(f"  {r['name']}: {r['report']}")
