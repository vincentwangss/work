"""Run 4 multi-product backtest scenarios and compare"""
import subprocess
import sys
import os
import re

PYTHON = r"D:\veighna_studio\python.exe"
SCRIPT = r"c:\Users\wang\WorkBuddy\20260425111208\directional_calendar\quick_backtest.py"
DATA_DIR = r"c:\Users\wang\WorkBuddy\20260425111208\directional_calendar\data"
REPORT_DIR = DATA_DIR
PRODUCTS = "IF IH IC IM"

scenarios = [
    ("S1_IF_baseline", "IF:+1"),
    ("S2_IM_strong", "IM:+2"),
    ("S3_mixed_long", "IM:+2,IC:+1,IF:+1"),
    ("S4_short_bias", "IC:-1,IH:-1"),
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
        "--html-report", report_path,
    ]
    print(f"\n{'='*70}")
    print(f"  Scenario: {name} | Signal: {signal}")
    print(f"{'='*70}")
    
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    
    # Extract key metrics from output
    ret_match = re.search(r"Total Return:\s*\+?([\-\d.]+)%", output)
    trades_match = re.search(r"Total Trades:\s*(\d+)", output)
    roll_match = re.search(r"Rollover:\s*(\d+)", output)
    switch_match = re.search(r"Product Switch:\s*(\d+)", output)
    wr_match = re.search(r"Win Rate:\s*([\d.]+)%", output)
    pf_match = re.search(r"Profit Factor:\s*([\d.]+)", output)
    dd_match = re.search(r"Max Drawdown:\s*([\-\d.]+)%", output)
    
    ret = float(ret_match.group(1)) if ret_match else 0
    n_trades = int(trades_match.group(1)) if trades_match else 0
    n_roll = int(roll_match.group(1)) if roll_match else 0
    n_switch = int(switch_match.group(1)) if switch_match else 0
    wr = float(wr_match.group(1)) if wr_match else 0
    pf = float(pf_match.group(1)) if pf_match else 0
    dd = float(dd_match.group(1)) if dd_match else 0
    
    results.append({
        "name": name,
        "signal": signal,
        "return_pct": ret,
        "trades": n_trades,
        "rollover": n_roll,
        "product_switch": n_switch,
        "win_rate": wr,
        "profit_factor": pf,
        "max_drawdown": dd,
        "report": report_path,
    })
    
    # Print key lines from output (trade list etc.)
    for line in output.split("\n"):
        if any(kw in line for kw in [
            "Total Return", "Annualized", "Max Drawdown",
            "Total Trades", "Win Rate", "Profit Factor",
            "OPEN ", "SWITCH_", "CLOSE ", "ROLLOVER",
            "Products Used",
        ]):
            print(f"  {line.strip()}")

# ============================================================
# Summary table
# ============================================================
print("\n")
print("=" * 90)
print("  MULTI-PRODUCT BACKTEST COMPARISON SUMMARY")
print("=" * 90)
print(f"{'Scenario':<24} {'Signal':<22} {'Ret%':>7} {'Trd':>5} {'Roll':>5} {'Sw':>4} {'WR%':>6} {'PF':>5} {'DD%':>7}")
print("-" * 90)
for r in results:
    print(f"{r['name']:<24} {r['signal']:<22} {r['return_pct']:>+6.2f}% {r['trades']:>5} "
          f"{r['rollover']:>5} {r['product_switch']:>4} {r['win_rate']:>5.1f}% {r['profit_factor']:>5.2f} {r['max_drawdown']:>+6.2f}%")

print()
print("Reports generated:")
for r in results:
    print(f"  {r['name']}: {r['report']}")
