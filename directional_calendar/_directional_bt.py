"""
定向信号回测 - 真实数据 + 券商分红预测 + 时间窗口信号

场景设计：
  S1: 2026-03-25 看多 IF，持续 5 个交易日，其他时间无观点
  S2: 对比 - 全程看多 IF（基准）
  S3: 对比 - 3/25 看多 IM 持续 5 天
  S4: 对比 - 3/25 看空 IC 持续 5 天

数据：真实 5 分钟基差数据 (akshare Sina, 30天)
分红：基于国信证券/东方证券研报的预测点数
"""

import sys
import os
import subprocess

DATA_DIR = r"C:\Users\wang\WorkBuddy\20260425111208\directional_calendar\data"
BT_PY = r"C:\Users\wang\WorkBuddy\20260425111208\directional_calendar\quick_backtest.py"
PYTHON = r"D:\veighna_studio\python.exe"

# ============================================================
# 场景定义: (name, fixed_signal_spec, description)
# ============================================================
SCENARIOS = [
    ("S1_325_IF_long5d", "IF:+1",
     "3/25起看多IF 5个交易日"),
    ("S2_full IF_long", "IF:+1",
     "全程看多IF(基准)"),
    ("S3_325_IM_long5d", "IM:+2",
     "3/25起看多IM 5个交易日"),
    ("S4_325_IC_short5d", "IC:-1",
     "3/25起看空IC 5个交易日"),
]

PRODUCTS = ["IF", "IH", "IC", "IM"]


def run_backtest(name: str, signal: str, desc: str) -> dict:
    """运行单个回测场景。"""
    html_out = os.path.join(DATA_DIR, f"bt_{name}.html")
    
    cmd = [
        PYTHON, BT_PY,
        "--data-dir", DATA_DIR,
        "--products"] + PRODUCTS + [
        "--fixed-signal", signal,
        "--html-report", html_out,
    ]
    
    print(f"\n{'='*70}")
    print(f"  Scenario: {name}")
    print(f"  Signal: {signal}  |  {desc}")
    print(f"  Output: {html_out}")
    print(f"{'='*70}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(BT_PY),
    )
    
    # 输出 stdout 的关键行
    lines = result.stdout.strip().split("\n")
    for line in lines:
        if any(kw in line for kw in [
            "Scenario", "Total P&L", "Total Return", "Annualized",
            "Trades", "Win Rate", "Profit Factor", "Max Drawdown",
            "Rollover", "Loaded", "Stats:", "Range:",
            "Contract chain", "Signal:", "[INFO]", "[WARN]",
            "=== Backtest", "equity",
        ]):
            print(f"  {line}")

    if result.returncode != 0:
        print(f"  [ERROR] returncode={result.returncode}")
        err_lines = result.stderr.strip().split("\n")[-10:]
        for el in err_lines:
            if el.strip():
                print(f"  STDERR: {el}")

    # 解析关键指标
    metrics = {"name": name, "signal": signal, "desc": desc}
    for line in lines:
        if "Total Return:" in line or "total_return" in line.lower():
            try:
                metrics["return_pct"] = float(line.split(":")[-1].strip().replace("%",""))
            except:
                pass
        if "Annualized" in line and "%" in line:
            try:
                metrics["annual"] = float(line.split(":")[-1].strip().replace("%",""))
            except:
                pass
        if "Total P&L" in line or "total_pnl" in line.lower():
            try:
                v = line.split(":")[-1].strip().replace(",","").replace("¥","").replace("+","")
                metrics["pnl"] = float(v)
            except:
                pass
        if "Trades:" in line or "trade_count" in line.lower():
            try:
                metrics["trades"] = int(''.join(filter(str.isdigit, line)))
            except:
                pass
        if "Win Rate:" in line:
            try:
                metrics["winrate"] = float(line.split(":")[-1].strip().replace("%",""))
            except:
                pass
        if "PF" in line and "Profit Factor" not in line and ":" in line:
            try:
                metrics["pf"] = float(line.split(":")[-1].strip())
            except:
                pass
        if "Max Drawdown" in line or "max_dd" in line.lower():
            try:
                metrics["maxdd"] = float(line.split(":")[-1].strip().replace("%","").replace("-",""))
            except:
                pass
        if "Rollover" in line and ":" in line:
            try:
                parts = line.split(":")
                metrics["rollover"] = int(''.join(filter(str.isdigit, parts[-1])))
            except:
                pass
    
    metrics["html"] = html_out
    metrics["ok"] = os.path.exists(html_out)
    return metrics


def main():
    results = []
    for name, signal, desc in SCENARIOS:
        m = run_backtest(name, signal, desc)
        results.append(m)
    
    # ============================================================
    # 汇总对比表
    # ============================================================
    print("\n")
    print("=" * 90)
    print("  定向信号回测汇总 | 真实数据 + 分红预测 | 2026-03-25 ~ 2026-04-24")
    print("=" * 90)
    
    header = f"{'场景':<22} {'信号':<14} {'收益率':>8} {'年化':>8} {'笔数':>6} {'胜率':>6} {'PF':>6} {'回撤':>8} {'换月':>5}"
    print(header)
    print("-" * 90)
    
    for m in results:
        ret = m.get("return_pct", 0)
        ann = m.get("annual", 0)
        trd = m.get("trades", 0)
        wr = m.get("winrate", 0)
        pf = m.get("pf", 0)
        dd = m.get("maxdd", 0)
        ro = m.get("rollover", 0)
        
        row = f"{m['name']:<22} {m['signal']:<14} {ret:>+7.2f}% {ann:>7.1f}% {trd:>5} {wr:>5.1f}% {pf:>5.2f} {-dd:>7.2f}% {ro:>5}"
        print(row)
        print(f"  └─ {m['desc']}")
    
    print("-" * 90)

    # 找最优
    valid = [r for r in results if r.get("return_pct") is not None]
    if valid:
        best = max(valid, key=lambda x: x.get("return_pct", 0))
        worst = min(valid, key=lambda x: x.get("return_pct", 0))
        print(f"\n  最佳: {best['name']} ({best['desc']}) → {best.get('return_pct',0):+.2f}%")
        print(f"  最差: {worst['name']} ({worst['desc']}) → {worst.get('return_pct',0):+.2f}%")

    # HTML 报告列表
    print(f"\n  HTML 报告:")
    for m in results:
        status = "OK" if m["ok"] else "MISSING"
        print(f"    [{status}] {m['html']}")

    return results


if __name__ == "__main__":
    main()
