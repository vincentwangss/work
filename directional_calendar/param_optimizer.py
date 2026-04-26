"""
:param_optimizer.py
基差套利参数优化器 - 网格搜索 + 综合评分 + 样本外测试

扫描维度：
  - z_entry: [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]   入场阈值
  - tp_pts:  [2.0, 3.0, 4.0, 5.0, 6.0]              止盈(点)
  - sl_pts:  [6.0, 8.0, 10.0, 12.0]                  止损(点)
  - time_exit: [144, 288, 432, 576]                   时间止损(根K线)

评估指标（加权综合分）：
  - Sharpe (权重 30%)
  - 年化收益 (权重 25%)
  - 最大回撤 (权重 20%，取负值，回撤越小越好)
  - 胜率 (权重 15%)
  - 盈亏比 (权重 10%)

样本外测试：
  - --oos-months N：留最后N个月做样本外验证（默认0=不分割）
  - 用样本内数据优化参数 -> 取TOP-K组 -> 在样本外数据上逐一验证
  - 报告中展示样本内 vs 样本外对比，检测过拟合

输出：
  - 控制台：TOP20 参数组合排名表 + 样本外验证结果
  - HTML：完整对比报告（热力图 + 参数雷达图 + OOS对比）
  - CSV：全部结果明细（可导入 Excel 分析）

用法:
  python param_optimizer.py --products IF IH IC IM --top-n 20
  python param_optimizer.py --products IF IH              # 只跑两个品种
  python param_optimizer.py --fast                        # 快速模式（减少组合数）
  python param_optimizer.py --oos-months 1                # 留1个月做样本外测试
  python param_optimizer.py --fast --oos-months 1         # 快速+样本外
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 导入回测引擎
from basis_arbitrage_backtest import (
    BacktestConfig, BasisArbitrageBacktesterV2, load_ccfx_data,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


# ------------------------------------------------------------------
# 数据时间范围探测
# ------------------------------------------------------------------

def detect_data_range(products: List[str]) -> Tuple[datetime, datetime]:
    """探测所有品种的数据起止时间，返回全局范围"""
    dt_min, dt_max = None, None
    for p in products:
        df = load_ccfx_data(p)
        if df.empty:
            continue
        dmin = df["datetime"].min()
        dmax = df["datetime"].max()
        if dt_min is None or dmin < dt_min:
            dt_min = dmin
        if dt_max is None or dmax > dt_max:
            dt_max = dmax
    return dt_min, dt_max


# ------------------------------------------------------------------
# 参数网格定义
# ------------------------------------------------------------------

def get_param_grid(fast: bool = False) -> dict:
    """返回参数搜索空间"""
    if fast:
        return {
            "z_entry":    [1.25, 1.5, 1.75, 2.0],
            "tp_pts":     [2.5, 3.0, 4.0, 5.0],
            "sl_pts":     [6.0, 8.0, 10.0],
            "time_exit":  [144, 288, 432],
        }
    else:
        return {
            "z_entry":    [1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
            "tp_pts":     [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
            "sl_pts":     [5.0, 6.0, 8.0, 10.0, 12.0, 15.0],
            "time_exit":  [96, 144, 288, 432, 576, 720],
        }


# ------------------------------------------------------------------
# 单次回测运行（静默模式，只返回统计）
# ------------------------------------------------------------------

def run_single_backtest(
    products: List[str],
    z_entry: float,
    tp_pts: float,
    sl_pts: float,
    time_exit: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """跑一次回测，返回汇总统计字典
    
    Args:
        start_date/end_date: 可选的数据日期过滤 (YYYY-MM-DD)
    """
    cfg = BacktestConfig(
        z_entry=z_entry,
        tp_basis_pts=tp_pts,
        sl_basis_pts=sl_pts,
        time_exit_bars=time_exit,
        lookback_bars=288,
        z_exit=0.2,
        start_date=start_date,
        end_date=end_date,
    )
    
    bt = BasisArbitrageBacktesterV2(cfg)
    results = bt.run(products)
    
    if not results or not results.get("by_product"):
        return {}
    
    by_prod = results["by_product"]
    all_trades = results["all_trades"]
    summary = results["summary"]
    
    # 汇总各品种 stats
    product_stats = {}
    total_pnl = summary["total_pnl"]
    total_trades = summary["total_trades"]
    win_rate = summary["win_rate"]
    
    # 计算全局指标
    all_equity = []
    for r in by_prod.values():
        all_equity.extend(r["equity_curve"].tolist())
    
    equity_arr = np.array(all_equity) if all_equity else np.array([1_000_000])
    
    # 全局 Sharpe（按5min bar）
    eq_diff = np.diff(equity_arr)
    sharpe = 0.0
    if len(eq_diff) > 30 and np.std(eq_diff) > 1e-6:
        sharpe = np.sqrt(288) * eq_diff.mean() / np.std(eq_diff)
    
    # 最大回撤
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / np.maximum(peak, 1e-6) * 100
    max_dd = drawdown.min()
    
    # 年化收益
    if all_trades:
        t_first = min(t.entry_time for t in all_trades)
        t_last = max(t.exit_time for t in all_trades if t.exit_time)
        span = max((t_last - t_first) / np.timedelta64(1, 'D'), 1)
        ann_ret = (total_pnl / 1_000_000) * 100 * (365 / span)
    else:
        ann_ret = 0.0
    
    # 胜率 / 盈亏比
    winning = [t for t in all_trades if t.pnl_rmb > 0]
    losing = [t for t in all_trades if t.pnl_rmb <= 0]
    wr = len(winning) / max(len(all_trades), 1) * 100
    
    gross_win = sum(t.pnl_rmb for t in winning)
    gross_loss = sum(abs(t.pnl_rmb) for t in losing)
    pf = gross_win / max(gross_loss, 1)
    
    # 平均持仓时长
    avg_hold = np.mean([t.hold_bars for t in all_trades]) * 5 / 60 if all_trades else 0
    
    # 各品种独立收益
    for p, r in by_prod.items():
        s = r["stats"]
        product_stats[p] = {
            "trades": s.get("总交易次数", 0),
            "return_pct": s.get("总收益率", "0%"),
            "ann_ret": s.get("年化收益率", "0%"),
            "max_dd": s.get("最大回撤", "0%"),
            "sharpe": s.get("Sharpe比率", "0"),
            "win_rate": s.get("胜率", "0%"),
            "pnl_yuan": s.get("总盈亏", "0元"),
        }
    
    return {
        "params": {
            "z_entry": z_entry,
            "tp_pts": tp_pts,
            "sl_pts": sl_pts,
            "time_exit": time_exit,
        },
        "summary": {
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "win_rate": wr,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "ann_ret": ann_ret,
            "profit_factor": pf,
            "avg_hold_hours": avg_hold,
        },
        "by_product": product_stats,
        "equity_curve_len": len(equity_arr),
    }


# ------------------------------------------------------------------
# 综合评分
# ------------------------------------------------------------------

def calc_score(result: dict) -> float:
    """
    综合评分 (0~100)
    
    权重分配：
      Sharpe     30%  -> 越高越好
      年化收益   25%  -> 越高越好  
      最大回撤   20%  -> 取负（越小越好）
      胜率       15%  -> 越高越好
      盈亏比     10%  -> 越高越好
    """
    s = result.get("summary", {})
    if not s or s.get("total_trades", 0) < 5:
        return -999.0  # 交易太少，无效
    
    # 归一化到 0-100 分
    def norm(val, lo, hi):
        return max(0, min(100, (val - lo) / (hi - lo) * 100))
    
    score_sharpe   = norm(s.get("sharpe", 0),   0,   3.0)   * 0.30
    score_ann_ret  = norm(s.get("ann_ret", 0),   0,   80.0)  * 0.25
    score_dd       = norm(-s.get("max_dd", 0),   -30, 0)     * 0.20  # 回撤取反
    score_wr       = norm(s.get("win_rate", 0),   40,  100.0) * 0.15
    score_pf       = norm(s.get("profit_factor", 0), 0.5, 8.0) * 0.10
    
    return score_sharpe + score_ann_ret + score_dd + score_wr + score_pf


# ------------------------------------------------------------------
# 主优化流程（支持样本内/外分割）
# ------------------------------------------------------------------

def run_optimization(
    products: List[str],
    top_n: int = 20,
    fast: bool = False,
    oos_months: int = 0,
    train_end: Optional[str] = None,
    oos_start: Optional[str] = None,
    oos_end: Optional[str] = None,
) -> Tuple[List[dict], List[dict]]:
    """执行参数网格搜索，支持样本外验证

    Args:
        oos_months: 样本外月数（>0时自动分割数据）
        train_end/oos_start/oos_end: 手动指定分割日期（优先于oos_months）
    
    Returns:
        (in_sample_results, oos_results)
        - in_sample_results: 训练集优化结果列表
        - oos_results: TOP-K参数在样本外的验证结果列表
    """
    
    # ---- 确定数据分割边界 ----
    global_start, global_end = detect_data_range(products)
    print(f"\n[数据范围] {global_start} ~ {global_end}")
    
    if oos_months > 0 and not train_end:
        # 自动计算：从全局结束日期往前推 oos_months 个月
        train_end_dt = global_end - timedelta(days=oos_months * 30)
        train_end = train_end_dt.strftime("%Y-%m-%d")
        oos_start = train_end  # 样本外起始日 = 训练集截止日的次日（含当天用 <=）
        oos_end = global_end.strftime("%Y-%m-%d")
    elif not train_end:
        train_end = None
        oos_start = None
        oos_end = None
    
    has_oos = (train_end is not None and oos_start is not None)
    
    grid = get_param_grid(fast)
    
    # 生成所有参数组合
    keys = list(grid.keys())
    values = list(grid.values())
    total_combos = 1
    for v in values:
        total_combos *= len(v)
    
    print("=" * 70)
    print("  基差套利参数优化器")
    print(f"  品种: {', '.join(products)}")
    print(f"  参数空间: {len(values)} 维 x {total_combos} 组合")
    for k, v in grid.items():
        print(f"    {k}: {v}")
    if has_oos:
        print(f"  [样本内] ~{train_end} | [样本外] {oos_start} ~ {oos_end}")
    else:
        print(f"  [全量数据] 无样本外分割")
    print("=" * 70)
    
    all_results = []
    start_time = time.time()
    
    combo_idx = 0
    for combo in itertools.product(*values):
        combo_idx += 1
        params = dict(zip(keys, combo))
        
        z_e = params["z_entry"]
        tp  = params["tp_pts"]
        sl  = params["sl_pts"]
        te  = params["time_exit"]
        
        # 安全检查：TP 必须 < SL
        if tp >= sl:
            continue
        
        elapsed = time.time() - start_time
        
        try:
            result = run_single_backtest(
                products, z_e, tp, sl, te,
                start_date=None,          # 全量或训练集起始
                end_date=train_end,      # 关键：只用到训练集截止日
            )
            
            if result:
                score = calc_score(result)
                result["_score"] = score
                result["_combo_id"] = combo_idx
                all_results.append(result)
                
                s = result["summary"]
                print(
                    f"  [{combo_idx}/{total_combos}] "
                    f"z={z_e:.2f} TP={tp:.1f} SL={sl:.1f} T={te} | "
                    f"PnL={s['total_pnl']:+,.0f}Y "
                    f"Trd={s['total_trades']} "
                    f"WR={s['win_rate']:.1f}% "
                    f"Sh={s['sharpe']:.2f} "
                    f"DD={s['max_dd']:.1f}% "
                    f"Ann={s['ann_ret']:+.1f}% "
                    f"Score={score:.1f}"
                )
            else:
                print(
                    f"  [{combo_idx}/{total_combos}] "
                    f"z={z_e:.2f} TP={tp:.1f} SL={sl:.1f} T={te} | "
                    f"(无有效结果)"
                )
        except Exception as e:
            print(
                f"  [{combo_idx}/{total_combos}] "
                f"z={z_e:.2f} TP={tp:.1f} SL={sl:.1f} T={te} | "
                f"ERROR: {e}"
            )
    
    total_time = time.time() - start_time
    
    # 按 score 排序
    all_results.sort(key=lambda x: x.get("_score", -999), reverse=True)
    
    print("\n" + "=" * 70)
    print(f"  扫描完成！共 {len(all_results)} 组有效参数 | 耗时 {total_time:.1f}s")
    print(f"  数据: ccfx 5min ({global_start.strftime('%Y-%m-%d')} ~ {global_end.strftime('%Y-%m-%d')})")
    if has_oos:
        print(f"  训练集: ~{train_end} | 样本外: {oos_start}~{oos_end}")
    print("=" * 70)
    
    # ---- 样本外验证 ----
    oos_results = []
    if has_oos and all_results:
        oos_top_k = min(top_n, len(all_results))  # 验证TOP-K组参数
        print(f"\n{'='*70}")
        print(f"  [样本外验证] 用 TOP-{oos_top_k} 参数在 {oos_start} ~ {oos_end} 上验证")
        print(f"{'='*70}")
        
        for i in range(oos_top_k):
            r = all_results[i]
            p = r["params"]
            
            try:
                oos_result = run_single_backtest(
                    products,
                    z_entry=p["z_entry"],
                    tp_pts=p["tp_pts"],
                    sl_pts=p["sl_pts"],
                    time_exit=p["time_exit"],
                    start_date=oos_start,
                    end_date=oos_end,
                )
                
                if oos_result:
                    oos_score = calc_score(oos_result)
                    oos_result["_is_score"] = oos_score  # IS = In-Sample
                    oos_result["_oos_score"] = oos_score  # OOS = Out-of-Sample
                    oos_result["_is_rank"] = i + 1
                    oos_result["_is_total_pnl"] = r["summary"]["total_pnl"]
                    oos_result["_is_ann_ret"] = r["summary"]["ann_ret"]
                    oos_result["_is_sharpe"] = r["summary"]["sharpe"]
                    oos_result["_is_win_rate"] = r["summary"]["win_rate"]
                    
                    os_ = oos_result["summary"]
                    decay_ratio = 0.0
                    if r["summary"]["ann_ret"] != 0:
                        decay_ratio = (os_["ann_ret"] - r["summary"]["ann_ret"]) / abs(r["summary"]["ann_ret"]) * 100
                    
                    print(
                        f"  [OOS #{i+1:>2}] z={p['z_entry']:.2f} TP={p['tp_pts']:.1f} "
                        f"SL={p['sl_pts']:.1f} T={p['time_exit']} | "
                        f"PnL={os_['total_pnl']:+,.0f}Y "
                        f"Trd={os_['total_trades']} "
                        f"WR={os_['win_rate']:.1f}% "
                        f"Sh={os_['sharpe']:.2f} "
                        f"DD={os_['max_dd']:.1f}% "
                        f"Ann={os_['ann_ret']:+.1f}% "
                        f"Decay={decay_ratio:+.1f}%"
                    )
                    oos_results.append(oos_result)
                else:
                    print(
                        f"  [OOS #{i+1:>2}] z={p['z_entry']:.2f} ... | (无交易)"
                    )
            except Exception as e:
                print(f"  [OOS #{i+1:>2}] ERROR: {e}")
        
        # 样本外汇总
        if oos_results:
            oos_winning = sum(1 for r in oos_results if r["summary"]["total_pnl"] > 0)
            avg_oos_pnl = np.mean([r["summary"]["total_pnl"] for r in oos_results])
            avg_is_pnl = np.mean([r["_is_total_pnl"] for r in oos_results])
            print(f"\n  [OOS汇总] {len(oos_results)} 组中 {oos_winning} 组盈利 | "
                  f"平均IS PnL={avg_is_pnl:+,.0f}Y | 平均OOS PnL={avg_oos_pnl:+,.0f}Y")
    
    return all_results, oos_results


# ------------------------------------------------------------------
# 输出 TOP-N 控制台表格
# ------------------------------------------------------------------

def print_top_n(results: List[dict], top_n: int = 20):
    """打印排名表格"""
    
    header = (
        f"{'Rank':<5} {'Score':>6} {'Z_in':>5} {'TP':>4} {'SL':>4} "
        f"{'Time':>4} {'PnL(Y)':>9} {'Trd':>4} {'WR%':>5} "
        f"{'Sh':>5} {'DD%':>6} {'Ann%':>7} {'PF':>5}"
    )
    print(f"\n{'=' * 95}")
    print(f"  TOP-{min(top_n, len(results))} 最优参数组合 (综合评分)")
    print(f"{'=' * 95}")
    print(header)
    print("-" * 95)
    
    for i, r in enumerate(results[:top_n]):
        p = r["params"]
        s = r["summary"]
        print(
            f"{i+1:<5} {r['_score']:>6.1f} {p['z_entry']:>5.2f} {p['tp_pts']:>4.1f} "
            f"{p['sl_pts']:>4.1f} {p['time_exit']:>4} "
            f"{s['total_pnl']:>+9,.0f} {s['total_trades']:>4} "
            f"{s['win_rate']:>5.1f} {s['sharpe']:>5.2f} "
            f"{s['max_dd']:>6.2f} {s['ann_ret']:>+7.1f} "
            f"{s['profit_factor']:>5.2f}"
        )
    
    print("-" * 95)


def print_oos_comparison(oos_results: List[dict]):
    """打印样本内外对比表格"""
    if not oos_results:
        return
    
    header = (
        f"{'Rank':<5} {'Z_in':>5} {'TP':>4} {'SL':>4} {'Time':>4} "
        f"{'IS_PnL':>9} {'IS_Ann':>7} {'IS_Sh':>5} "
        f"{'OOS_PnL':>9} {'OOS_Ann':>7} {'OOS_Sh':>5} "
        f"{'Decay%':>7}"
    )
    print(f"\n{'=' * 105}")
    print(f"  样本内(IS) vs 样本外(OOS) 对比")
    print(f"{'=' * 105}")
    print(header)
    print("-" * 105)
    
    for i, r in enumerate(oos_results):
        p = r["params"]
        is_s = {"total_pnl": r["_is_total_pnl"], "ann_ret": r["_is_ann_ret"],
                "sharpe": r["_is_sharpe"]}
        os_s = r["summary"]
        
        decay = 0.0
        if is_s["ann_ret"] != 0:
            decay = (os_s["ann_ret"] - is_s["ann_ret"]) / abs(is_s["ann_ret"]) * 100
        
        is_pnl_str = f"{is_s['total_pnl']:+,.0f}"
        oos_pnl_str = f"{os_s['total_pnl']:+,.0f}"
        
        print(
            f"{r['_is_rank']:<5} {p['z_entry']:>5.2f} {p['tp_pts']:>4.1f} "
            f"{p['sl_pts']:>4.1f} {p['time_exit']:>4} "
            f"{is_pnl_str:>9s} {is_s['ann_ret']:>+7.1f}% {is_s['sharpe']:>5.2f} "
            f"{oos_pnl_str:>9s} {os_s['ann_ret']:>+7.1f}% {os_s['sharpe']:>5.2f} "
            f"{decay:>+7.1f}%"
        )
    
    print("-" * 105)


# ------------------------------------------------------------------
# HTML 报告生成（增强版：包含样本外对比）
# ------------------------------------------------------------------

def generate_optimization_report(
    results: List[dict],
    products: List[str],
    output_path: str,
    oos_results: Optional[List[dict]] = None,
    train_end: Optional[str] = None,
    oos_start: Optional[str] = None,
    oos_end: Optional[str] = None,
):
    """生成参数优化 HTML 报告（含样本外对比）"""
    
    top_n = min(50, len(results))
    top = results[:top_n]
    
    has_oos = oos_results is not None and len(oos_results) > 0
    
    # ---- 排名表格行 ----
    table_rows = ""
    for i, r in enumerate(top):
        p = r["params"]
        s = r["summary"]
        cls = "row-top3" if i < 3 else ""
        pnl_cls = "positive" if s["total_pnl"] > 0 else "negative"
        dd_cls = "negative" if s["max_dd"] < -5 else ""
        
        # 查找对应的OOS结果
        oos_info = ""
        if has_oos and i < len(oos_results):
            oos_r = oos_results[i]
            oos_s = oos_r["summary"]
            oos_mark = "OOS+" if oos_s["total_pnl"] > 0 else "OOS-"
            oos_info = f'<td class="{"positive" if oos_s["total_pnl"]>0 else "negative"}">{oos_s["total_pnl"]/10000:+.2f}</td><td>{oos_s["total_trades"]}</td>'
        else:
            oos_info = '<td>-</td><td>-</td>'
        
        table_rows += f"""<tr class="{cls}">
<td>{i+1}</td>
<td>{r['_score']:.1f}</td>
<td>{p['z_entry']:.2f}</td>
<td>{p['tp_pts']:.1f}</td>
<td>{p['sl_pts']:.1f}</td>
<td>{p['time_exit']}</td>
<td class="{pnl_cls}">{s['total_pnl']/10000:+.2f}</td>
<td>{s['total_trades']}</td>
<td>{s['win_rate']:.1f}%</td>
<td>{s['sharpe']:.2f}</td>
<td class="{dd_cls}">{s['max_dd']:.2f}%</td>
<td class="{pnl_cls}">{s['ann_ret']:+.1f}%</td>
<td>{s['profit_factor']:.2f}</td>
<td>{s['avg_hold_hours']:.1f}h</td>
{oos_info}
</tr>\n"""
    
    # ---- TOP1 详细展示 ----
    best = results[0]
    bp = best["params"]
    bs = best["summary"]
    
    # 各品种明细
    prod_detail = ""
    for prod_name, ps in best.get("by_product", {}).items():
        prod_detail += f"""<div class="prod-mini">
<strong style="color:{_prod_color(prod_name)}">{prod_name}</strong>: 
{ps['trades']}笔 | {ps['ann_ret']} | DD={ps['max_dd']} | Sh={ps['sharpe']} | PnL={ps['pnl_yuan']}
</div>\n"""
    
    # ---- 参数分布热力图数据 ----
    z_vals = sorted(set(r["params"]["z_entry"] for r in results))
    tp_vals = sorted(set(r["params"]["tp_pts"] for r in results))
    
    heatmap_data = []
    for tp_v in tp_vals:
        row = []
        for z_v in z_vals:
            matches = [
                r for r in results
                if abs(r["params"]["tp_pts"] - tp_v) < 0.01
                and abs(r["params"]["z_entry"] - z_v) < 0.01
                and r["params"]["sl_pts"] == 8.0
                and r["params"]["time_exit"] == 288
            ]
            if matches:
                row.append(round(matches[0]["_score"], 1))
            else:
                row.append(None)
        heatmap_data.append(row)
    
    now_ts = datetime.now().strftime('%Y-%m-%d %H:%M')
    best_score = best['_score']
    neg_class_ann = "neg" if bs['ann_ret'] < 0 else ""

    # ---- 样本外对比表 ----
    oos_section = ""
    if has_oos:
        oos_rows = ""
        for i, r in enumerate(oos_results):
            p = r["params"]
            os_s = r["summary"]
            is_pnl = r["_is_total_pnl"]
            is_ann = r["_is_ann_ret"]
            is_sh = r["_is_sharpe"]
            decay_val = 0.0
            if is_ann != 0:
                decay_val = (os_s["ann_ret"] - is_ann) / abs(is_ann) * 100
            
            oos_cls_pnl = "positive" if os_s["total_pnl"] > 0 else "negative"
            is_pnl_cls = "positive" if is_pnl > 0 else "negative"
            
            oos_rows += f"""<tr>
<td>{i+1}</td>
<td>{p['z_entry']:.2f}</td>
<td>{p['tp_pts']:.1f}</td>
<td>{p['sl_pts']:.1f}</td>
<td>{p['time_exit']}</td>
<td class="{is_pnl_cls}">{is_pnl/10000:+.2f}</td>
<td>{is_ann:+.1f}%</td>
<td>{is_sh:.2f}</td>
<td class="{oos_cls_pnl}">{os_s['total_pnl']/10000:+.2f}</td>
<td>{os_s['ann_ret']:+.1f}%</td>
<td>{os_s['sharpe']:.2f}</td>
<td>{os_s['total_trades']}</td>
<td>{os_s['win_rate']:.1f}%</td>
<td class="{'negative' if decay_val < -30 else ''}">{decay_val:+.1f}%</td>
</tr>\n"""
        
        oos_section = f"""
<div class="card">
<h2>样本外(OOS)验证对比 (Top-{len(oos_results)})</h2>
<p class="subtitle">
训练集: ~{train_end} | 测试集: {oos_start} ~ {oos_end}<br>
<span style="color:#dc2626;font-weight:600;">Decay列：年化收益率衰减幅度。若OOS大幅衰减或转负，说明可能过拟合。</span>
</p>
<table>
<tr><th>Rank</th><th>Z_entry</th><th>TP(pt)</th><th>SL(pt)</th><th>Time</th>
<th>IS_PnL(万)</th><th>IS_Ann%</th><th>IS_Sharpe</th>
<th>OOS_PnL(万)</th><th>OOS_Ann%</th><th>OOS_Sharpe</th>
<th>OOS_Trd</th><th>OOS_WR%</th><th>Decay%</th></tr>
{oos_rows}
</table>
</div>"""

    parts = []

    # CSS 样式
    css = """body{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
margin:16px;background:#f8fafc;}
.card{background:white;border-radius:12px;padding:24px;margin:16px 0;
box-shadow:0 2px 12px rgba(0,0,0,.06);}
h1{color:#0f172a;text-align:center;font-size:24px;}
h2{color:#1e293b;border-bottom:2px solid #e2e8f0;padding-bottom:8px;}
.subtitle{text-align:center;color:#64748b;margin-bottom:24px;}
table{width:100%;border-collapse:collapse;font-size:13px;}
th{background:#1e293b;color:white;padding:10px 8px;text-align:center;font-weight:600;}
td{padding:8px;border-bottom:1px solid #f1f5f9;text-align:center;}
tr:hover{background:#f8fafc;}
.row-top3{background:#fefce8 !important;font-weight:600;}
.positive{color:#dc2626;font-weight:600;}
.negative{color:#16a34a;font-weight:600;}
.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
gap:12px;margin:20px 0;}
.metric-card{background:linear-gradient(135deg,#2563eb,#7c3aed);color:white;
border-radius:12px;padding:18px;text-align:center;}
.metric-card.neg{background:linear-gradient(135deg,#dc2626,#991b1b);}
.metric-val{font-size:32px;font-weight:700;}
.metric-label{font-size:12px;opacity:0.85;margin-top:4px;}
.prod-mini{padding:8px;background:#f1f5f9;border-radius:6px;margin:4px 0;font-size:13px;}
.best-box{background:#eff6ff;border-left:4px solid #2563eb;padding:16px;
border-radius:0 8px 8px 0;margin:16px 0;}
.param-tag{display:inline-block;background:#dbeafe;color:#1e40af;
padding:2px 10px;border-radius:4px;font-size:13px;margin:2px;font-weight:600;}
.chart-row{display:flex;gap:16px;flex-wrap:wrap;}
.chart-container{flex:1;min-width:400px;height:350px;}
.section-title{font-size:14px;color:#475569;margin:12px 0 8px;font-weight:600;}
.oos-badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;
margin-left:4px;}
.oos-good{background:#dcfce7;color:#166534;}
.oos-bad{background:#fee2e2;color:#991b1b;}"""

    data_label = f"""ccfx 5min | {', '.join(products)} | {len(results)} 组"""
    if has_oos:
        data_label += f" | IS:~{train_end} | OOS:{oos_start}~{oos_end}"

    parts.append(f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>基差套利参数优化报告{" (含样本外验证)" if has_oos else ""}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
{css}
</style></head>
<body>

<div class="card">
<h1>基差套利参数优化报告{" (含样本外验证)" if has_oos else ""}</h1>
<p class="subtitle">
{now_ts} |
{data_label} |
<span style="color:#dc2626;">红色=盈利/好 | 绿色=亏损/坏</span>
</p>
</div>

<!-- 最佳参数 -->
<div class="card best-box">
<h2>TOP-1 最优参数组合 (Score={best_score:.1f}/100)</h2>
<div class="summary-grid">
<div class="metric-card"><div class="metric-val">z={bp['z_entry']}</div><div class="metric-label">入场阈值</div></div>
<div class="metric-card"><div class="metric-val">TP={bp['tp_pts']}pt</div><div class="metric-label">止盈点数</div></div>
<div class="metric-card"><div class="metric-val">SL={bp['sl_pts']}pt</div><div class="metric-label">止损点数</div></div>
<div class="metric-card"><div class="metric-val">T={bp['time_exit']}bars</div><div class="metric-label">时间止损(K线)</div></div>
<div class="metric-card"><div class="metric-val">{bs['total_pnl']/10000:+.2f}万</div><div class="metric-label">总盈亏</div></div>
<div class="metric-card {neg_class_ann}"><div class="metric-val">{bs['ann_ret']:+.1f}%</div><div class="metric-label">年化收益</div></div>
<div class="metric-card"><div class="metric-val">{bs['sharpe']:.2f}</div><div class="metric-label">Sharpe</div></div>
<div class="metric-card neg"><div class="metric-val">{bs['max_dd']:.2f}%</div><div class="metric-label">最大回撤</div></div>
<div class="metric-card"><div class="metric-val">{bs['win_rate']:.1f}%</div><div class="metric-label">胜率</div></div>
<div class="metric-card"><div class="metric-val">{bs['profit_factor']:.2f}</div><div class="metric-label">盈亏比</div></div>
<div class="metric-card"><div class="metric-val">{bs['total_trades']}</div><div class="metric-label">总交易笔数</div></div>
<div class="metric-card"><div class="metric-val">{bs['avg_hold_hours']:.1f}h</div><div class="metric-label">平均持仓</div></div>
</div>

<h3 class="section-title">各品种表现</h3>
{prod_detail}
</div>

<!-- 排名表格 -->
<div class="card">
<h2>参数排名 TOP-{top_n}</h2>
<table>
<tr><th>Rank</th><th>Score</th><th>Z_entry</th><th>TP(pt)</th><th>SL(pt)</th>
<th>Time(bars)</th><th>PnL(万)</th><th>Trades</th><th>WinRate</th>
<th>Sharpe</th><th>MaxDD%</th><th>AnnRet%</th><th>P/F</th><th>AvgHold</th>
<th>{"OOS_PnL(万)" if has_oos else ""}</th><th>{"OOS_Trd" if has_oos else ""}</th>
</tr>
{table_rows}
</table>
</div>{oos_section}""")

    # JS 部分
    js_scores = json.dumps([r['_score'] for r in results])
    js_scatter = json.dumps([
        {"x": r["summary"]["sharpe"], "y": r["summary"]["ann_ret"],
         "score": r["_score"], "z": r["params"]["z_entry"]}
        for r in results[:100]
    ])
    js_z_vals = json.dumps(z_vals)
    js_tp_vals = json.dumps(tp_vals)
    js_heatmap = json.dumps(heatmap_data)

    js_code = """
<script>
const scores = SCORES_PLACEHOLDER;
const histCtx = document.getElementById('chart_score_dist').getContext('2d');
new Chart(histCtx, {
    type: 'bar',
    data: {
        labels: scores.map((_,i) => i+1),
        datasets: [{
            label: '综合评分',
            data: scores,
            backgroundColor: scores.map(s => s >= 60 ? '#2563eb' : s >= 40 ? '#f59e0b' : '#ef4444'),
            borderRadius: 2,
        }]
    },
    options: { responsive:true, plugins:{legend:{display:false},
        title:{display:true,text:'参数组合评分分布 (TOP-" + str(top_n) + ")'}},
        scales:{y:{beginAtZero:true,title:{display:true,text:'Score'}}}} }
});

const scatterData = SCATTER_PLACEHOLDER;
const scCtx = document.getElementById('chart_scatter').getContext('2d');
new Chart(scCtx, {
    type: 'scatter',
    data: {
        datasets: [{
            label: '参数组合',
            data: scatterData.map(d => ({x: d.x, y: d.y, score: d.score})),
            backgroundColor: scatterData.map(d =>
                d.score > 60 ? 'rgba(37,99,235,0.7)' : d.score > 40 ? 'rgba(245,158,11,0.6)' : 'rgba(239,68,68,0.5)'),
            pointRadius: scatterData.map(d => Math.max(4, d.score / 10)),
        }]
    },
    options: { responsive:true, plugins:{legend:{display:false},
        title:{display:true,text:'Sharpe vs 年化收益率 (气泡大小=Score)'}},
        scales:{x:{title:{display:true,text:'Sharpe'}}, y:{title:{display:true,text:'年化收益%'}}}} }
});

const hmLabelsZ = Z_VALS_PLACEHOLDER;
const hmLabelsTP = TP_VALS_PLACEHOLDER;
const hmData = HEATMAP_PLACEHOLDER;
const hmCtx = document.getElementById('chart_heatmap').getContext('2d');
function toColor(v) {
    if(v==null) return 'transparent';
    if(v>=60) return 'rgba(37,99,235,' + (0.3+(v-60)/40*0.7) + ')';
    if(v>=40) return 'rgba(245,158,11,' + (0.3+(v-40)/20*0.7) + ')';
    return 'rgba(239,68,68,' + (0.3+v/40*0.5) + ')';
}
const flatHM = [];
for(let r=0;r<hmData.length;r++) for(let c=0;c<hmData[r].length;c++)
    flatHM.push({x:c,y:r,v:hmData[r][c]});
new Chart(hmCtx, {
    type: 'matrix',
    data: {
        labels: hmLabelsZ,
        datasets: [{
            label: 'Score',
            data: flatHM.map(d => ((d.v||0)+100).toFixed(1)),
            backgroundColor: flatHM.map(d => toColor(d.v)),
            width: (ctx) => 36,
            height: (ctx) => 28,
        }]
    },
    options: { responsive:true, plugins:{legend:{display:false},
        title:{display:true,text:'参数热力图: Z_entry vs TP (SL=8, Time=288)'}},
        scales:{x:{labels:hmLabelsZ,title:{display:true,text:'Z_entry'}},
        y:{labels:hmLabelsTP,title:{display:true,text:'TP (pt)'}}} } }
});
</script>""".replace("SCORES_PLACEHOLDER", js_scores).replace(
    "SCATTER_PLACEHOLDER", js_scatter).replace(
    "Z_VALS_PLACEHOLDER", js_z_vals).replace(
    "TP_VALS_PLACEHOLDER", js_tp_vals).replace(
    "HEATMAP_PLACEHOLDER", js_heatmap)

    parts.append("""
<!-- 图表区域 -->
<div class="card chart-row">
<div class=\"chart-container\"><canvas id=\"chart_score_dist\"></canvas></div>
<div class=\"chart-container\"><canvas id=\"chart_scatter\"></canvas></div>
</div>
<div class="card chart-row\">
<div class=\"chart-container\"><canvas id=\"chart_heatmap\"></canvas></div>
</div>
""")

    parts.append(js_code)
    parts.append("\n</body></html>")

    html = "\n".join(parts)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"\nOptimization report: {output_path}")


def _prod_color(p: str) -> str:
    return {"IF":"#dc2626", "IH":"#2563eb", "IC":"#059669", "IM":"#7c3aed"}.get(p, "#64748b")


# ------------------------------------------------------------------
# CSV 导出（增强版：包含OOS字段）
# ------------------------------------------------------------------

def export_csv(results: List[dict], output_path: str,
               oos_results: Optional[List[dict]] = None):
    """导出全部结果为 CSV"""
    rows = []
    for i, r in enumerate(results):
        p = r["params"]
        s = r["summary"]
        row = {
            "rank": i + 1,
            "score": round(r["_score"], 2),
            "z_entry": p["z_entry"],
            "tp_pts": p["tp_pts"],
            "sl_pts": p["sl_pts"],
            "time_exit_bars": p["time_exit"],
            "total_pnl": s["total_pnl"],
            "total_trades": s["total_trades"],
            "win_rate": round(s["win_rate"], 2),
            "sharpe": round(s["sharpe"], 3),
            "max_dd": round(s["max_dd"], 3),
            "ann_ret": round(s["ann_ret"], 2),
            "profit_factor": round(s["profit_factor"], 3),
            "avg_hold_hours": round(s["avg_hold_hours"], 2),
        }
        
        # 附加OOS字段
        if oos_results and i < len(oos_results):
            oos_r = oos_results[i]
            os_s = oos_r["summary"]
            row.update({
                "oos_pnl": os_s["total_pnl"],
                "oos_trades": os_s["total_trades"],
                "oos_win_rate": round(os_s["win_rate"], 2),
                "oos_sharpe": round(os_s["sharpe"], 3),
                "oos_max_dd": round(os_s["max_dd"], 3),
                "oos_ann_ret": round(os_s["ann_ret"], 2),
                "oos_profit_factor": round(os_s.get("profit_factor", 0), 3),
            })
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"CSV exported: {output_path}")


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="基差套利参数优化器 (支持样本外测试)")
    parser.add_argument("--products", nargs="+", default=["IF","IH","IC","IM"])
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--fast", action="store_true",
                       help="快速模式（减少参数组合数）")
    parser.add_argument("--oos-months", type=int, default=0,
                       help="样本外测试月数（如1表示最后1个月留作测试）")
    parser.add_argument("--train-end", type=str, default=None,
                       help="手动指定训练集截止日 (YYYY-MM-DD)，与--oos互斥")
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()
    
    results, oos_results = run_optimization(
        products=args.products,
        top_n=args.top_n,
        fast=args.fast,
        oos_months=args.oos_months,
    )
    
    if not results:
        print("无有效结果！")
        return
    
    # 输出
    print_top_n(results, args.top_n)
    
    if oos_results:
        print_oos_comparison(oos_results)
    
    rpt = args.report or os.path.join(
        REPORT_DIR, f"param_optimize_{datetime.now():%Y%m%d_%H%M%S}.html"
    )
    generate_optimization_report(
        results, args.products, rpt,
        oos_results=oos_results if oos_results else None,
    )
    
    csv_path = args.csv or os.path.join(
        REPORT_DIR, f"param_optimize_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    export_csv(results, csv_path, oos_results=oos_results)


if __name__ == "__main__":
    main()
