"""
basis_arbitrage_backtest.py v2
纯基差回归套利回测引擎（多合约对扫描，无方向判断，完全对冲）

v2 核心升级：
  - 每个时刻扫描所有可用合约对（C1-C2, C2-C3, C3-C4...）
  - 选 |z-score| 最大的那对来交易（偏离最极端 = 回归预期最强）
  - 同一品种同时最多持有一个仓位
  
策略逻辑：
  - 不判断涨跌方向
  - 每次建仓都是对冲仓位（多合约A+空合约B / 反过来）
  - 只交易基差偏离 → 回归的过程

入场：
  - 某合约对的 spread z-score 超过 ±z_entry → 开仓

出场：
  - 止盈 / 止损 / 时间止损 / z-score 回归

数据源：ccfx 5 分钟 K 线 CSV
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")


# ------------------------------------------------------------------
# 数据结构
# ------------------------------------------------------------------

class TradeSide(Enum):
    SHORT_BASIS = "做空基差"    # 多近+空远 (赌 spread 收窄)
    LONG_BASIS  = "做多基差"     # 空近+多远 (赌 spread 扩大)


class ExitReason(Enum):
    TAKE_PROFIT = "止盈"
    STOP_LOSS   = "止损"
    TIME_EXIT   = "时间止损"
    ZSCORE_EXIT = "Z-Score回归"
    EXPIRY      = "交割到期"


@dataclass
class TradeRecord:
    """单笔交易记录"""
    trade_id: int
    product: str
    pair_key: str               # 合约对标识，如 "IF2509-IF2512"
    side: TradeSide
    entry_time: datetime
    exit_time: Optional[datetime] = None
    leg_a_symbol: str = ""      # 合约A代码（近端/价格较低端）
    leg_b_symbol: str = ""      # 合约B代码（远端/价格较高端）
    entry_leg_a_price: float = 0.0
    entry_leg_b_price: float = 0.0
    entry_spread: float = 0.0        # 入场 B价 - A价 (raw_spread)
    entry_basis_rate: float = 0.0
    entry_zscore: float = 0.0
    exit_leg_a_price: float = 0.0
    exit_leg_b_price: float = 0.0
    exit_spread: float = 0.0
    pnl_points: float = 0.0          # 点数盈亏
    pnl_rmb: float = 0.0
    volume: int = 1
    hold_bars: int = 0
    exit_reason: Optional[ExitReason] = None
    commission: float = 0.0


@dataclass
class BacktestConfig:
    """回测参数"""
    z_entry: float = 1.5
    lookback_bars: int = 288         # rolling 窗口
    tp_basis_pts: float = 3.0        # 止盈(点)
    sl_basis_pts: float = 8.0        # 止损(点)
    time_exit_bars: int = 288        # 时间止损(K线数)
    z_exit: float = 0.2              # z-score 出场阈值
    commission_rate: float = 0.000023
    slippage_ticks: float = 0.2
    initial_capital: float = 1_000_000
    volume_per_trade: int = 1
    multipliers: Dict[str, int] = field(default_factory=lambda: {
        "IF": 300, "IH": 300, "IC": 200, "IM": 200,
    })
    max_concurrent_pairs: int = 1    # 同一品种最大同时持仓对数
    # 样本内/外分割
    start_date: Optional[str] = None   # 数据起始日期 YYYY-MM-DD
    end_date: Optional[str] = None     # 数据截止日期 YYYY-MM-DD


# ------------------------------------------------------------------
# 数据加载
# ------------------------------------------------------------------

def load_ccfx_data(product: str,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> pd.DataFrame:
    """加载 ccfx 5分钟基差数据

    Args:
        product: 品种代码 (IF/IH/IC/IM)
        start_date: 起始日期过滤 (YYYY-MM-DD)，包含当天
        end_date: 截止日期过滤 (YYYY-MM-DD)，包含当天
    """
    filename = f"5min_basis_{product}_ccfx_20260425.csv"
    filepath = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        logger.warning(f"[数据] 文件不存在: {filepath}")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # 日期范围过滤
    if start_date:
        df = df[df["datetime"] >= start_date]
    if end_date:
        df = df[df["datetime"] <= end_date]
    
    required_cols = ["near_close", "far_close", "raw_spread", 
                     "adj_annualized_rate", "near_symbol", "far_symbol"]
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"[数据] 缺少列: {col}")
            return pd.DataFrame()
    
    df = df.dropna(subset=required_cols)
    
    # 构建合约对标识
    df["pair_key"] = df["near_symbol"] + "-" + df["far_symbol"]
    
    logger.info(
        f"[数据] {product}: {len(df)} 条5分钟记录 "
        f"({df['datetime'].min() if len(df) else 'N/A'} ~ "
        f"{df['datetime'].max() if len(df) else 'N/A'}) "
        f"{df['pair_key'].nunique()} 个合约对"
    )
    return df


# ------------------------------------------------------------------
# v2 核心：多合约对扫描引擎
# ------------------------------------------------------------------

class BasisArbitrageBacktesterV2:
    """
    多合约对纯基差回归套利回测。
    
    每个时刻：
      1. 对每个合约对的 spread 计算 z-score（各自独立 rolling）
      2. 选 |z-score| 最大且超过阈值的那一对开仓
      3. 已有仓位则只跟踪该对的出场条件
    
    优势：
      - C1-C2 的基差可能不极端，但 C2-C4 的基差可能很极端
      - 自动捕捉任意相邻/跨月合约间的定价偏差
    """

    def __init__(self, config: BacktestConfig):
        self.cfg = config
        self.trades: List[TradeRecord] = []

    def run(self, products: List[str]) -> dict:
        all_results = {}
        for product in products:
            result = self._run_single(product)
            if result:
                all_results[product] = result
        return self._aggregate_results(all_results)

    def _run_single(self, product: str) -> Optional[dict]:
        """单个品种回测 —— 多合约对扫描版"""
        
        df = load_ccfx_data(
            product,
            start_date=self.cfg.start_date,
            end_date=self.cfg.end_date,
        )
        if df.empty:
            return None
        
        multiplier = self.cfg.multipliers.get(product, 300)
        trades: List[TradeRecord] = []
        
        # ---- 按 pair_key 分组计算独立 rolling z-score ----
        pairs = sorted(df["pair_key"].unique())
        
        # 给每列加上 pair 前缀的 spread 和 z-score
        dfs_by_pair: Dict[str, pd.DataFrame] = {}
        for pk in pairs:
            mask = df["pair_key"] == pk
            sub = df.loc[mask].copy()
            
            lookback = self.cfg.lookback_bars
            min_p = min(60, len(sub) // 10)  # 动态最小样本
            
            sub["spread_mean"] = sub["raw_spread"].rolling(
                window=lookback, min_periods=min_p).mean()
            sub["spread_std"] = sub["raw_spread"].rolling(
                window=lookback, min_periods=min_p).std()
            sub["zscore"] = (
                (sub["raw_spread"] - sub["spread_mean"]) 
                / sub["spread_std"].replace(0, np.nan)
            )
            dfs_by_pair[pk] = sub
        
        # ---- 主循环：按时间遍历 ----
        # 收集所有时间点
        all_times = np.sort(np.array(df["datetime"].unique()))
        
        position_open = False
        current_trade: Optional[TradeRecord] = None
        trade_id = 0
        
        equity_series = []
        times_series = []
        
        for ts in all_times:
            best_signal = None  # (pair_key, row, abs_z)
            
            # ====== 空仓时：扫描所有合约对找最佳机会 ======
            if not position_open:
                for pk in pairs:
                    sub = dfs_by_pair[pk]
                    idx = sub.index[sub["datetime"] == ts]
                    if len(idx) == 0:
                        continue
                    row = sub.loc[idx[0]]
                    z = row.get("zscore", np.nan)
                    
                    if pd.isna(z):
                        continue
                    
                    abs_z = abs(z)
                    if abs_z > self.cfg.z_entry:
                        if best_signal is None or abs_z > best_signal[2]:
                            best_signal = (pk, row, z)
                
                # 如果找到信号，选最强的一个开仓
                if best_signal:
                    pk, row, z_val = best_signal
                    trade_id += 1
                    
                    side = TradeSide.SHORT_BASIS if z_val > 0 else TradeSide.LONG_BASIS
                    
                    current_trade = TradeRecord(
                        trade_id=trade_id,
                        product=product,
                        pair_key=pk,
                        side=side,
                        entry_time=ts,
                        leg_a_symbol=row["near_symbol"],
                        leg_b_symbol=row["far_symbol"],
                        entry_leg_a_price=row["near_close"],
                        entry_leg_b_price=row["far_close"],
                        entry_spread=row["raw_spread"],
                        entry_basis_rate=row.get("adj_annualized_rate", 0),
                        entry_zscore=z_val,
                        volume=self.cfg.volume_per_trade,
                    )
                    position_open = True
            
            # ====== 有持仓时：跟踪该对的出场条件 ======
            else:
                assert current_trade is not None
                pk = current_trade.pair_key
                sub = dfs_by_pair[pk]
                
                idx = sub.index[sub["datetime"] == ts]
                if len(idx) == 0:
                    # 该合约对今天没有数据（可能换月了），强制平仓
                    self._close_trade(current_trade, ts, 
                                     current_trade.entry_leg_a_price,
                                     current_trade.entry_leg_b_price,
                                     current_trade.entry_spread,
                                     0.0, multiplier, ExitReason.EXPIRY)
                    trades.append(current_trade)
                    current_trade = None
                    position_open = False
                    equity_series.append(
                        self.cfg.initial_capital + sum(t.pnl_rmb for t in trades))
                    times_series.append(ts)
                    continue
                
                row = sub.loc[idx[0]]
                spread = row["raw_spread"]
                z = row.get("zscore", np.nan)
                near_c = row["near_close"]
                far_c = row["far_close"]
                
                current_trade.hold_bars += 1
                
                spread_change = spread - current_trade.entry_spread
                
                if current_trade.side == TradeSide.SHORT_BASIS:
                    float_pnl_pts = -spread_change
                else:
                    float_pnl_pts = spread_change
                
                closed = False
                
                # --- 止盈 ---
                if not closed and float_pnl_pts >= self.cfg.tp_basis_pts:
                    self._close_trade(current_trade, ts, near_c, far_c, spread,
                                       float_pnl_pts, multiplier, ExitReason.TAKE_PROFIT)
                    trades.append(current_trade)
                    closed = True
                
                # --- 止损 ---
                if not closed and float_pnl_pts <= -self.cfg.sl_basis_pts:
                    self._close_trade(current_trade, ts, near_c, far_c, spread,
                                       float_pnl_pts, multiplier, ExitReason.STOP_LOSS)
                    trades.append(current_trade)
                    closed = True
                
                # --- 时间止损 ---
                if not closed and current_trade.hold_bars >= self.cfg.time_exit_bars:
                    self._close_trade(current_trade, ts, near_c, far_c, spread,
                                       float_pnl_pts, multiplier, ExitReason.TIME_EXIT)
                    trades.append(current_trade)
                    closed = True
                
                # --- Z-Score 回归 ---
                if not closed and not pd.isna(z):
                    if (current_trade.side == TradeSide.SHORT_BASIS and z < self.cfg.z_exit) or \
                       (current_trade.side == TradeSide.LONG_BASIS and z > -self.cfg.z_exit):
                        self._close_trade(current_trade, ts, near_c, far_c, spread,
                                           float_pnl_pts, multiplier, ExitReason.ZSCORE_EXIT)
                        trades.append(current_trade)
                        closed = True
                
                if closed:
                    current_trade = None
                    position_open = False
                    equity_series.append(
                        self.cfg.initial_capital + sum(t.pnl_rmb for t in trades))
                else:
                    # 浮动权益
                    if current_trade.side == TradeSide.SHORT_BASIS:
                        fp = -(spread - current_trade.entry_spread) * multiplier * current_trade.volume
                    else:
                        fp = (spread - current_trade.entry_spread) * multiplier * current_trade.volume
                    equity_series.append(
                        self.cfg.initial_capital + sum(t.pnl_rmb for t in trades) + fp)
                
                times_series.append(ts)
                continue
            
            # 空仓时的权益记录
            if not position_open:
                equity_series.append(
                    self.cfg.initial_capital + sum(t.pnl_rmb for t in trades))
                times_series.append(ts)
        
        # 强制平最后一笔
        if position_open and current_trade:
            last_ts = all_times[-1]
            spread = current_trade.entry_spread  # 用入场spread兜底
            sc = current_trade.entry_spread
            if current_trade.side == TradeSide.SHORT_BASIS:
                pts = 0.0
            else:
                pts = 0.0
            self._close_trade(
                current_trade, last_ts,
                current_trade.entry_leg_a_price,
                current_trade.entry_leg_b_price, sc,
                pts, multiplier, ExitReason.EXPIRY
            )
            trades.append(current_trade)
        
        equity_arr = np.array(equity_series) if equity_series else np.array([self.cfg.initial_capital])
        
        result = {
            "product": product,
            "trades": trades,
            "equity_curve": equity_arr,
            "equity_times": times_series,
            "total_bars": len(df),
            "pairs_traded": list(set(t.pair_key for t in trades)),
            "date_range": (df["datetime"].min(), df["datetime"].max()),
            "stats": self._calc_stats(trades, equity_arr, multiplier),
        }
        
        self._print_product_summary(result)
        self.trades.extend(trades)
        return result

    def _close_trade(self, trade: TradeRecord, ts: datetime,
                     a_price: float, b_price: float, spread: float,
                     pnl_pts: float, multiplier: int, reason: ExitReason):
        trade.exit_time = ts
        trade.exit_leg_a_price = a_price
        trade.exit_leg_b_price = b_price
        trade.exit_spread = spread
        trade.pnl_points = pnl_pts
        trade.exit_reason = reason
        
        gross_pnl = pnl_pts * multiplier * trade.volume
        avg_price = (trade.entry_leg_a_price + trade.entry_leg_b_price + a_price + b_price) / 4
        commission = avg_price * multiplier * trade.volume * self.cfg.commission_rate * 4
        slippage = self.cfg.slippage_ticks * multiplier * trade.volume * 2
        trade.pnl_rmb = gross_pnl - commission - slippage
        trade.commission = commission + slippage

    def _calc_stats(self, trades: List[TradeRecord], 
                    equity: np.ndarray, multiplier: int) -> dict:
        if not trades:
            return {"total_trades": 0}
        
        total_pnl = sum(t.pnl_rmb for t in trades)
        winning = [t for t in trades if t.pnl_rmb > 0]
        losing = [t for t in trades if t.pnl_rmb <= 0]
        
        win_rate = len(winning) / max(len(trades), 1) * 100
        avg_win = np.mean([t.pnl_rmb for t in winning]) if winning else 0
        avg_loss = np.mean([abs(t.pnl_rmb) for t in losing]) if losing else 1e-6
        profit_factor = sum(t.pnl_rmb for t in winning) / max(sum(abs(t.pnl_rmb) for t in losing), 1)
        
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / np.maximum(peak, 1e-6) * 100
        max_dd = drawdown.min()
        
        avg_hold = np.mean([t.hold_bars for t in trades]) if trades else 0
        avg_hold_hours = avg_hold * 5 / 60
        
        by_reason = {}
        for t in trades:
            r = t.exit_reason.value if t.exit_reason else "?"
            by_reason[r] = by_reason.get(r, 0) + 1
        
        total_return = total_pnl / self.cfg.initial_capital * 100
        if trades:
            time_span = (max(t.exit_time for t in trades) - 
                        min(t.entry_time for t in trades))
            time_span = max(time_span / np.timedelta64(1, 'D'), 1)
            annualized_return = total_return * (365 / time_span)
        else:
            annualized_return = 0
        
        daily_eq_diff = np.diff(equity) if len(equity) > 1 else np.array([0])
        if len(daily_eq_diff) > 10 and np.std(daily_eq_diff) > 1e-6:
            sharpe = np.sqrt(288) * daily_eq_diff.mean() / np.std(daily_eq_diff)
        else:
            sharpe = 0.0
        
        short_t = [t for t in trades if t.side == TradeSide.SHORT_BASIS]
        long_t = [t for t in trades if t.side == TradeSide.LONG_BASIS]
        
        # 按合约对统计
        by_pair = {}
        for t in trades:
            pk = t.pair_key
            if pk not in by_pair:
                by_pair[pk] = {"cnt": 0, "pnl": 0.0}
            by_pair[pk]["cnt"] += 1
            by_pair[pk]["pnl"] += t.pnl_rmb
        top_pairs = sorted(by_pair.items(), key=lambda x: x[1]["pnl"], reverse=True)[:5]
        
        return {
            "总交易次数": len(trades),
            "总收益率": f"{total_return:+.2f}%",
            "年化收益率": f"{annualized_return:+.2f}%",
            "总盈亏": f"{total_pnl:+,.0f}元",
            "最大回撤": f"{max_dd:.2f}%",
            "Sharpe比率": f"{sharpe:.2f}",
            "胜率": f"{win_rate:.1f}%",
            "盈利次数": len(winning),
            "亏损次数": len(losing),
            "平均盈利": f"{avg_win:+,.0f}元",
            "平均亏损": f"{avg_loss:+,.0f}元",
            "盈亏比": f"{profit_factor:.2f}",
            "平均持仓时长": f"{avg_hold_hours:.1f}小时({avg_hold:.0f}根K线)",
            "做空基次数": len(short_t),
            "做多基次数": len(long_t),
            "做空均盈亏": f"{np.mean([t.pnl_rmb for t in short_t]):+.0f}元" if short_t else "N/A",
            "做多均盈亏": f"{np.mean([t.pnl_rmb for t in long_t]):+.0f}元" if long_t else "N/A",
            "出场分布": json.dumps(by_reason, ensure_ascii=False),
            "活跃合约对": f"{len(by_pair)} 个",
            "TOP3合约对": "; ".join(f"{k}({v['cnt']}次/{v['pnl']:+.0f}元)" for k,v in top_pairs[:3]),
        }

    def _print_product_summary(self, result: dict):
        stats = result["stats"]
        p = result["product"]
        dr = result["date_range"]
        pairs = result.get("pairs_traded", [])
        
        print(f"\n{'='*65}")
        print(f"  [{p}] 多合约对纯基差回归套利")
        print(f"  时间: {dr[0]} ~ {dr[1]} | K线: {result['total_bars']}")
        if pairs:
            print(f"  参与合约对: {', '.join(pairs)} ({len(pairs)}个)")
        print(f"{'='*65}")
        for k, v in stats.items():
            print(f"  {k:<18s}: {v}")
        print(f"{'-'*65}")

    def _aggregate_results(self, all_results: dict) -> dict:
        all_trades = []
        for r in all_results.values():
            all_trades.extend(r["trades"])
        
        total_pnl = sum(t.pnl_rmb for t in all_trades)
        winning = [t for t in all_trades if t.pnl_rmb > 0]
        
        # 全局合约对统计
        all_pairs = set(t.pair_key for t in all_trades)
        
        print(f"\n{'='*70}")
        print(f"  [汇总] 多品种 · 多合约对扫描")
        print(f"  品种: {', '.join(all_results.keys())}")
        print(f"  总交易: {len(all_trades)} 笔 | 涉及合约对: {len(all_pairs)} 个")
        print(f"  总盈亏: {total_pnl:+,.0f} 元 | 胜率: {len(winning)/max(len(all_trades),1)*100:.1f}%")
        print(f"{'='*70}")
        
        return {
            "by_product": all_results,
            "all_trades": all_trades,
            "all_pairs": all_pairs,
            "summary": {
                "total_pnl": total_pnl,
                "total_trades": len(all_trades),
                "win_rate": len(winning)/max(len(all_trades),1)*100,
            }
        }


# ------------------------------------------------------------------
# HTML 报告生成
# ------------------------------------------------------------------

def generate_html_report(results: dict, cfg: BacktestConfig, output_path: str):
    by_product = results["by_product"]
    
    product_cards = ""
    for p, r in sorted(by_product.items()):
        stats = r["stats"]
        eq = r["equity_curve"]
        times = r["equity_times"]
        
        step = max(1, len(eq) // 500)
        eq_times_pd = pd.to_datetime(times[::step])
        eq_labels = [t.strftime("%m-%d %H:%M") for t in eq_times_pd]
        eq_vals = [round(x, 2) for x in eq[::step]]
        
        trades_html = _build_trades_table_v2(r["trades"])
        card_id = f"chart_{p}"
        
        product_cards += f"""<div class="card">
<h2>{p} - {stats.get('总交易次数', 0)} 笔交易 ({stats.get('活跃合约对', '?')} 个合约对)</h2>
<table class="stats-table">
"""
        for k, v in stats.items():
            if k not in ("出场分布", "TOP3合约对"):
                cls = "positive" if ("+" in str(v) and ("%" in str(v) or "元" in str(v))) else ""
                product_cards += f'<tr><td>{k}</td><td class="{cls}">{v}</td></tr>'
        
        if "TOP3合约对" in stats:
            product_cards += f'<tr><td>TOP3合约对</td><td>{stats["TOP3合约对"]}</td></tr>'
        
        product_cards += f"</table>"
        product_cards += f'<div id="{card_id}" class="chart-container"></div>'
        product_cards += f"<h3>交易明细</h3>{trades_html}</div>\n"
        
        product_cards += f"""<script>
new Chart(document.getElementById('{card_id}'), {{
    type: 'line',
    data: {{
        labels: {json.dumps(eq_labels)},
        datasets: [{{
            label: '{p}',
            data: {json.dumps(eq_vals)},
            borderColor: '#{_color_for(p)}',
            backgroundColor: '#{_color_for(p)}15',
            fill: true, tension: 0.1, pointRadius: 0,
        }}]
    }},
    options: {{ responsive: true, plugins: {{legend:{{display:false}}}} }}
}});
</script>
"""
    
    params_rows = (
        f"<tr><td>Z-Score 入场</td><td>±{cfg.z_entry}σ (选|z|最大的对)</td></tr>"
        f"<tr><td>Z-Score 出场</td><td>±{cfg.z_exit}σ</td></tr>"
        f"<tr><td>止盈/止损</td><td>{cfg.tp_basis_pts}/{cfg.sl_basis_pts} 点</td></tr>"
        f"<tr><td>时间止损</td><td>{cfg.time_exit_bars} 根K线 ({cfg.time_exit_bars*5//60}h)</td></tr>"
        f"<tr><td>Rolling窗口</td><td>{cfg.lookback_bars} 根 ({cfg.lookback_bars*5//60}h)</td></tr>"
        f"<tr><td>成本</td><td>{cfg.commission_rate*10000:.2f}‱×4腿 + {cfg.slippage_ticks}tick×2腿滑点</td></tr>"
    )
    
    summary = results["summary"]
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>v2 多合约对纯基差回归套利报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{{font-family:-apple-system,"Segoe UI",sans-serif;margin:16px;background:#f0f2f5;}}
.card{{background:white;border-radius:10px;padding:24px;margin:16px 0;box-shadow:0 2px 8px rgba(0,0,0,.08);}}
h1{{color:#1a1a2e;text-align:center;}} h2,h3{{color:#16213e;}}
table{{width:100%;border-collapse:collapse;margin:8px 0;}}
.stats-table td{{padding:8px 12px;border-bottom:1px solid #eee;font-size:14px;}}
.stats-table td:first-child{{color:#666;width:45%;}}
.positive{{color:#dc2626;font-weight:600;}} .negative{{color:#16a34a;font-weight:600;}}
.trade-table{{font-size:12px;}} .trade-table th{{background:#f1f5f9;}}
.chart-container{{height:300px;margin:16px 0;}}
.param-card{{max-width:600px;}}
.summary-box{{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0;}}
.summary-item{{background:linear-gradient(135deg,#2563eb,#7c3aed);color:white;
    border-radius:10px;padding:20px;flex:1;min-width:150px;}}
.summary-item.neg{{background:linear-gradient(135deg,#dc2626,#991b1b);}}
.summary-num{{font-size:28px;font-weight:700;}} .summary-label{{font-size:13px;opacity:0.9;}}
.pair-tag{{display:inline-block;background:#e0e7ff;color:#3730a3;padding:2px 8px;
    border-radius:4px;font-size:12px;margin:2px;}}
</style></head>
<body>

<div class="card">
<h1>[v2] 多合约对扫描 · 纯基差回归套利</h1>
<p style="text-align:center;color:#666;">
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
自动选择 |z-score| 最大的合约对交易 | 无方向判断 · 完全对冲
</p>
</div>

<div class="card summary-box">
<div class="summary-item {'neg' if summary['total_pnl'] < 0 else ''}">
<div class="summary-num">¥{summary['total_pnl']:+,.0f}</div>
<div class="summary-label">总盈亏 ({len(by_product)}品种)</div>
</div>
<div class="summary-item">
<div class="summary-num">{summary['total_trades']}</div>
<div class="summary-label">总交易次数</div>
</div>
<div class="summary-item {'neg' if summary['win_rate'] < 50 else ''}">
<div class="summary-num">{summary['win_rate']:.1f}%</div>
<div class="summary-label">胜率</div>
</div>
<div class="summary-item">
<div class="summary-num">{len(results.get('all_pairs', []))}</div>
<div class="summary-label">参与合约对</div>
</div>
</div>

<div class="card param-card"><h3>参数</h3><table>{params_rows}</table></div>
{product_cards}
</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nHTML report: {output_path}")


def _build_trades_table_v2(trades: List[TradeRecord]) -> str:
    if not trades:
        return "<p>无交易</p>"
    rows = ""
    for t in trades[-80:]:
        c = "positive" if t.pnl_rmb > 0 else "negative"
        entry_ts = pd.Timestamp(t.entry_time).strftime('%m-%d %H:%M')
        rows += f"""<tr>
<td>{entry_ts}</td>
<td><span class="pair-tag">{t.pair_key}</span></td>
<td>{t.side.value}</td>
<td>{t.entry_zscore:+.2f}</td>
<td>{t.entry_spread:+.1f}</td>
<td>{t.exit_spread:+.1f}</td>
<td class="{c}">{t.pnl_points:+.1f}</td>
<td class="{c}">{t.pnl_rmb:+,.0f}</td>
<td>{t.hold_bars}</td>
<td>{t.exit_reason.value}</td>
</tr>\n"""
    
    return f"""<table class="trade-table">
<tr><th>时间</th><th>合约对</th><th>方向</th><th>入Z</th><th>入Spread</th>
<th>出Spread</th><th>点PnL</th><th>元PnL</th><th>K线</th><th>原因</th></tr>
{rows}</table>"""


def _color_for(p: str) -> str:
    return {"IF":"2563eb","IH":"059669","IC":"dc2626","IM":"7c3aed"}.get(p,"6b7280")


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="[v2] 多合约对纯基差回归套利回测")
    parser.add_argument("--products", nargs="+", default=["IF","IH","IC","IM"])
    parser.add_argument("--z-entry", type=float, default=1.5)
    parser.add_argument("--z-exit", type=float, default=0.2)
    parser.add_argument("--tp-pts", type=float, default=3.0)
    parser.add_argument("--sl-pts", type=float, default=8.0)
    parser.add_argument("--time-exit", type=int, default=288)
    parser.add_argument("--lookback", type=int, default=288)
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--volume", type=int, default=1)
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()
    
    cfg = BacktestConfig(
        z_entry=args.z_entry, z_exit=args.z_exit,
        tp_basis_pts=args.tp_pts, sl_basis_pts=args.sl_pts,
        time_exit_bars=args.time_exit, lookback_bars=args.lookback,
        initial_capital=args.capital, volume_per_trade=args.volume,
    )
    
    print("=" * 65)
    print("  [v2] Multi-Pair Basis Arbitrage Backtest")
    print(f"  Z_entry={cfg.z_entry}  TP={cfg.tp_basis_pts}pt  "
          f"SL={cfg.sl_basis_pts}pt  Time={cfg.time_exit_bars}bars")
    print("  => Each bar scans ALL contract pairs, picks highest |z|")
    print("=" * 65)
    
    bt = BasisArbitrageBacktesterV2(cfg)
    results = bt.run(args.products)
    
    rpt = args.report or os.path.join(REPORT_DIR, f"basis_arb_v2_{datetime.now():%Y%m%d_%H%M%S}.html")
    generate_html_report(results, cfg, rpt)


if __name__ == "__main__":
    main()
