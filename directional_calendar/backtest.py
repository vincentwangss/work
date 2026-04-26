"""
backtest.py
离线回测框架

功能：
  - 加载历史行情（akshare 拉取或从 CSV）
  - 模拟方向信号（可自定义：MA/固定/外部文件）
  - 跑完整决策引擎，记录每次换仓
  - 计算 PnL、最大回撤、胜率等统计指标
  - 输出 HTML 可视化报告

用法：
    # 基础回测（IF品种，最近180天，固定多头信号）
    python backtest.py --products IF --days 180 --signal fixed_long

    # 使用均线作为方向信号
    python backtest.py --products IF IH IC --days 365 --signal ma

    # 自定义参数回测
    python backtest.py --sigma-entry 1.5 --sigma-exit 0.2 --cooldown 120

    # 输出 HTML 报告
    python backtest.py --html-report report.html
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from basis_calculator import BasisCalculator, ContractInfo
from direction_signal import Direction, DirectionProvider, DirectionSignal, MAProvider
from risk_manager import RiskManager
from spread_engine import (
    HoldingType, ProductState, SpreadEngine,
    SwitchAction, SwitchDecision, SwitchDecision as SD,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 回测数据源
# ------------------------------------------------------------------

@dataclass
class Bar:
    """模拟K线数据"""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_date: date


def fetch_history_for_backtest(product: str, near_sym: str, far_sym: str,
                                start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """拉取近远月合约历史日线"""
    import akshare as ak
    
    df_near = ak.futures_zh_daily_sina(symbol=f"{near_sym}.CFE")
    df_far = ak.futures_zh_daily_sina(symbol=f"{far_sym}.CFE")
    
    if df_near is None or df_far is None or df_near.empty or df_far.empty:
        return None, None
    
    for df in [df_near, df_far]:
        if df is not None and not df.empty:
            df.columns = ['datetime', 'open', 'high', 'low', 'close',
                          'volume', 'hold', 'open_oi']
            df['date'] = pd.to_datetime(df['datetime']).dt.date
            df.set_index('date', inplace=True)
            for col in ['open','high','low','close','volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    start_d = date.fromisoformat(start_date)
    end_d = date.fromisoformat(end_date)
    mask = (df_near.index >= start_d) & (df_near.index <= end_d) & \
           (df_far.index >= start_d) & (df_far.index <= end_d)
    
    return df_near.loc[mask], df_far.loc[mask]


# ------------------------------------------------------------------
# 方向信号提供者（回测专用）
# ------------------------------------------------------------------

class FixedDirectionProvider(DirectionProvider):
    """固定方向信号（回测用：测试基差逻辑本身）"""

    def __init__(self, direction_map: Dict[str, int]):
        """
        Args:
            direction_map: {产品代码: 1/-1/0}
        """
        self.direction_map = {k: Direction(v) for k, v in direction_map.items()}

    def fetch(self, products: List[str]) -> Dict[str, DirectionSignal]:
        result = {}
        for p in products:
            d = self.direction_map.get(p, Direction.FLAT)
            result[p] = DirectionSignal(
                product=p, direction=d, confidence=1.0,
                source="fixed_backtest"
            )
        return result


class MABacktestProvider(MAProvider):
    """基于指数/合约收盘价的均线方向（回测用）"""

    def __init__(self, fast_period=5, slow_period=20):
        super().__init__(fast_period, slow_period)

    def feed_bar(self, bar: Bar):
        """推入一根 K 线"""
        product = bar.symbol[:2]
        self.push_close(product, bar.close)


# ------------------------------------------------------------------
# 回测交易记录
# ------------------------------------------------------------------

class TradeType(Enum):
    OPEN_NEAR   = "开仓_近月"
    OPEN_FAR    = "开仓_远月"
    SWITCH_TO_FAR  = "换仓_到远月"
    SWITCH_TO_NEAR = "换仓_回近月"
    ROLLOVER    = "换月"
    CLOSE       = "平仓"


@dataclass
class TradeRecord:
    """单条交易记录"""
    timestamp: datetime
    product: str
    action: TradeType
    from_symbol: str
    to_symbol: str
    price_from: float      # 平仓/换出价格
    price_to: float        # 开仓/换入价格
    volume: int            # 手数
    direction: Direction
    pnl_point: float = 0.0  # 点数盈亏
    pnl_rmb: float = 0.0    # 人民币盈亏
    commission: float = 0.0 # 手续费（约万0.23）
    reason: str = ""
    zscore_at_entry: Optional[float] = None
    basis_rate_at_entry: Optional[float] = None
    holding_days: int = 0


# ------------------------------------------------------------------
# 回测引擎
# ------------------------------------------------------------------

@dataclass
class BacktestResult:
    """回测结果汇总"""
    trades: List[TradeRecord]
    equity_curve: pd.Series
    daily_pnl: pd.Series
    stats: dict = field(default_factory=dict)


class Backtester:
    """
    离线回测引擎。
    
    核心流程：
      1. 按日期遍历历史行情
      2. 每天调用 SpreadEngine.on_tick() 获取决策
      3. 执行交易并计算 PnL
      4. 记录 equity curve
      5. 输出统计指标
    """

    def __init__(
        self,
        config_path: str,
        signal_provider: DirectionProvider,
        sigma_entry: float = 1.0,
        sigma_exit: float = 0.3,
        cooldown_min: int = 60,
        initial_capital: float = 1_000_000,
        volume_per_trade: int = 1,
        commission_rate: float = 0.000023,  # 万0.23
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 覆盖配置中的参数
        self.config["basis"]["sigma_entry"] = sigma_entry
        self.config["basis"]["sigma_exit"] = sigma_exit
        self.config["execution"]["rebalance_cooldown_min"] = cooldown_min
        
        self.signal_provider = signal_provider
        self.initial_capital = initial_capital
        self.volume_per_trade = volume_per_trade
        self.commission_rate = commission_rate

        # 构建模块
        self.calc = BasisCalculator()
        
        # 初始化分红
        instruments = self.config.get("instruments", {})
        for product, cfg in instruments.items():
            ds = cfg.get("dividend_schedule", {})
            if ds:
                self.calc.load_dividend_schedule(product, ds, "config")
        
        self.engine = SpreadEngine(self.calc, signal_provider, self.config)
        self.risk_mgr = RiskManager(self.config.get("risk", {}))
        self.risk_mgr.set_equity(initial_capital)

        # 回测运行时状态
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = []
        self.daily_records: Dict[date, float] = {}
        self._positions: Dict[str, dict] = {}  # {product: {symbol, volume, direction, entry_price}}

    def run(
        self,
        days_back: int = 180,
        products: Optional[List[str]] = None,
    ) -> BacktestResult:
        """执行回测"""
        
        if products is None:
            products = [
                p for p, cfg in self.config["instruments"].items()
                if cfg.get("enabled", False)
            ]

        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        print(f"\n{'='*70}")
        print(f"  跨期套利回测 | {start_date} ~ {end_date}")
        print(f"  品种: {', '.join(products)}")
        print(f"  参数: σ_entry={self.config['basis']['sigma_entry']} "
              f"σ_exit={self.config['basis']['sigma_exit']} "
              f"cooldown={self.config['execution']['rebalance_cooldown_min']}min")
        print(f"{'='*70}")

        all_bars = {}
        for product in products:
            bars = self._load_product_bars(product, start_date, end_date)
            if bars:
                all_bars[product] = bars

        if not all_bars:
            print("❌ 无有效数据，回测终止")
            return BacktestResult(trades=[], equity_curve=pd.Series(), 
                                  daily_pnl=pd.Series())

        # 按日期排序遍历
        all_dates = set()
        for bars in all_bars.values():
            all_dates.update(b.trade_date for b in bars)
        
        sorted_dates = sorted(all_dates)
        print(f"  交易日数: {len(sorted_dates)}")

        current_equity = self.initial_capital
        peak_equity = self.initial_capital

        for trade_day in sorted_dates:
            day_pnl = 0.0
            
            # 风控每日重置检查
            self.risk_mgr.on_new_day(trade_day)
            
            contracts_today = {}
            for product in products:
                bars = all_bars.get(product, [])
                day_bars = [b for b in bars if b.trade_date == trade_day]
                
                if len(day_bars) < 2:
                    continue
                
                # 近月和远月的当日Bar（按合约名区分）
                cfg = self.config["instruments"][product]
                near_cfg = None
                far_cfg = None
                
                for b in day_bars:
                    sym_prefix = b.symbol[:len(cfg["near_symbol"])]
                    if sym_prefix.startswith(product):
                        if b.symbol == cfg["near_symbol"] or (not near_cfg):
                            near_cfg = b
                        elif b.symbol == cfg["far_symbol"] or (not far_cfg):
                            far_cfg = b
                
                if not near_cfg or not far_cfg:
                    continue

                # 更新均线信号
                if isinstance(self.signal_provider, MABacktestProvider):
                    self.signal_provider.feed_bar(near_cfg)

                # 构建合约信息
                near_exp = _third_friday_of(near_cfg.symbol)
                far_exp = _third_friday_of(far_cfg.symbol)
                
                near_contract = ContractInfo(
                    near_cfg.symbol, product, near_exp, near_cfg.close,
                    bid=near_cfg.low, ask=near_cfg.high,
                )
                far_contract = ContractInfo(
                    far_cfg.symbol, product, far_exp, far_cfg.close,
                    bid=far_cfg.low, ask=far_cfg.high,
                )

                contracts_today[product] = {"near": near_contract, "far": far_contract}

            # 触发决策
            decisions = self.engine.on_tick(contracts_today, 
                                            as_of=datetime.combine(trade_day, datetime.min.time()))
            
            # 执行决策
            for decision in decisions:
                exec_result = self._execute_decision(decision, trade_day)
                if exec_result:
                    trade_rec, pnl = exec_result
                    self.trades.append(trade_rec)
                    day_pnl += pnl
            
            current_equity += day_pnl
            peak_equity = max(peak_equity, current_equity)
            self.equity_curve.append(current_equity)
            self.daily_records[trade_day] = day_pnl

            # 风控更新
            self.risk_mgr.on_trade(day_pnl)

        # --- 统计 ---
        result = self._build_result()
        self._print_summary(result)
        return result

    def _load_product_bars(self, product: str, 
                           start_date: date, end_date: date) -> List[Bar]:
        """加载单个品种的历史Bar数据"""
        cfg = self.config["instruments"][product]
        near_sym = cfg["near_symbol"]
        far_sym = cfg["far_symbol"]
        
        try:
            df_near, df_far = fetch_history_for_backtest(
                product, near_sym, far_sym,
                start_date.isoformat(), end_date.isoformat()
            )
        except Exception as e:
            logger.error(f"[回测] {product} 数据拉取失败: {e}")
            return []

        if df_near is None or df_near.empty:
            return []

        # 合并为统一格式
        bars = []
        common_idx = df_near.index.intersection(df_far.index)
        for idx in common_idx:
            row_n = df_near.loc[idx]
            row_f = df_far.loc[idx]
            d = idx if isinstance(idx, date) else date.fromisoformat(str(idx))
            if not isinstance(d, date):
                d = pd.to_datetime(d).date()

            bars.extend([
                Bar(symbol=near_sym, open=row_n['open'], high=row_n['high'],
                    low=row_n['low'], close=row_n['close'], volume=row_n['volume'],
                    trade_date=d),
                Bar(symbol=far_sym, open=row_f['open'], high=row_f['high'],
                    low=row_f['low'], close=row_f['close'], volume=row_f['volume'],
                    trade_date=d),
            ])
        return bars

    def _execute_decision(self, decision: SwitchDecision,
                           trade_day: date) -> Optional[Tuple[TradeRecord, float]]:
        """执行一条换仓指令，返回 (交易记录, PnL_rmb)"""
        product = decision.product
        multiplier = self.config["instruments"][product].get("multiplier", 300)
        volume = self.volume_per_trade
        action = decision.action
        direction = decision.direction

        # 获取当前持仓
        pos = self._positions.get(product)
        pnl_total = 0.0

        if action == SwitchAction.CLOSE:
            if pos:
                close_price = (decision.from_symbol in self.engine._contract_cache.get(product, {}).get("near", {})
                               and self.engine._contract_cache[product]["near"].last_price) or \
                              (pos.get("entry_price") or 3700)
                
                # 实际应该用当天close
                snap = self.engine.get_state(product).last_basis_snapshot
                if direction == Direction.LONG:
                    close_price = snap.near_price if snap else close_price
                    pnl_pt = (close_price - pos["entry_price"])
                else:
                    close_price = snap.near_price if snap else close_price
                    pnl_pt = (pos["entry_price"] - close_price)
                
                pnl_rmb = pnl_pt * volume * multiplier
                comm = close_price * volume * multiplier * self.commission_rate * 2  # 开+平
                pnl_total -= comm

                rec = TradeRecord(
                    timestamp=datetime.combine(trade_day, datetime.min.time()),
                    product=product, action=TradeType.CLOSE,
                    from_symbol=pos["symbol"], to_symbol="",
                    price_from=close_price, price_to=0, volume=volume,
                    direction=pos["direction"], pnl_point=pnl_pt * volume,
                    pnl_rmb=pnl_rmb - comm, commission=comm,
                    reason=decision.reason,
                )
                self.trades.append(rec)
                del self._positions[product]
                return rec, pnl_rmb - comm
            return None

        # 开仓 / 换仓
        snap = self.engine.get_state(product).last_basis_snapshot
        if not snap:
            return None

        to_price = snap.far_price if action in (SwitchAction.OPEN_FAR, SwitchAction.SWITCH_TO_FAR, SwitchAction.ROLLOVER) \
                  else snap.near_price
        from_price = snap.near_price if snap else 0

        # 如果是换仓，先算旧仓位PnL
        old_pnl_rmb = 0.0
        if pos and action in (SwitchAction.SWITCH_TO_FAR, SwitchAction.SWITCH_TO_NEAR, SwitchAction.ROLLOVER):
            if pos["direction"] == Direction.LONG:
                old_pnl_pt = (from_price - pos["entry_price"]) * pos["volume"]
            else:
                old_pnl_pt = (pos["entry_price"] - from_price) * pos["volume"]
            old_pnl_rmb = old_pnl_pt * multiplier
            comm_old = from_price * pos["volume"] * multiplier * self.commission_rate
            old_pnl_rmb -= comm_old

        # 新仓位手续费
        comm_new = to_price * volume * multiplier * self.commission_rate
        total_comm = comm_new + (old_pnl_rmb != 0 and abs(old_pnl_rmb) > 0.01 and comm_new or 0)

        # 映射action到TradeType
        action_map = {
            SwitchAction.OPEN_NEAR: TradeType.OPEN_NEAR,
            SwitchAction.OPEN_FAR: TradeType.OPEN_FAR,
            SwitchAction.SWITCH_TO_FAR: TradeType.SWITCH_TO_FAR,
            SwitchAction.SWITCH_TO_NEAR: TradeType.SWITCH_TO_NEAR,
            SwitchAction.ROLLOVER: TradeType.ROLLOVER,
        }
        
        to_symbol = decision.to_symbol
        from_sym = decision.from_symbol

        rec = TradeRecord(
            timestamp=datetime.combine(trade_day, datetime.min.time()),
            product=product, action=action_map.get(action, TradeType.OPEN_NEAR),
            from_symbol=from_sym, to_symbol=to_symbol,
            price_from=from_price, price_to=to_price, volume=volume,
            direction=direction, pnl_point=0, pnl_rmb=old_pnl_rmb - total_comm,
            commission=total_comm, reason=decision.reason,
            zscore_at_entry=decision.zscore,
            basis_rate_at_entry=decision.basis_rate,
        )
        
        self._positions[product] = {
            "symbol": to_symbol,
            "volume": volume,
            "direction": direction,
            "entry_price": to_price,
            "entry_date": trade_day,
        }
        return rec, old_pnl_rmb - total_comm

    def _build_result(self) -> BacktestResult:
        """构建回测结果"""
        eq = pd.Series(
            self.equity_curve,
            index=pd.date_range(
                start=self.daily_records.keys().__iter__().__next__()
                    if self.daily_records else datetime.now(),
                periods=len(self.equity_curve), freq='D'
            ) if self.equity_curve else pd.DatetimeIndex([]),
        )
        
        daily = pd.Series(self.daily_records).sort_index()
        
        # 统计指标
        stats = self._calc_stats(eq, daily, self.trades)
        
        return BacktestResult(
            trades=list(self.trades),  # copy
            equity_curve=eq,
            daily_pnl=daily,
            stats=stats,
        )

    def _calc_stats(self, equity: pd.Series, daily_pnl: pd.Series,
                     trades: List[TradeRecord]) -> dict:
        """计算统计指标"""
        if equity.empty:
            return {}

        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital * 100
        
        # 最大回撤
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax * 100
        max_dd = drawdown.min()
        
        # 盈亏比
        winning = [t.pnl_rmb for t in trades if t.pnl_rmb > 0]
        losing = [abs(t.pnl_rmb) for t in trades if t.pnl_rmb < 0]
        
        win_rate = len(winning) / max(len(trades), 1) * 100
        avg_win = np.mean(winning) if winning else 0
        avg_loss = np.mean(losing) if losing else 1e-6
        profit_factor = sum(winning) / max(sum(losing), 1)
        
        # 年化收益
        n_days = len(equity)
        annualized_return = total_return * (252 / max(n_days, 1))
        
        # Sharpe (简化)
        if len(daily_pnl) > 10:
            sharpe = np.sqrt(252) * daily_pnl.mean() / max(daily_pnl.std(), 1e-6)
        else:
            sharpe = 0.0

        return {
            "初始资金": f"{self.initial_capital:,.0f}",
            "最终权益": f"{equity.iloc[-1]:,.0f}",
            "总收益率": f"{total_return:+.2f}%",
            "年化收益率": f"{annualized_return:+.2f}%",
            "最大回撤": f"{max_dd:.2f}%",
            "Sharpe比率": f"{sharpe:.2f}",
            "总交易次数": len(trades),
            "盈利次数": len(winning),
            "亏损次数": len(losing),
            "胜率": f"{win_rate:.1f}%",
            "平均盈利": f"{avg_win:,.0f}",
            "平均亏损": f"{avg_loss:,.0f}",
            "盈亏比": f"{profit_factor:.2f}",
            "交易日数": n_days,
        }

    @staticmethod
    def _print_summary(result: BacktestResult):
        """打印结果摘要"""
        stats = result.stats
        if not stats:
            print("无统计数据")
            return
        
        print(f"\n{'─'*50}")
        print("  📊 回测结果摘要")
        print(f"{'─'*50}")
        for key, val in stats.items():
            print(f"  {key:<12s}: {val}")
        print(f"{'─'*50}")


def _third_friday_of(symbol: str) -> date:
    """从合约符号解析交割日"""
    try:
        year = int("20" + symbol[2:4])
        month = int(symbol[4:6])
        d = date(year, month, 1)
        days_to_fri = (4 - d.weekday()) % 7
        first_fri = d.replace(day=1 + days_to_fri)
        return first_fri.replace(day=min(first_fri.day + 14, 28))
    except:
        return date.today()


# ------------------------------------------------------------------
# HTML 报告生成
# ------------------------------------------------------------------

def generate_html_report(result: BacktestResult, output_path: str,
                          config: dict) -> None:
    """生成可视化 HTML 报告"""
    
    trades_df = pd.DataFrame([{
        "时间": t.timestamp.strftime("%Y-%m-%d"),
        "品种": t.product,
        "操作": t.action.value,
        "方向": t.direction.name,
        "价格(出)": f"{t.price_from:.1f}",
        "价格(入)": f"{t.price_to:.1f}",
        "手数": t.volume,
        "盈亏(元)": f"{t.pnl_rmb:+,.0f}",
        "原因": t.reason,
        "z-score": f"{t.zscore_at_entry:.2f}" if t.zscore_at_entry is not None else "-",
    } for t in result.trades])

    equity_html = ""
    if not result.equity_curve.empty:
        dates = result.equity_curve.index.strftime("%Y-%m-%d").tolist()
        values = result.equity_curve.tolist()
        
        equity_html = f"""
        <div id="equity_chart" style="width:100%; height:400px;"></div>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
        new Chart(document.getElementById('equity_chart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(dates)},
                datasets: [{{
                    label: '权益曲线',
                    data: {json.dumps(values)},
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37,99,235,0.1)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                }}]
            }},
            options: {{ responsive: true, plugins: {{legend:{{display:false}}}} }}
        }});
        </script>"""

    stats_rows = "".join(
        f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
        for k, v in result.stats.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>跨期套利回测报告</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 20px; background:#f8fafc; }}
.card {{ background:white; border-radius:8px; padding:24px; margin:16px 0; box-shadow:0 1px 3px rgba(0,0,0,.08); }}
h1,h2 {{ color:#1e293b; }} table {{ width:100%; border-collapse:collapse; }}
th, td {{ padding:8px 12px; text-align:left; border-bottom:1px solid #e2e8f0; font-size:13px; }}
th {{ background:#f1f5f9; font-weight:600; }}
.positive {{ color:#dc2626; }} .negative {{ color:#16a34a; }}
</style></head>
<body>
<div class="card">
<h1>📈 股指期货跨期套利回测报告</h1>
<p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>

<div class="card"><h2>📋 统计概览</h2><table>{stats_rows}</table></div>

<div class="card"><h2>📈 权益曲线</h2>{equity_html}</div>

<div class="card"><h2>📝 交易明细</h2>
{trades_df.to_html(index=False, escape=False) if not trades_df.empty else '<p>无交易记录</p>'}
</div>

</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n📄 HTML 报告已保存: {output_path}")


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="跨期套利离线回测工具")
    parser.add_argument("--config",
                        default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--days", type=int, default=180, help="回测天数")
    parser.add_argument("--products", nargs="+", help="指定品种")
    parser.add_argument("--signal", choices=["fixed_long", "fixed_short", "ma"],
                        default="fixed_long", help="方向信号模式")
    parser.add_argument("--sigma-entry", type=float, default=None)
    parser.add_argument("--sigma-exit", type=float, default=None)
    parser.add_argument("--cooldown", type=int, default=None)
    parser.add_argument("--capital", type=float, default=1_000_000, help="初始资金")
    parser.add_argument("--volume", type=int, default=1, help="每笔手数")
    parser.add_argument("--html-report", type=str, default=None, help="HTML输出路径")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # 构建信号提供者
    if args.signal == "fixed_long":
        provider = FixedDirectionProvider({"IF": 1, "IH": 1, "IC": 1, "IM": 1})
    elif args.signal == "fixed_short":
        provider = FixedDirectionProvider({"IF": -1, "IH": -1, "IC": -1, "IM": -1})
    elif args.signal == "ma":
        provider = MABacktestProvider()
    else:
        raise ValueError(f"未知信号模式: {args.signal}")

    # 运行回测
    bt = Backtester(
        config_path=args.config,
        signal_provider=provider,
        sigma_entry=args.sigma_entry or 1.0,
        sigma_exit=args.sigma_exit or 0.3,
        cooldown_min=args.cooldown or 60,
        initial_capital=args.capital,
        volume_per_trade=args.volume,
    )

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    result = bt.run(days_back=args.days, products=args.products)

    # HTML报告
    if args.html_report and result.trades or result.equity_curve.size > 0:
        generate_html_report(result, args.html_report, config)


if __name__ == "__main__":
    main()
