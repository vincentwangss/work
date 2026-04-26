"""
quick_backtest.py v2
带方向性日历价差策略回测引擎（完整版）

核心功能：
  1. 四合约轮转：近月到期前自动展期到下一合约
  2. 跨品种切换：根据方向信号强度在不同品种间切换（IF/IH/IC/IM）
  3. 基差 Z-score 驱动的近/远月切换
  4. 完整的 PnL 计算（含换月滑点、跨品种切换成本）

策略逻辑：
  方向信号决定持什么品种：
    强看多(+2) → IM (中证1000，弹性最大)
    看多(+1)   → IF (沪深300，默认)
    中性(0)    → 不操作/维持
    看空(-1)   → IH (上证50，偏防御)
    强看空(-2) → IC (中证500，空方弹性大)

  品种内部由基差 Z-score 决定持近月还是远月：
    持近月 + z > sigma_entry  → 换远月（基差偏高）
    持远月 + z < sigma_exit   → 换回近月（基差回归）

用法：
    # 单品种基础回测（兼容 v1）
    python quick_backtest.py --data data/5min_basis_IF_20260425.csv

    # 多品种 + 方向信号文件
    python quick_backtest.py --data-dir data/ --products IF IH IC \
        --signal-file signals/direction.json

    # 模拟：固定强看多IM
    python quick_backtest.py --data-dir data/ --products IF IH IC IM \
        --fixed-signal IM:+2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

# ------------------------------------------------------------------
# 常量 & 配置
# ------------------------------------------------------------------

MULTIPLIER_MAP = {"IF": 300, "IH": 300, "IC": 200, "IM": 200}

# 品种特征：用于跨品种选择的 beta/波动率排序
PRODUCT_BETA = {
    "IM": 1.5,   # 中证1000 — 弹性最大，强看多首选
    "IF": 1.0,   # 沪深300 — 基准
    "IC": 1.2,   # 中证500 — 偏中小盘
    "IH": 0.8,   # 上证50 — 大盘蓝筹，偏防御（看空时用）
}
# 空头逻辑：越跌越凶的排前面
PRODUCT_BETA_SHORT = {
    "IC": 1.3,   # 中证500 — 空方弹性
    "IM": 1.4,   # 中证1000 — 空方弹性也大但流动性稍差
    "IF": 1.0,
    "IH": 0.6,   # 上交50 — 最抗跌（强看空时反而不选）
}


class TradeType:
    OPEN       = "OPEN"
    SWITCH_FAR = "SWITCH_TO_FAR"     # 同品种内 近→远
    SWITCH_NEAR= "SWITCH_TO_NEAR"    # 同品种内 远→近
    ROLLOVER   = "ROLLOVER"          # 合约展期（近月换下一合约）
    SWITCH_PRODUCT = "SWITCH_PRODUCT" # 跨品种切换
    CLOSE      = "CLOSE"


class TradeRecord:
    __slots__ = [
        "timestamp", "action", "product",
        "from_symbol", "to_symbol",
        "price_from", "price_to", "volume",
        "pnl_point", "pnl_rmb", "commission",
        "reason", "zscore", "basis_rate",
        "near_price", "far_price", "direction_signal",
        "equity_after", "position_after",
    ]

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k, 0))


# ------------------------------------------------------------------
# 合约管理
# ------------------------------------------------------------------

def parse_expiry(symbol: str) -> date:
    """从合约代码解析第三周五交割日"""
    try:
        year = int("20" + symbol[2:4])
        month = int(symbol[4:6])
        d = date(year, month, 1)
        days_to_fri = (4 - d.weekday()) % 7
        first_fri = d.replace(day=1 + days_to_fri)
        return first_fri.replace(day=min(first_fri.day + 14, 28))
    except Exception:
        return date.today() + timedelta(days=55)


def next_contract(symbol: str) -> str:
    """
    获取下一个合约代码。
    规则：月份+3（季度合约循环：3/6/9/12）
    例：IF2606 → IF2609, IF2609 → IF2612, IF2612 → IF2703
    """
    prefix = symbol[:2]
    year = int("20" + symbol[2:4])
    month = int(symbol[4:6])
    month += 3
    if month > 12:
        month -= 12
        year += 1
    return f"{prefix}{year % 100:02d}{month:02d}"


def build_contract_chain(
    start_near: str,
    start_far: str,
    n_contracts: int = 4,
) -> List[str]:
    """
    构建合约链（从近到远）。

    股指期货是季度合约（3/6/9/12月），所以链是：
      [当月, 下季, 隔季, 再隔季] = 4个活跃合约

    Args:
        start_near: 起始近月合约，如 "IF2606"
        start_far:  起始远季合约，如 "IF2609"
        n_contracts: 链长度（默认4个）
    """
    chain = [start_near]
    cur = start_near
    for _ in range(n_contracts - 1):
        cur = next_contract(cur)
        chain.append(cur)

    # 确保 start_far 在链中
    if start_far not in chain:
        # start_far 可能不是简单的+3关系（如跨年），手动添加
        if start_far not in chain:
            chain.append(start_far)
            chain.sort(key=lambda s: (int("20"+s[2:4]), int(s[4:6])))

    return chain


def get_active_pair(
    chain: List[str],
    as_of: date,
    rollover_days: int = 5,
    preferred_far_offset: int = 1,
) -> Tuple[str, str]:
    """
    根据日期获取当前活跃的近月/远月合约对。

    规则：
      - 近月 = 链中第一个未过期的合约
      - 远月 = 近月之后第 preferred_far_offset 个合约
        默认 offset=1 表示用相邻的下一季度合约(如 IF2606 + IF2609)
      - 如果近月距离交割 <= rollover_days，自动展期到下一个

    Returns:
        (near_symbol, far_symbol)
    """
    for i, sym in enumerate(chain):
        exp = parse_expiry(sym)
        days_left = (exp - as_of).days
        if days_left > rollover_days:
            near = sym
            # 远月 = 近月之后的第 N 个（默认1=下一个季度）
            far_idx = i + preferred_far_offset
            if far_idx < len(chain):
                far = chain[far_idx]
            else:
                far = chain[-1]
            return near, far

    # 所有合约都快到期了，用最后两个
    return chain[-2], chain[-1]


# ------------------------------------------------------------------
# 方向信号
# ------------------------------------------------------------------

class DirectionSignal:
    """
    方向信号管理。

    信号来源优先级：
      1. 固定信号（回测模拟用）
      2. 外部信号文件（JSON）
      3. 默认信号
    """

    def __init__(self):
        self.fixed_signals: Dict[str, int] = {}   # {product: signal}
        self.signal_file: Optional[str] = None
        self.signal_ttl_minutes: int = 60
        self._cached: Dict[str, Tuple[int, datetime]] = {}
        self._signal_history: List[Tuple[datetime, Dict[str, int]]] = []

    @classmethod
    def from_fixed(cls, spec: str) -> "DirectionSignal":
        """
        从字符串创建固定信号。
        格式: "IF:+1" 或 "IM:+2,IH:-1"
        """
        ds = cls()
        for item in spec.split(","):
            item = item.strip()
            if ":" in item:
                prod, sig = item.split(":")
                ds.fixed_signals[prod.strip()] = int(sig.strip())
        return ds

    @classmethod
    def from_file(cls, path: str) -> "DirectionSignal":
        ds = cls()
        ds.signal_file = path
        return ds

    def get_signal(self, product: str, dt: datetime) -> int:
        """获取指定品种在时刻 dt 的方向信号 (-2 ~ +2)"""
        # 1. 固定信号优先
        if product in self.fixed_signals:
            return self.fixed_signals[product]

        # 2. 外部文件信号
        if self.signal_file:
            cached_sig, cached_time = self._cached.get(product, (0, datetime.min))
            age = (dt - cached_time).total_seconds() / 60.0
            if age < self.signal_ttl_minutes:
                return cached_sig

            # 尝试读取文件
            try:
                if os.path.exists(self.signal_file):
                    with open(self.signal_file, "r") as f:
                        data = json.load(f)
                    sig = int(data.get(product, 0))
                    self._cached[product] = (sig, dt)
                    return sig
            except Exception:
                pass

        # 3. 默认：无信号（不操作）
        return 0

    def get_all_signals(self, dt: datetime) -> Dict[str, int]:
        """获取所有品种当前信号"""
        return {p: self.get_signal(p, dt) for p in MULTIPLIER_MAP}

    def select_product(self, signals: Dict[str, int]) -> Optional[str]:
        """
        根据信号强度选择最优品种。

        选择逻辑：
          正信号(看多): 选 beta 最大的品种（IM > IC > IF > IH）
          负信号(看空): 选空方弹性大的品种（IC > IM > IF > IH）
          零信号:       返回 None（不切换）
        """
        max_abs_sig = max(abs(v) for v in signals.values()) if signals else 0
        if max_abs_sig == 0:
            return None

        # 找出最强信号的品种
        best_product = None
        best_score = -999

        for prod, sig in signals.items():
            if sig == 0:
                continue
            abs_sig = abs(sig)
            # 分数 = 信号强度 * 品种beta（多头用多头的beta，空头用空头的beta）
            if sig > 0:
                beta = PRODUCT_BETA.get(prod, 1.0)
            else:
                beta = PRODUCT_BETA_SHORT.get(prod, 1.0)

            score = abs_sig * beta
            if score > best_score:
                best_score = score
                best_product = prod

        return best_product


# ------------------------------------------------------------------
# 回测引擎 v2
# ------------------------------------------------------------------

class BacktestEngineV2:
    """
    完整版回测引擎。

    支持功能：
      - 四合约自动展期（rollover）
      - 跨品种切换（product switch）
      - 基差 Z-score 驱动的近/远月切换
      - 多品种数据同时输入
    """

    def __init__(
        self,
        sigma_entry: float = 1.0,
        sigma_exit: float = 0.3,
        cooldown_bars: int = 6,
        rollover_days: int = 5,
        initial_capital: float = 1_000_000,
        volume: int = 1,
        commission_rate: float = 0.000023,
        slippage_ticks: float = 0.5,         # 展期滑点(tick数)
        product_switch_cost_bps: float = 2.0, # 跨品种切换成本(bp)
        use_precomputed_stats: bool = True,
    ):
        self.sigma_entry = sigma_entry
        self.sigma_exit = sigma_exit
        self.cooldown_bars = cooldown_bars
        self.rollover_days = rollover_days
        self.initial_capital = initial_capital
        self.volume = volume
        self.commission_rate = commission_rate
        self.slippage_ticks = slippage_ticks
        self.product_switch_cost_bps = product_switch_cost_bps
        self.use_precomputed_stats = use_precomputed_stats

        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = []
        self.equity_times: List[datetime] = []

        # 当前持仓状态
        self.position: Optional[dict] = None
        # 格式: {
        #   "product": "IF",
        #   "symbol": "IF2606",
        #   "holding": "NEAR" | "FAR",
        #   "entry_price": float,
        #   "entry_time": datetime,
        #   "direction": 1 | -1,  (1=多, -1=空)
        # }

    def run(
        self,
        data_map: Dict[str, pd.DataFrame],
        direction: DirectionSignal,
    ) -> dict:
        """
        执行回测。

        Args:
            data_map: {product: DataFrame} 各品种的基差CSV数据
            direction: 方向信号源
        """
        # ----------------------------------------------------------
        # 初始化：构建各品种合约链、预计算统计量
        # ----------------------------------------------------------
        chains: Dict[str, List[str]] = {}
        stats_map: Dict[str, Tuple[float, float]] = {}

        print(f"\n{'='*70}")
        print(f"  Backtest Engine V2")
        print(f"{'='*70}")

        for product, df in data_map.items():
            df["datetime"] = pd.to_datetime(df["datetime"])
            near_sym = df["near_symbol"].iloc[0]
            far_sym = df["far_symbol"].iloc[0]

            # Handle synthetic data (e.g., IFMAIN) - infer real contract from data dates
            import re as _re
            if not _re.match(r'^[A-Z]{2}\d{4}$', str(near_sym)):
                dt_first = df["datetime"].iloc[0]
                y2 = str(dt_first.year)[-2:]
                m = ((dt_first.month - 1) // 3) * 3 + 1
                near_sym = f"{product}{y2}{m:02d}"
                far_m = m + 3 if m <= 6 else 3
                far_y = int(y2) + 1 if far_m < m else int(y2)
                far_sym = f"{product}{far_y:02d}{far_m:02d}"
                print(f"    [INFO] Synthetic symbols detected, using {near_sym}/{far_sym}")

            chain = build_contract_chain(near_sym, far_sym, n_contracts=4)
            chains[product] = chain

            print(f"\n  [{product}] Contract chain: {' → '.join(chain)}")

            if self.use_precomputed_stats:
                col = "adj_annualized_rate" if "adj_annualized_rate" in df.columns else "raw_annualized_rate"
                mu = df[col].mean()
                sigma = df[col].std()
                stats_map[product] = (mu, sigma)
                print(f"    Stats: mu={mu:+.3f}%  sigma={sigma:.3f}%  ({len(df)} bars)")
                print(f"    Range: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

        # 全局时间范围（取所有品种的并集）
        all_times = []
        for df in data_map.values():
            all_times.extend(df["datetime"].tolist())
        all_times_sorted = sorted(set(all_times))
        print(f"\n  Total time range: {all_times_sorted[0]} ~ {all_times_sorted[-1]}")
        print(f"  Total bars across products: {sum(len(df) for df in data_map.values())}")

        # ----------------------------------------------------------
        # 主循环：逐根 bar 遍历
        # ----------------------------------------------------------
        current_equity = self.initial_capital
        last_switch_bar = -9999
        state = "NONE"           # NONE / NEAR / FAR
        current_product = None   # 当前持有品种
        bar_counter = 0

        for dt in all_times_sorted:
            bar_counter += 1
            as_of_date = dt.date() if hasattr(dt, 'date') else dt

            # ---- 获取当前时刻的方向信号 ----
            all_sigs = direction.get_all_signals(dt)
            target_product = direction.select_product(all_sigs)

            # ---- 获取当前品种的数据 ----
            # 首次运行时确定目标品种
            if current_product is None:
                if target_product and target_product in data_map:
                    current_product = target_product
                else:
                    # 回退到第一个有数据的品种
                    current_product = target_product or (list(data_map.keys())[0] if data_map else "IF")

            prod = current_product

            # 安全检查：确保当前品种有数据
            if prod not in data_map:
                # 尝试切换到有数据的品种
                available = [p for p in data_map if p != prod]
                if available:
                    current_product = available[0]
                    prod = current_product
                else:
                    continue  # 没有任何数据，跳过
            if prod not in data_map:
                continue

            df_prod = data_map[prod]
            # 找到最接近 dt 的行
            mask = df_prod["datetime"] == dt
            if not mask.any():
                # 用最近的前一根
                mask = df_prod["datetime"] <= dt
                if not mask.any():
                    continue
                row_idx = df_prod[mask].index[-1]
            else:
                row_idx = df_prod[mask].index[0]

            row = df_prod.loc[row_idx]
            near_price = float(row["near_close"])
            far_price = float(row["far_close"])
            near_sym = str(row["near_symbol"])
            far_sym = str(row["far_symbol"])
            adj_rate = float(row.get("adj_annualized_rate", row.get("raw_annualized_rate", 0)))

            # Z-score
            mu, sigma = stats_map.get(prod, (0.0, 1.0))
            zscore = (adj_rate - mu) / sigma if sigma > 1e-8 else 0.0

            # ---- 检查是否需要展期（rollover） ----
            action = None
            reason = ""
            rollover_triggered = False

            if current_product and current_product in chains:
                chain = chains[current_product]
                active_near, active_far = get_active_pair(chain, as_of_date, self.rollover_days)

                if self.position:
                    pos_sym = self.position.get("symbol", "")
                    # 如果当前持有的合约不等于活跃近月，需要展期
                    if self.position["holding"] == "NEAR" and pos_sym != active_near:
                        rollover_triggered = True
                    elif self.position["holding"] == "FAR" and pos_sym != active_far:
                        rollover_triggered = True

            # ---- 冷却期检查 ----
            in_cooldown = (bar_counter - last_switch_bar) < self.cooldown_bars

            # ---- 决策状态机 ----
            sig_val = all_sigs.get(prod, 1)

            # 1. 跨品种切换检查（仅在有持仓时）
            if (target_product and target_product != current_product
                    and current_product is not None
                    and self.position is not None):
                if not in_cooldown:
                    action = TradeType.SWITCH_PRODUCT
                    reason = f"{current_product} -> {target_product}: signal {all_sigs}"
                    last_switch_bar = bar_counter

            # 2. 展期检查
            elif rollover_triggered and not in_cooldown:
                action = TradeType.ROLLOVER
                reason = f"Rollover: {self.position.get('symbol','')} expiring soon"
                last_switch_bar = bar_counter

            # 3. 正常状态机
            elif state == "NONE":
                if target_product:
                    action = TradeType.OPEN
                    reason = f"Open {target_product} LONG (signal={sig_val})"
                    current_product = target_product
                    state = "NEAR"

            elif state == "NEAR":
                if not in_cooldown and zscore > self.sigma_entry:
                    action = TradeType.SWITCH_FAR
                    reason = f"NEAR->FAR: z={zscore:.2f} > +{self.sigma_entry}"
                    state = "FAR"
                    last_switch_bar = bar_counter

            elif state == "FAR":
                if not in_cooldown and zscore < self.sigma_exit:
                    action = TradeType.SWITCH_NEAR
                    reason = f"FAR->NEAR: z={zscore:.2f} < {self.sigma_exit}"
                    state = "NEAR"
                    last_switch_bar = bar_counter

            # ---- 执行动作 ----
            pnl_rmb = 0.0
            mult = MULTIPLIER_MAP.get(prod, 300)

            if action == TradeType.OPEN:
                comm = near_price * self.volume * mult * self.commission_rate
                pnl_rmb = -comm
                self.position = {
                    "product": prod, "symbol": near_sym,
                    "entry_price": near_price, "holding": "NEAR",
                    "entry_time": dt, "direction": 1,
                }
                rec = TradeRecord(
                    timestamp=dt, action=action, product=prod,
                    from_symbol="", to_symbol=near_sym,
                    price_from=0, price_to=near_price, volume=self.volume,
                    pnl_point=0, pnl_rmb=pnl_rmb, commission=comm,
                    reason=reason, zscore=zscore, basis_rate=adj_rate,
                    near_price=near_price, far_price=far_price,
                    direction_signal=sig_val, equity_after=current_equity + pnl_rmb,
                    position_after=f"{prod}/{near_sym}/NEAR",
                )
                self.trades.append(rec)

            elif action == TradeType.SWITCH_FAR:
                pnl_rmb, rec = self._exec_switch(
                    dt, prod, near_sym, far_sym, near_price, far_price,
                    TradeType.SWITCH_FAR, "FAR", zscore, adj_rate, sig_val,
                    current_equity, mult,
                )
                self.trades.append(rec)

            elif action == TradeType.SWITCH_NEAR:
                pnl_rmb, rec = self._exec_switch(
                    dt, prod, far_sym, near_sym, far_price, near_price,
                    TradeType.SWITCH_NEAR, "NEAR", zscore, adj_rate, sig_val,
                    current_equity, mult,
                )
                self.trades.append(rec)

            elif action == TradeType.ROLLOVER:
                pnl_rmb, rec = self._exec_rollover(
                    dt, prod, chains[prod],
                    near_price, far_price, as_of_date,
                    zscore, adj_rate, sig_val, current_equity, mult,
                )
                self.trades.append(rec)

            elif action == TradeType.SWITCH_PRODUCT:
                pnl_rmb, rec = self._exec_product_switch(
                    dt, current_product, target_product,
                    data_map, chains, stats_map, row,
                    zscore, adj_rate, all_sigs, current_equity,
                )
                self.trades.append(rec)
                if target_product:
                    current_product = target_product
                    # 重置状态为 NEAR（新品种重新开近月）
                    state = "NEAR"

            current_equity += pnl_rmb

            # 权益曲线（每小时记录一次）
            if bar_counter % 12 == 0:
                self.equity_curve.append(current_equity)
                self.equity_times.append(dt)

        # ---- 最终结算 ----
        if self.position and len(all_times_sorted) > 0:
            last_dt = all_times_sorted[-1]
            pos = self.position
            prod = pos["product"]
            mult = MULTIPLIER_MAP.get(prod, 300)

            # 获取最后价格
            if prod in data_map:
                df_p = data_map[prod]
                last_row = df_p.iloc[-1]
                if pos["holding"] == "NEAR":
                    settle_price = float(last_row["near_close"])
                else:
                    settle_price = float(last_row["far_close"])
            else:
                settle_price = pos["entry_price"]

            settle_pnl_pt = settle_price - pos["entry_price"]
            settle_pnl_rmb = settle_pnl_pt * self.volume * mult
            settle_comm = settle_price * self.volume * mult * self.commission_rate
            settle_net = settle_pnl_rmb - settle_comm
            current_equity += settle_net

            rec = TradeRecord(
                timestamp=last_dt, action=TradeType.CLOSE, product=prod,
                from_symbol=pos["symbol"], to_symbol="",
                price_from=settle_price, price_to=0, volume=self.volume,
                pnl_point=settle_pnl_pt * self.volume, pnl_rmb=settle_net,
                commission=settle_comm, reason="SETTLE at end of data",
                zscore=zscore, basis_rate=adj_rate,
                near_price=near_price, far_price=far_price,
                direction_signal=sig_val, equity_after=current_equity,
                position_after="-",
            )
            self.trades.append(rec)

        n_total_bars = bar_counter
        return self._build_stats(current_equity, n_total_bars)

    # ---- 动作执行方法 ----

    def _exec_switch(
        self, dt, product, from_sym, to_sym,
        from_price, to_price, action_type, new_holding,
        zscore, adj_rate, sig_val, equity, mult,
    ) -> Tuple[float, TradeRecord]:
        """执行同品种内的近/远月切换"""
        pos = self.position
        old_pnl_pt = from_price - pos["entry_price"]
        old_pnl_rmb = old_pnl_pt * self.volume * mult
        comm_old = from_price * self.volume * mult * self.commission_rate
        comm_new = to_price * self.volume * mult * self.commission_rate
        total_comm = comm_old + comm_new
        net_pnl = old_pnl_rmb - total_comm

        rec = TradeRecord(
            timestamp=dt, action=action_type, product=product,
            from_symbol=pos["symbol"], to_symbol=to_sym,
            price_from=from_price, price_to=to_price, volume=self.volume,
            pnl_point=old_pnl_pt * self.volume, pnl_rmb=net_pnl,
            commission=total_comm, reason="",
            zscore=zscore, basis_rate=adj_rate,
            near_price=from_price if action_type == TradeType.SWITCH_FAR else to_price,
            far_price=to_price if action_type == TradeType.SWITCH_FAR else from_price,
            direction_signal=sig_val, equity_after=equity + net_pnl,
            position_after=f"{product}/{to_sym}/{new_holding}",
        )
        # 补充 reason
        if action_type == TradeType.SWITCH_FAR:
            rec.reason = f"NEAR->FAR: z={zscore:.2f}"
        else:
            rec.reason = f"FAR->NEAR: z={zscore:.2f}"

        self.position = {
            "product": product, "symbol": to_sym,
            "entry_price": to_price, "holding": new_holding,
            "entry_time": dt, "direction": 1,
        }
        return net_pnl, rec

    def _exec_rollover(
        self, dt, product, chain,
        near_price, far_price, as_of_date,
        zscore, adj_rate, sig_val, equity, mult,
    ) -> Tuple[float, TradeRecord]:
        """执行合约展期"""
        pos = self.position
        holding = pos["holding"]
        old_sym = pos["symbol"]

        # 获取新的活跃合约对
        new_near, new_far = get_active_pair(chain, as_of_date, self.rollover_days)

        if holding == "NEAR":
            # 平旧近月 → 开新近月
            exit_price = near_price
            entry_price = near_price  # 假设新旧近月价格相近（实际有微小价差）
            new_sym = new_near
            new_holding = "NEAR"
        else:
            # 平旧远月 → 开新远月
            exit_price = far_price
            entry_price = far_price
            new_sym = new_far
            new_holding = "FAR"

        # PnL: 旧持仓盈亏
        old_pnl_pt = exit_price - pos["entry_price"]
        old_pnl_rmb = old_pnl_pt * self.volume * mult

        # 手续费（平旧 + 开新）
        tick_size = 0.2
        slippage = self.slippage_ticks * tick_size
        comm_exit = (exit_price + slippage) * self.volume * mult * self.commission_rate
        comm_entry = entry_price * self.volume * mult * self.commission_rate
        total_comm = comm_exit + comm_entry

        net_pnl = old_pnl_rmb - total_comm

        reason = f"Rollover: {old_sym} -> {new_sym} (expiry near)"

        rec = TradeRecord(
            timestamp=dt, action=TradeType.ROLLOVER, product=product,
            from_symbol=old_sym, to_symbol=new_sym,
            price_from=exit_price, price_to=entry_price, volume=self.volume,
            pnl_point=old_pnl_pt * self.volume, pnl_rmb=net_pnl,
            commission=total_comm, reason=reason,
            zscore=zscore, basis_rate=adj_rate,
            near_price=near_price, far_price=far_price,
            direction_signal=sig_val, equity_after=equity + net_pnl,
            position_after=f"{product}/{new_sym}/{new_holding}",
        )

        self.position = {
            "product": product, "symbol": new_sym,
            "entry_price": entry_price, "holding": new_holding,
            "entry_time": dt, "direction": pos.get("direction", 1),
        }
        return net_pnl, rec

    def _exec_product_switch(
        self, dt, from_product, to_product,
        data_map, chains, stats_map, row,
        zscore, adj_rate, all_sigs, equity,
    ) -> Tuple[float, TradeRecord]:
        """执行跨品种切换"""
        pos = self.position
        from_mult = MULTIPLIER_MAP.get(from_product, 300)
        to_mult = MULTIPLIER_MAP.get(to_product, 300)

        # 平旧品种持仓
        if from_product in data_map:
            df_from = data_map[from_product]
            last_from = df_from.iloc[-1]  # 取最后一行的价格作为参考
            if pos["holding"] == "NEAR":
                exit_price = float(last_from["near_close"])
            else:
                exit_price = float(last_from["far_close"])
        else:
            exit_price = pos["entry_price"]

        old_pnl_pt = exit_price - pos["entry_price"]
        old_pnl_rmb = old_pnl_pt * self.volume * from_mult

        # 开新品种
        if to_product in data_map:
            df_to = data_map[to_product]
            # 找到最接近 dt 的行
            mask_to = df_to["datetime"] <= dt
            if mask_to.any():
                row_to = df_to[mask_to].iloc[-1]
                if True:  # 新品种总是开近月
                    entry_price = float(row_to["near_close"])
                    new_near_p = entry_price
                    new_far_p = float(row_to.get("far_close", entry_price))
                else:
                    entry_price = float(row_to["far_close"])
                    new_near_p = float(row_to.get("near_close", entry_price))
                    new_far_p = entry_price
            else:
                entry_price = exit_price  # fallback
                new_near_p = entry_price
                new_far_p = entry_price
        else:
            entry_price = exit_price
            new_near_p = exit_price
            new_far_p = exit_price

        # 成本：平旧手续费 + 开新手续费 + 跨品种切换成本
        comm_old = exit_price * self.volume * from_mult * self.commission_rate
        comm_new = entry_price * self.volume * to_mult * self.commission_rate
        switch_cost = exit_price * self.volume * from_mult * self.product_switch_cost_bps / 10000
        total_cost = comm_old + comm_new + switch_cost

        net_pnl = old_pnl_rmb - total_cost

        to_sig = all_sigs.get(to_product, 0)
        reason = f"Switch: {from_product}->{to_product} (signals: {all_sigs})"

        rec = TradeRecord(
            timestamp=dt, action=TradeType.SWITCH_PRODUCT,
            product=to_product,
            from_symbol=pos["symbol"],
            to_symbol=f"{to_product}{entry_price:.0f}".replace(".",""),  # placeholder
            price_from=exit_price, price_to=entry_price, volume=self.volume,
            pnl_point=old_pnl_pt * self.volume, pnl_rmb=net_pnl,
            commission=total_cost, reason=reason,
            zscore=zscore, basis_rate=adj_rate,
            near_price=new_near_p, far_price=new_far_p,
            direction_signal=to_sig, equity_after=equity + net_pnl,
            position_after=f"{to_product}/NEAR",
        )
        # 修正 to_symbol
        if to_product in data_map:
            df_to = data_map[to_product]
            mask_to = df_to["datetime"] <= dt
            if mask_to.any():
                rec.to_symbol = str(df_to[mask_to].iloc[-1]["near_symbol"])

        self.position = {
            "product": to_product,
            "symbol": rec.to_symbol,
            "entry_price": entry_price,
            "holding": "NEAR",
            "entry_time": dt,
            "direction": 1 if to_sig >= 0 else -1,
        }
        return net_pnl, rec

    # ---- 统计 & 报告 ----

    def _build_stats(self, final_eq: float, n_bars: int) -> dict:
        total_return = (final_eq - self.initial_capital) / self.initial_capital * 100

        eq_arr = np.array(self.equity_curve) if self.equity_curve else np.array([self.initial_capital])
        cummax_arr = np.maximum.accumulate(eq_arr)
        drawdown = (eq_arr - cummax_arr) / np.maximum(cummax_arr, 1) * 100
        max_dd = drawdown.min() if len(drawdown) > 0 else 0.0

        winning = [t.pnl_rmb for t in self.trades if getattr(t, 'pnl_rmb', 0) > 0]
        losing = [abs(getattr(t, 'pnl_rmb', 0)) for t in self.trades if getattr(t, 'pnl_rmb', 0) < 0]
        n_trades = len(self.trades)

        win_rate = len(winning) / max(n_trades, 1) * 100
        avg_win = np.mean(winning) if winning else 0
        avg_loss = np.mean(losing) if losing else 1e-6
        profit_factor = sum(winning) / max(sum(losing), 1)
        total_commission = sum(getattr(t, 'commission', 0) for t in self.trades)

        if self.equity_times:
            days_span = (self.equity_times[-1] - self.equity_times[0]).days
            days_span = max(days_span, 1)
            annualized = total_return * (252 / days_span)
        else:
            annualized = 0.0
            days_span = 0

        # 分类统计
        n_open = sum(1 for t in self.trades if t.action == TradeType.OPEN)
        n_switch_far = sum(1 for t in self.trades if t.action == TradeType.SWITCH_FAR)
        n_switch_near = sum(1 for t in self.trades if t.action == TradeType.SWITCH_NEAR)
        n_rollover = sum(1 for t in self.trades if t.action == TradeType.ROLLOVER)
        n_product_switch = sum(1 for t in self.trades if t.action == TradeType.SWITCH_PRODUCT)
        n_close = sum(1 for t in self.trades if t.action == TradeType.CLOSE)

        # 品种使用情况
        products_used = set()
        for t in self.trades:
            if t.product:
                products_used.add(t.product)

        stats = {
            "total_return_pct": round(total_return, 2),
            "annualized_pct": round(annualized, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "final_equity": round(final_eq, 0),
            "initial_capital": self.initial_capital,
            "n_trades": n_trades,
            "n_open": n_open,
            "n_switch_far": n_switch_far,
            "n_switch_near": n_switch_near,
            "n_rollover": n_rollover,
            "n_product_switch": n_product_switch,
            "n_close": n_close,
            "win_rate": round(win_rate, 1),
            "avg_win": round(avg_win, 0),
            "avg_loss": round(avg_loss, 0),
            "profit_factor": round(profit_factor, 2),
            "total_commission": round(total_commission, 0),
            "days_span": days_span,
            "n_bars": n_bars,
            "products_used": sorted(products_used),
        }

        # 打印摘要
        print(f"\n{'='*70}")
        print(f"  BACKTEST RESULTS V2")
        print(f"{'='*70}")
        print(f"  Initial Capital:     {self.initial_capital:>14,.0f} CNY")
        print(f"  Final Equity:        {final_eq:>14,.0f} CNY")
        print(f"  Total Return:        {total_return:>+13.2f}%")
        print(f"  Annualized:          {annualized:>+13.2f}%")
        print(f"  Max Drawdown:        {max_dd:>13.2f}%")
        print(f"  {'─'*56}")
        print(f"  Total Trades:        {n_trades:>14d}")
        print(f"    Open:              {n_open:>14d}")
        print(f"    Near->Far:         {n_switch_far:>14d}")
        print(f"    Far->Near:         {n_switch_near:>14d}")
        print(f"    Rollover:          {n_rollover:>14d}")
        print(f"    Product Switch:    {n_product_switch:>14d}")
        print(f"    Close:             {n_close:>14d}")
        print(f"  {'─'*56}")
        print(f"  Win Rate:            {win_rate:>13.1f}%")
        print(f"  Profit Factor:       {profit_factor:>13.2f}")
        print(f"  Total Commission:    {total_commission:>14,.0f} CNY")
        print(f"  Days Span:           {days_span:>14d}")
        print(f"  Bars Processed:      {n_bars:>14d}")
        print(f"  Products Used:       {', '.join(sorted(products_used)) or '-'}")
        print(f"{'='*70}")

        # 详细交易列表
        if self.trades:
            print(f"\n  TRADE LIST ({n_trades} trades):")
            print(f"  {'─'*100}")
            header = (f"  {'Time':<18s} {'Action':<16s} {'Product':<6s} "
                      f"{'From':>10s} {'To':>10s} {'PnL(CNY)':>12s} {'Equity':>13s} Reason")
            print(header)
            print(f"  {'─'*100}")
            for t in self.trades:
                pn = getattr(t, 'pnl_rmb', 0)
                eq = getattr(t, 'equity_after', 0)
                fr = f"{t.price_from:.0f}" if t.price_from else "-"
                to = f"{t.price_to:.0f}" if t.price_to else "-"
                print(f"  {t.timestamp.strftime('%m-%d %H:%M'):<18s} {t.action:<16s} "
                      f"{t.product:<6s} {fr:>10s} {to:>10s} "
                      f"{pn:>+12,.0f} {eq:>13,.0f}  "
                      f"{str(t.reason)[:40]}")

        return stats

    def generate_html_report(self, output_path: str) -> None:
        """生成 HTML 报告"""
        trades_data = []
        for t in self.trades:
            trades_data.append({
                "Time": t.timestamp.strftime("%Y-%m-%d %H:%M"),
                "Action": t.action,
                "Product": t.product,
                "From": t.from_symbol or "-",
                "To": t.to_symbol or "-",
                "Price_From": f"{t.price_from:.1f}",
                "Price_To": f"{t.price_to:.1f}",
                "PnL_CNY": f"{getattr(t,'pnl_rmb',0):+,.0f}",
                "Commission": f"{getattr(t,'commission',0):.0f}",
                "Reason": t.reason,
                "Zscore": f"{t.zscore:.2f}" if t.zscore is not None else "-",
                "BasisRate": f"{t.basis_rate:.2f}%" if t.basis_rate is not None else "-",
                "Equity": f"{getattr(t,'equity_after',0):,.0f}",
                "Position": getattr(t, 'position_after', ''),
            })

        dates = [d.strftime("%m-%d %H:%M") for d in self.equity_times]
        values = [round(v, 2) for v in self.equity_curve]

        stats = {
            "Initial Capital": f"{self.initial_capital:,} CNY",
            "Final Equity": f"{self.equity_curve[-1]:,}" if self.equity_curve else "-",
            "Total Return": f"{(self.equity_curve[-1]-self.initial_capital)/self.initial_capital*100:+.2f}%" if self.equity_curve else "-",
            "Trades": len(self.trades),
            "Commission": f"{sum(getattr(t,'commission',0) for t in self.trades):,.0f}",
        }
        stats_rows = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in stats.items())

        trades_df_html = ""
        if trades_data:
            trades_df_html = pd.DataFrame(trades_data).to_html(index=False, escape=False)

        equity_js = ""
        if dates and values:
            equity_js = f"""
<div id="eq_chart" style="width:100%; height:400px;"></div>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
new Chart(document.getElementById('eq_chart'), {{
    type:'line',
    data:{{
        labels:{json.dumps(dates)},
        datasets:[{{
            label:'Equity (CNY)',
            data:{json.dumps(values)},
            borderColor:'#2563eb',
            backgroundColor:'rgba(37,99,235,0.08)',
            fill:true, tension:0.15, pointRadius:0
        }}]
    }},
    options:{{ responsive:true, plugins:{{legend:{{display:false}}}},
        scales:{{y:{{ticks:{{callback:v=>v.toLocaleString()+' CNY'}}}}}}
    }}
}});
</script>"""

        html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>Backtest Report V2</title>
<style>
body {{ font-family:-apple-system,"Segoe UI",sans-serif; margin:20px; background:#f8fafc; }}
.card {{ background:#fff; border-radius:8px; padding:24px; margin:16px 0; box-shadow:0 1px 3px rgba(0,0,0,.08); }}
h1,h2 {{ color:#1e293b; }} table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ padding:8px 12px; text-align:left; border-bottom:1px solid #e2e8f0; }}
th {{ background:#f1f5f9; }} .pos {{ color:#dc2626; }} .neg {{ color:#16a34a; }}
.tag {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:bold; }}
.tag-open {{ bg:#dbeafe; color:#1d4ed8; }}
.tag-switch {{ bg:#fef3c7; color:#d97706; }}
.tag-rollover {{ bg:#ede9fe; color:#7c3aed; }}
tag-product {{ bg:#fce7f3; color:#db2777; }}
.tag-close {{ bg:#f1f5f9; color:#475569; }}
</style></head>
<body>
<div class="card">
<h1>Directional Calendar Spread - Backtest V2</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<p>Features: <b>Rollover</b> | <b>Multi-Product Switch</b> | <b>Z-Score Basis</b></p>
</div>

<div class="card"><h2>Summary</h2><table>{stats_rows}</table></div>

<div class="card"><h2>Equity Curve</h2>{equity_js}</div>

<div class="card"><h2>Trade Details ({len(self.trades)} trades)</h2>
{trades_df_html or '<p>No trades recorded.</p>'}
</div>

<div class="card"><h2>Strategy Logic</h2>
<ul>
<li><b>Contract Rollover:</b> Auto-rollover {self.rollover_days} days before expiry</li>
<li><b>Product Switch:</b> Based on direction signal strength x product beta</li>
<li><b>Near/Far Switch:</b> Z-score > {self.sigma_entry}σ → Far | Z-score < {self.sigma_exit}σ → Near</li>
<li><b>Cooldown:</b> {self.cooldown_bars} bars between switches</li>
</ul>
</div>

</body></html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n  HTML report saved: {output_path}")


# ------------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backtest engine V2 with rollover & multi-product support")
    parser.add_argument("--data", help="Single CSV file (compat mode)")
    parser.add_argument("--data-dir", help="Directory containing multiple product CSV files")
    parser.add_argument("--products", nargs="+", help="Products to include, e.g., IF IH IC IM")
    parser.add_argument("--fixed-signal", help="Fixed signal, format: 'IF:+1' or 'IM:+2,IH:-1'")
    parser.add_argument("--signal-file", help="Path to external signal JSON file")
    parser.add_argument("--sigma-entry", type=float, default=1.0)
    parser.add_argument("--sigma-exit", type=float, default=0.3)
    parser.add_argument("--cooldown", type=int, default=6)
    parser.add_argument("--rollover-days", type=int, default=5)
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--volume", type=int, default=1)
    parser.add_argument("--slippage", type=float, default=0.5)
    parser.add_argument("--html-report", default=None)
    args = parser.parse_args()

    # ---- 加载数据 ----
    data_map: Dict[str, pd.DataFrame] = {}

    if args.data:
        # 单文件模式（兼容 v1）
        df = pd.read_csv(args.data)
        product = df["product"].iloc[0]
        data_map[product] = df
        products = [product]
    elif args.data_dir:
        # 多文件模式
        products = args.products or ["IF", "IH", "IC", "IM"]
        for prod in products:
            # Priority: long history > regular 5min > minute_basis (legacy)
            patterns = [
                f"5min*basis_{prod}_long_*.csv",
                f"5min*basis_{prod}_*.csv",
                f"*basis_{prod}_*.csv",  # fallback
            ]
            import glob
            found = False
            for pat in patterns:
                files = sorted(glob.glob(os.path.join(args.data_dir, pat)))
                if files:
                    df = pd.read_csv(files[-1])  # 用最新文件
                    data_map[prod] = df
                    print(f"  Loaded {prod}: {files[-1]} ({len(df)} rows)")
                    found = True
                    break
            if not found:
                print(f"  [WARN] No data file found for {prod}")
            else:
                print(f"  [WARN] No data file found for {prod}")
    else:
        parser.error("Need --data or --data-dir")

    if not data_map:
        print("No data loaded, exiting.")
        return

    # ---- 方向信号 ----
    if args.fixed_signal:
        direction = DirectionSignal.from_fixed(args.fixed_signal)
        print(f"  Signal: FIXED {args.fixed_signal}")
    elif args.signal_file:
        direction = DirectionSignal.from_file(args.signal_file)
        print(f"  Signal: FILE {args.signal_file}")
    else:
        # 默认：全部看多 IF
        direction = DirectionSignal.from_fixed("IF:+1")
        print(f"  Signal: DEFAULT (IF:+1 always long)")

    # ---- 运行回测 ----
    bt = BacktestEngineV2(
        sigma_entry=args.sigma_entry,
        sigma_exit=args.sigma_exit,
        cooldown_bars=args.cooldown,
        rollover_days=args.rollover_days,
        initial_capital=args.capital,
        volume=args.volume,
        slippage_ticks=args.slippage,
        use_precomputed_stats=True,
    )

    result = bt.run(data_map, direction)

    if args.html_report:
        bt.generate_html_report(args.html_report)


if __name__ == "__main__":
    main()
