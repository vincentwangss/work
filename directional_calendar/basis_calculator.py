"""
basis_calculator.py
分红调整后的年化基差率计算模块

核心公式：
  理论远月价 = 近月价 × e^(r × T) - Σ(Di × e^(r × ti))
  其中 Di 为各成分股在 [近月交割日, 远月交割日] 区间内的分红（折算为指数点）

  年化基差率（分红调整后）= (远月价 - 理论远月价) / 近月价 / 年化剩余期限差

  正值：远月相对理论值升水（高估），看多时适合持远月
  负值：远月相对理论值贴水（低估），看多时持近月更合算
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ContractInfo:
    """合约基本信息"""
    symbol: str                  # 合约代码，如 IF2506
    product: str                 # 品种，如 IF
    expiry_date: date            # 交割日
    last_price: float            # 最新价
    bid: float = 0.0
    ask: float = 0.0


@dataclass
class DividendRecord:
    """分红记录（从券商研报录入）"""
    expiry_date: date            # 对应交割日（分红截止日）
    dividend_points: float       # 从上一交割日到本交割日区间内的累计分红（指数点）
    source: str = ""             # 数据来源（券商研报名称/日期）
    update_time: datetime = field(default_factory=datetime.now)


@dataclass
class BasisSnapshot:
    """基差快照"""
    timestamp: datetime
    product: str
    near_symbol: str
    far_symbol: str
    near_price: float
    far_price: float
    raw_basis: float             # 原始基差（点数）= 远月 - 近月
    raw_annualized_rate: float   # 原始年化基差率（%）
    dividend_adjusted_basis: float    # 分红调整后基差（点数）
    adj_annualized_rate: float        # 分红调整后年化基差率（%）
    near_expiry: date
    far_expiry: date
    days_near: int               # 近月剩余天数
    days_far: int                # 远月剩余天数
    dividend_between: float      # 两个交割日之间的分红（点数）


class BasisCalculator:
    """
    年化基差率计算器（支持分红调整）

    分红调整逻辑：
      期货定价公式：F = S × e^(rT) - PV(dividends)
      对于跨期价差：
        理论价差 = 近月价 × e^(r × ΔT) - Σ(Di)
        其中 Σ(Di) 是在 [近月交割日, 远月交割日] 区间内的分红之和（折算指数点）

      实际价差 - 理论价差 = 超额基差（即"纯粹"的市场供需/资金成本偏差）
    """

    # 无风险利率（年化），默认 0（不考虑资金成本，仅看分红影响）
    RISK_FREE_RATE: float = 0.0

    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate
        # 品种 -> 分红日程 {expiry_date: dividend_points}
        self._dividend_table: Dict[str, Dict[date, DividendRecord]] = {}
        # 历史基差序列 {product: [BasisSnapshot]}
        self._history: Dict[str, List[BasisSnapshot]] = {}

    # ------------------------------------------------------------------
    # 分红数据管理
    # ------------------------------------------------------------------

    def load_dividend_schedule(self, product: str, schedule: Dict[str, float],
                                source: str = "config") -> None:
        """
        从配置加载分红日程。

        Args:
            product: 品种代码，如 "IF"
            schedule: {日期字符串: 分红点数}，如 {"2026-06-20": 35.0}
            source: 数据来源描述
        """
        if product not in self._dividend_table:
            self._dividend_table[product] = {}

        for date_str, points in schedule.items():
            expiry = date.fromisoformat(date_str)
            record = DividendRecord(
                expiry_date=expiry,
                dividend_points=float(points),
                source=source,
            )
            self._dividend_table[product][expiry] = record
            logger.info(f"[分红] {product} {expiry} = {points:.2f}点 ({source})")

    def update_dividend(self, product: str, expiry_date: date,
                        dividend_points: float, source: str = "manual") -> None:
        """手动更新单条分红记录（研报更新时调用）"""
        if product not in self._dividend_table:
            self._dividend_table[product] = {}
        self._dividend_table[product][expiry_date] = DividendRecord(
            expiry_date=expiry_date,
            dividend_points=dividend_points,
            source=source,
        )
        logger.info(f"[分红更新] {product} {expiry_date} → {dividend_points:.2f}点 ({source})")

    def get_dividend_between(self, product: str,
                              near_expiry: date, far_expiry: date) -> float:
        """
        获取 (near_expiry, far_expiry] 区间内的累计分红预测（指数点）。

        数据来源为券商研报预测（非已实现分红），按交割日索引。
        匹配策略：
          1. 精确匹配 far_expiry
          2. 最近可用匹配：取所有记录中距 far_expiry 最近的，
             偏差 ≤ 30 个自然日则采用（研报日期与实际交割日常有1~2天误差）
          3. 都没有则返回 0.0 并发出警告
        """
        table = self._dividend_table.get(product, {})

        # 1. 精确匹配
        record = table.get(far_expiry)
        if record is not None:
            return record.dividend_points

        # 2. 最近可用匹配（≤30天容差）
        if table:
            best_date = None
            best_delta = 999
            for d in table:
                delta = abs((d - far_expiry).days)
                if delta <= 30 and delta < best_delta:
                    best_date = d
                    best_delta = delta
            if best_date is not None:
                record = table[best_date]
                logger.info(
                    f"[分红] {product} 预测匹配: 目标{far_expiry} "
                    f"-> {best_date} (差{best_delta}天, "
                    f"{record.dividend_points:.2f}pt)"
                )
                return record.dividend_points

        # 3. 无数据
        logger.warning(
            f"[分红] {product} 缺少 {far_expiry} 附近(±30天)的分红预测，"
            f"基差计算将不含分红调整（偏差可能达数十点）"
        )
        return 0.0

    # ------------------------------------------------------------------
    # 核心计算
    # ------------------------------------------------------------------

    def calc_snapshot(
        self,
        near: ContractInfo,
        far: ContractInfo,
        product: str,
        as_of: Optional[datetime] = None,
    ) -> BasisSnapshot:
        """
        计算当前基差快照（含分红调整）。

        Args:
            near: 近月合约信息
            far: 远月合约信息
            product: 品种代码
            as_of: 计算时刻，默认 now()

        Returns:
            BasisSnapshot 包含原始和调整后的年化基差率
        """
        if as_of is None:
            as_of = datetime.now()

        today = as_of.date()

        # 剩余天数（自然日）
        days_near = max((near.expiry_date - today).days, 1)
        days_far = max((far.expiry_date - today).days, 1)
        dt_near = days_near / 365.0    # 年化剩余期限
        dt_far = days_far / 365.0
        dt_diff = dt_far - dt_near     # 两合约剩余期限差（年）

        if dt_diff <= 0:
            raise ValueError(
                f"远月合约({far.symbol})到期日必须晚于近月合约({near.symbol})"
            )

        # --- 原始基差 ---
        raw_basis = far.last_price - near.last_price
        # 原始年化基差率 = 原始基差 / 近月价 / 期限差 × 100%
        raw_annualized_rate = (raw_basis / near.last_price / dt_diff) * 100.0

        # --- 分红调整 ---
        dividend_between = self.get_dividend_between(
            product, near.expiry_date, far.expiry_date
        )

        # 理论价差 = 近月价 × (e^(r×ΔT) - 1) - 分红现值
        # 分红现值 ≈ 分红 × e^(-r × t_mid)（t_mid 为近远月中点）
        t_mid = (dt_near + dt_far) / 2.0
        dividend_pv = dividend_between * math.exp(-self.risk_free_rate * t_mid)
        theoretical_spread = (
            near.last_price * (math.exp(self.risk_free_rate * dt_diff) - 1.0)
            - dividend_pv
        )

        # 分红调整后基差 = 实际价差 - 理论价差（超出理论部分）
        dividend_adjusted_basis = raw_basis - theoretical_spread

        # 分红调整后年化基差率
        adj_annualized_rate = (
            dividend_adjusted_basis / near.last_price / dt_diff
        ) * 100.0

        snapshot = BasisSnapshot(
            timestamp=as_of,
            product=product,
            near_symbol=near.symbol,
            far_symbol=far.symbol,
            near_price=near.last_price,
            far_price=far.last_price,
            raw_basis=raw_basis,
            raw_annualized_rate=raw_annualized_rate,
            dividend_adjusted_basis=dividend_adjusted_basis,
            adj_annualized_rate=adj_annualized_rate,
            near_expiry=near.expiry_date,
            far_expiry=far.expiry_date,
            days_near=days_near,
            days_far=days_far,
            dividend_between=dividend_between,
        )

        # 存入历史序列
        self._push_history(product, snapshot)

        logger.debug(
            f"[基差] {product} {near.symbol}/{far.symbol} "
            f"原始年化={raw_annualized_rate:.2f}% "
            f"分红调整后={adj_annualized_rate:.2f}% "
            f"分红贴水={dividend_between:.1f}pt"
        )
        return snapshot

    # ------------------------------------------------------------------
    # 统计量计算（μ 和 σ）
    # ------------------------------------------------------------------

    def get_stats(
        self,
        product: str,
        lookback: int = 60,
        use_adjusted: bool = True,
    ) -> Tuple[float, float, int]:
        """
        获取历史年化基差率的均值和标准差。

        Args:
            product: 品种代码
            lookback: 回溯天数
            use_adjusted: 是否用分红调整后的基差率

        Returns:
            (mean, std, count) — count 为实际有效样本数
        """
        history = self._history.get(product, [])
        if not history:
            return 0.0, 0.0, 0

        cutoff = datetime.now() - timedelta(days=lookback)
        recent = [s for s in history if s.timestamp >= cutoff]

        if len(recent) < 5:
            logger.warning(
                f"[统计] {product} 历史样本不足({len(recent)}条)，"
                f"σ阈值可能不可靠，建议先积累数据再开启换仓"
            )

        values = [
            s.adj_annualized_rate if use_adjusted else s.raw_annualized_rate
            for s in recent
        ]
        arr = np.array(values)
        return float(np.mean(arr)), float(np.std(arr, ddof=1)), len(arr)

    def get_zscore(
        self,
        current_rate: float,
        product: str,
        lookback: int = 60,
        use_adjusted: bool = True,
    ) -> Optional[float]:
        """
        计算当前基差率相对历史分布的 Z-score。
        返回 None 表示样本不足。
        """
        mean, std, count = self.get_stats(product, lookback, use_adjusted)
        if count < 5 or std < 1e-6:
            return None
        return (current_rate - mean) / std

    # ------------------------------------------------------------------
    # 历史数据管理
    # ------------------------------------------------------------------

    def _push_history(self, product: str, snapshot: BasisSnapshot) -> None:
        if product not in self._history:
            self._history[product] = []
        self._history[product].append(snapshot)

    def load_history_from_records(
        self,
        product: str,
        records: List[BasisSnapshot],
    ) -> None:
        """从持久化存储加载历史快照（启动时初始化用）"""
        self._history[product] = sorted(records, key=lambda s: s.timestamp)
        logger.info(f"[历史] {product} 加载 {len(records)} 条历史基差快照")

    def get_latest_snapshot(self, product: str) -> Optional[BasisSnapshot]:
        history = self._history.get(product)
        if not history:
            return None
        return history[-1]

    def export_history_df(self, product: str):
        """导出历史基差序列为 pandas DataFrame（用于分析/可视化）"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("需要安装 pandas: pip install pandas")

        history = self._history.get(product, [])
        if not history:
            return pd.DataFrame()

        rows = []
        for s in history:
            rows.append({
                "timestamp": s.timestamp,
                "near_symbol": s.near_symbol,
                "far_symbol": s.far_symbol,
                "near_price": s.near_price,
                "far_price": s.far_price,
                "raw_basis": s.raw_basis,
                "raw_annualized_rate": s.raw_annualized_rate,
                "dividend_between": s.dividend_between,
                "adj_basis": s.dividend_adjusted_basis,
                "adj_annualized_rate": s.adj_annualized_rate,
            })
        return pd.DataFrame(rows).set_index("timestamp")
