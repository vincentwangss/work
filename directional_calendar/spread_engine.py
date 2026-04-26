"""
spread_engine.py
换仓决策引擎（多品种）

职责：
  - 综合方向信号 + 基差 Z-score，生成换仓指令
  - 多品种独立决策，各品种状态隔离
  - 换仓冷却保护，防止频繁切换
  - 临近交割自动换月
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional

from basis_calculator import BasisCalculator, BasisSnapshot, ContractInfo
from direction_signal import Direction, DirectionProvider, DirectionSignal

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 持仓状态
# ------------------------------------------------------------------

class HoldingType(Enum):
    """当前持有的合约类型"""
    NEAR   = auto()   # 持有近月合约
    FAR    = auto()   # 持有远月合约（已换仓）
    NONE   = auto()   # 空仓


@dataclass
class ProductState:
    """单品种当前状态"""
    product: str
    holding: HoldingType = HoldingType.NONE
    direction: Direction = Direction.FLAT
    position_volume: int = 0          # 持仓手数（正=多头，负=空头）
    near_symbol: str = ""
    far_symbol: str = ""
    last_switch_time: Optional[datetime] = None
    last_basis_snapshot: Optional[BasisSnapshot] = None


# ------------------------------------------------------------------
# 换仓指令
# ------------------------------------------------------------------

class SwitchAction(Enum):
    SWITCH_TO_FAR  = "switch_to_far"    # 换仓到远月
    SWITCH_TO_NEAR = "switch_to_near"   # 换回近月
    HOLD           = "hold"             # 维持现状
    ROLLOVER       = "rollover"         # 临近交割换月（强制）
    OPEN_NEAR      = "open_near"        # 新开近月（方向刚出现）
    OPEN_FAR       = "open_far"         # 新开远月（方向+基差信号同时出现）
    CLOSE          = "close"            # 平仓（方向消失）


@dataclass
class SwitchDecision:
    """换仓决策结果"""
    product: str
    action: SwitchAction
    reason: str
    from_symbol: str = ""
    to_symbol: str = ""
    volume: int = 0              # 换仓手数（0=全部）
    zscore: Optional[float] = None
    basis_rate: Optional[float] = None
    direction: Direction = Direction.FLAT
    timestamp: datetime = field(default_factory=datetime.now)
    urgency: str = "normal"      # "normal" | "urgent"（强制换月时为 urgent）

    def __str__(self):
        _z = f"{self.zscore:.2f}" if self.zscore is not None else "N/A"
        _b = f"{self.basis_rate:.2f}" if self.basis_rate is not None else "N/A"
        return (
            f"[{self.product}] {self.action.value}: "
            f"{self.from_symbol} -> {self.to_symbol} "
            f"| z={_z} "
            f"| basis={_b}% "
            f"| dir={self.direction.name} "
            f"| reason: {self.reason}"
        )


# ------------------------------------------------------------------
# 换仓决策引擎
# ------------------------------------------------------------------

class SpreadEngine:
    """
    多品种换仓决策引擎。

    决策逻辑：
    ┌─────────────────────────────────────────────────────────────┐
    │  状态机：NONE → 开仓 → NEAR/FAR → 换仓 → 平仓 → NONE       │
    └─────────────────────────────────────────────────────────────┘

    换仓条件（看多为例）：
      1. 当前持近月多头
      2. 远月年化基差率（分红调整后）的 Z-score > +sigma_entry
         即：远月相对历史过度升水，买远月性价比高
      3. 换仓成本（手续费 + 冲击）< 预期基差收益
      → 卖近买远（SWITCH_TO_FAR）

    换回条件：
      - Z-score < sigma_exit（基差回归，优势消失）
      - 方向信号消失（平仓）
      - 临近交割（强制换月）
    """

    def __init__(
        self,
        basis_calc: BasisCalculator,
        direction_provider: DirectionProvider,
        config: dict,
    ):
        self.basis_calc = basis_calc
        self.direction_provider = direction_provider
        self.config = config

        # 参数
        self.sigma_entry: float = config["basis"]["sigma_entry"]
        self.sigma_exit: float = config["basis"]["sigma_exit"]
        self.lookback_days: int = config["basis"]["lookback_days"]
        self.use_adjusted: bool = config["basis"]["use_dividend_adjusted"]
        self.min_annualized_spread: float = config["basis"]["min_annualized_spread"]
        self.rollover_days: int = config["risk"]["rollover_days_before_expiry"]
        self.cooldown_min: int = config["execution"]["rebalance_cooldown_min"]

        # 品种状态表
        self._states: Dict[str, ProductState] = {}

        # 初始化品种
        instruments = config.get("instruments", {})
        for product, inst_cfg in instruments.items():
            if inst_cfg.get("enabled", False):
                self._states[product] = ProductState(
                    product=product,
                    near_symbol=inst_cfg["near_symbol"],
                    far_symbol=inst_cfg["far_symbol"],
                )
                # 加载分红日程
                dividend_schedule = inst_cfg.get("dividend_schedule", {})
                if dividend_schedule:
                    self.basis_calc.load_dividend_schedule(
                        product, dividend_schedule, source="config"
                    )
        logger.info(f"[SpreadEngine] 初始化品种: {list(self._states.keys())}")

    # ------------------------------------------------------------------
    # 主决策入口
    # ------------------------------------------------------------------

    def on_tick(
        self,
        contracts: Dict[str, Dict[str, ContractInfo]],
        as_of: Optional[datetime] = None,
    ) -> List[SwitchDecision]:
        """
        每个 Tick/Bar 调用，返回所有品种的换仓决策列表。

        Args:
            contracts: {product: {"near": ContractInfo, "far": ContractInfo}}
            as_of: 当前时刻

        Returns:
            需要执行的换仓决策列表（action=HOLD 的不返回）
        """
        if as_of is None:
            as_of = datetime.now()

        products = list(self._states.keys())
        direction_signals = self.direction_provider.fetch(products)

        decisions = []
        for product, state in self._states.items():
            c = contracts.get(product)
            if not c or "near" not in c or "far" not in c:
                logger.debug(f"[{product}] 行情数据缺失，跳过")
                continue

            try:
                decision = self._decide_product(
                    product, state, c["near"], c["far"],
                    direction_signals.get(product), as_of
                )
                if decision and decision.action != SwitchAction.HOLD:
                    decisions.append(decision)
                    logger.info(str(decision))
            except Exception as e:
                logger.error(f"[{product}] 决策异常: {e}", exc_info=True)

        return decisions

    # ------------------------------------------------------------------
    # 单品种决策
    # ------------------------------------------------------------------

    def _decide_product(
        self,
        product: str,
        state: ProductState,
        near: ContractInfo,
        far: ContractInfo,
        dir_signal: Optional[DirectionSignal],
        as_of: datetime,
    ) -> SwitchDecision:
        """单品种决策状态机"""
        today = as_of.date()

        # --- 0. 更新合约代码 ---
        state.near_symbol = near.symbol
        state.far_symbol = far.symbol

        # --- 1. 强制换月检查（最高优先级）---
        days_to_near_expiry = (near.expiry_date - today).days
        if days_to_near_expiry <= self.rollover_days and state.holding == HoldingType.NEAR:
            return SwitchDecision(
                product=product,
                action=SwitchAction.ROLLOVER,
                reason=f"临近交割({days_to_near_expiry}日)，强制换月",
                from_symbol=near.symbol,
                to_symbol=far.symbol,
                direction=state.direction,
                urgency="urgent",
            )

        # --- 2. 方向信号 ---
        if dir_signal is None or dir_signal.direction == Direction.FLAT:
            # 无方向信号
            if state.holding != HoldingType.NONE:
                return SwitchDecision(
                    product=product,
                    action=SwitchAction.CLOSE,
                    reason="方向信号消失，平仓",
                    from_symbol=state.near_symbol if state.holding == HoldingType.NEAR else state.far_symbol,
                    direction=Direction.FLAT,
                )
            return SwitchDecision(product=product, action=SwitchAction.HOLD,
                                  reason="无方向信号，空仓等待")

        current_direction = dir_signal.direction

        # 方向反转时先平仓
        if (state.holding != HoldingType.NONE and
                state.direction != Direction.FLAT and
                state.direction != current_direction):
            return SwitchDecision(
                product=product,
                action=SwitchAction.CLOSE,
                reason=f"方向反转 {state.direction.name}→{current_direction.name}，先平仓",
                from_symbol=state.near_symbol if state.holding == HoldingType.NEAR else state.far_symbol,
                direction=current_direction,
            )

        # --- 3. 计算基差 ---
        snapshot = self.basis_calc.calc_snapshot(near, far, product, as_of)
        state.last_basis_snapshot = snapshot
        current_rate = (snapshot.adj_annualized_rate
                        if self.use_adjusted else snapshot.raw_annualized_rate)

        # Z-score
        zscore = self.basis_calc.get_zscore(current_rate, product, self.lookback_days,
                                             self.use_adjusted)

        # 冷却期检查
        in_cooldown = False
        if state.last_switch_time:
            elapsed = (as_of - state.last_switch_time).total_seconds() / 60.0
            if elapsed < self.cooldown_min:
                in_cooldown = True
                logger.debug(f"[{product}] 冷却中({elapsed:.0f}分钟/{self.cooldown_min}分钟)")

        # --- 4. 状态机决策 ---
        return self._state_machine(
            product, state, current_direction, current_rate, zscore,
            near, far, in_cooldown, as_of
        )

    def _state_machine(
        self,
        product: str,
        state: ProductState,
        direction: Direction,
        basis_rate: float,
        zscore: Optional[float],
        near: ContractInfo,
        far: ContractInfo,
        in_cooldown: bool,
        as_of: datetime,
    ) -> SwitchDecision:
        holding = state.holding

        # 辅助：构建决策对象
        def _dec(action, reason, from_sym="", to_sym=""):
            d = SwitchDecision(
                product=product, action=action, reason=reason,
                from_symbol=from_sym, to_symbol=to_sym,
                zscore=zscore, basis_rate=basis_rate,
                direction=direction, timestamp=as_of,
            )
            if action != SwitchAction.HOLD:
                state.direction = direction
                if action in (SwitchAction.SWITCH_TO_FAR, SwitchAction.OPEN_FAR):
                    state.holding = HoldingType.FAR
                elif action in (SwitchAction.SWITCH_TO_NEAR, SwitchAction.OPEN_NEAR):
                    state.holding = HoldingType.NEAR
                elif action == SwitchAction.CLOSE:
                    state.holding = HoldingType.NONE
                    state.direction = Direction.FLAT
                elif action == SwitchAction.ROLLOVER:
                    # 换月后持仓类型不变（但实际合约已滚到下一档）
                    pass
                state.last_switch_time = as_of
            return d

        zscore_valid = zscore is not None

        # 当前空仓
        if holding == HoldingType.NONE:
            if (direction == Direction.LONG and zscore_valid and
                    zscore > self.sigma_entry and
                    basis_rate > self.min_annualized_spread):
                return _dec(SwitchAction.OPEN_FAR,
                            f"空仓+看多+远月超升水(z={zscore:.2f})，直接开远月",
                            to_sym=far.symbol)
            else:
                return _dec(SwitchAction.OPEN_NEAR,
                            f"空仓+看{direction.name}，开近月",
                            to_sym=near.symbol)

        # 当前持近月
        if holding == HoldingType.NEAR:
            if direction == Direction.LONG:
                # 看多持近月：当远月超升水时换到远月
                if (not in_cooldown and zscore_valid and
                        zscore > self.sigma_entry and
                        basis_rate > self.min_annualized_spread):
                    return _dec(SwitchAction.SWITCH_TO_FAR,
                                f"看多+远月超升水(z={zscore:.2f} > +{self.sigma_entry}σ)，换仓到远月",
                                from_sym=near.symbol, to_sym=far.symbol)
            elif direction == Direction.SHORT:
                # 看空持近月：当远月超贴水时换到远月
                if (not in_cooldown and zscore_valid and
                        zscore < -self.sigma_entry and
                        basis_rate < -self.min_annualized_spread):
                    return _dec(SwitchAction.SWITCH_TO_FAR,
                                f"看空+远月超贴水(z={zscore:.2f} < -{self.sigma_entry}σ)，换仓到远月做空",
                                from_sym=near.symbol, to_sym=far.symbol)
            return _dec(SwitchAction.HOLD, f"HOLD near, z={zscore:.2f}")

        # 当前持远月
        if holding == HoldingType.FAR:
            if direction == Direction.LONG:
                # 远月升水已回归，换回近月
                if (not in_cooldown and zscore_valid and zscore < self.sigma_exit):
                    return _dec(SwitchAction.SWITCH_TO_NEAR,
                                f"看多+远月升水回归(z={zscore:.2f} < {self.sigma_exit}σ)，换回近月",
                                from_sym=far.symbol, to_sym=near.symbol)
            elif direction == Direction.SHORT:
                # 远月贴水已回归
                if (not in_cooldown and zscore_valid and zscore > -self.sigma_exit):
                    return _dec(SwitchAction.SWITCH_TO_NEAR,
                                f"看空+远月贴水回归(z={zscore:.2f} > -{self.sigma_exit}σ)，换回近月",
                                from_sym=far.symbol, to_sym=near.symbol)
            return _dec(SwitchAction.HOLD, f"HOLD far, z={zscore:.2f}")

        # 兜底
        return _dec(SwitchAction.HOLD, "兜底：维持现状")

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_state(self, product: str) -> Optional[ProductState]:
        return self._states.get(product)

    def get_all_states(self) -> Dict[str, ProductState]:
        return dict(self._states)

    def update_contract(self, product: str, near_symbol: str, far_symbol: str) -> None:
        """更新品种的近/远月合约代码（换月时调用）"""
        if product in self._states:
            self._states[product].near_symbol = near_symbol
            self._states[product].far_symbol = far_symbol
            logger.info(f"[{product}] 合约更新: near={near_symbol} far={far_symbol}")

    def update_position(self, product: str, symbol: str, volume: int) -> None:
        """从交易账户同步实际持仓（防止系统状态与账户不一致）"""
        state = self._states.get(product)
        if not state:
            return
        state.position_volume = volume
        if symbol == state.near_symbol:
            state.holding = HoldingType.NEAR if volume != 0 else HoldingType.NONE
        elif symbol == state.far_symbol:
            state.holding = HoldingType.FAR if volume != 0 else HoldingType.NONE
        logger.debug(f"[{product}] 持仓同步: {symbol} {volume}手 → {state.holding.name}")

    def get_decision_summary(self) -> str:
        """输出当前状态摘要（监控用）"""
        lines = ["=" * 60, "换仓引擎状态摘要", "=" * 60]
        for product, state in self._states.items():
            snap = state.last_basis_snapshot
            lines.append(
                f"{product:4s} | {state.holding.name:6s} | {state.direction.name:6s} | "
                f"adj_basis={snap.adj_annualized_rate:+.2f}% " if snap
                else f"{product:4s} | {state.holding.name:6s} | 无基差数据"
            )
        return "\n".join(lines)
