"""
execution.py
VeighNa CtaTemplate 策略执行层

架构说明：
  - 继承 CtaTemplate，在 VeighNa 框架内运行
  - 订阅近月和远月合约行情
  - 每根 Bar/Tick 调用 SpreadEngine 获取换仓决策
  - 用 SpreadTrading 引擎（或手动双腿）执行换仓
  - 持仓变化实时回调更新引擎状态

注意：双腿换仓使用"先开后平"策略：
  1. 先开新合约（防止单腿成交导致敞口扩大）
  2. 新仓成交后再平旧合约
  （也可根据偏好改为先平后开）
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import yaml

from vnpy.trader.constant import Direction as VnDirection, Offset, Status
from vnpy.trader.object import BarData, OrderData, TickData, TradeData
from vnpy_ctastrategy import CtaTemplate, StopOrder

from basis_calculator import BasisCalculator, ContractInfo
from direction_signal import Direction, build_provider_from_config
from risk_manager import RiskManager
from spread_engine import SpreadEngine, SwitchAction, SwitchDecision

logger = logging.getLogger(__name__)

# VeighNa 方向映射
DIRECTION_MAP = {
    Direction.LONG:  VnDirection.LONG,
    Direction.SHORT: VnDirection.SHORT,
}


class DirectionalCalendarStrategy(CtaTemplate):
    """
    带方向的股指期货跨期套利策略（VeighNa CtaTemplate）

    参数（在 VeighNa 策略界面可配置）：
      config_file: 配置文件路径
      log_level:   日志级别
    """

    author = "DirectionalCalendar"

    # 策略参数（VeighNa 参数面板）
    config_file: str = "C:/Users/wang/WorkBuddy/20260425111208/directional_calendar/config.yaml"
    log_level: str = "INFO"

    parameters = ["config_file", "log_level"]
    variables = []

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # 延迟初始化（on_init 里完成）
        self._config: dict = {}
        self._basis_calc: Optional[BasisCalculator] = None
        self._spread_engine: Optional[SpreadEngine] = None
        self._risk_manager: Optional[RiskManager] = None

        # {product: {"near": ContractInfo, "far": ContractInfo}}
        self._contract_cache: Dict[str, Dict[str, ContractInfo]] = {}

        # 挂起的换仓任务 {order_id: SwitchDecision}
        self._pending_switches: Dict[str, SwitchDecision] = {}

        # 已开出的新腿 order_id -> 是否成交
        self._new_leg_filled: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # 策略生命周期
    # ------------------------------------------------------------------

    def on_init(self):
        self.write_log("策略初始化...")
        self._setup_logging()

        # 加载配置
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
        except Exception as e:
            self.write_log(f"配置文件加载失败: {e}", level=logging.ERROR)
            return

        # 构建各模块
        self._basis_calc = BasisCalculator(
            risk_free_rate=0.020  # 可从配置读取
        )
        direction_provider = build_provider_from_config(self._config["direction"])
        self._spread_engine = SpreadEngine(
            self._basis_calc, direction_provider, self._config
        )
        self._risk_manager = RiskManager(self._config["risk"])

        # 订阅所有合约行情
        self._subscribe_all()

        # 加载历史基差数据（如有）
        self._load_historical_basis()

        self.write_log("策略初始化完成")

    def on_start(self):
        self.write_log("策略启动")
        # 同步账户持仓到引擎
        self._sync_positions()

    def on_stop(self):
        self.write_log("策略停止")

    # ------------------------------------------------------------------
    # Tick 驱动
    # ------------------------------------------------------------------

    def on_tick(self, tick: TickData):
        """每个 Tick 更新行情缓存并触发决策"""
        product, leg = self._parse_symbol(tick.vt_symbol)
        if not product or not leg:
            return

        # 更新合约行情缓存
        if product not in self._contract_cache:
            self._contract_cache[product] = {}

        state = self._spread_engine.get_state(product) if self._spread_engine else None
        if not state:
            return

        expiry = self._get_expiry(tick.symbol)
        info = ContractInfo(
            symbol=tick.symbol,
            product=product,
            expiry_date=expiry,
            last_price=tick.last_price,
            bid=tick.bid_price_1,
            ask=tick.ask_price_1,
        )
        self._contract_cache[product][leg] = info

        # 只有近远月都有行情时才触发决策
        if "near" in self._contract_cache[product] and "far" in self._contract_cache[product]:
            self._run_decision()

    def on_bar(self, bar: BarData):
        """日线 Bar 驱动（日终决策 / 均线计算）"""
        # 如果使用均线方向，在这里推入收盘价
        product, _ = self._parse_symbol(bar.vt_symbol)
        if product:
            direction_provider = self._spread_engine.direction_provider if self._spread_engine else None
            if hasattr(direction_provider, "fallback") and hasattr(direction_provider.fallback, "push_close"):
                direction_provider.fallback.push_close(product, bar.close_price)

    # ------------------------------------------------------------------
    # 决策执行
    # ------------------------------------------------------------------

    def _run_decision(self):
        """触发换仓决策"""
        if not self._spread_engine or not self._risk_manager:
            return

        # 风控前置检查
        if not self._risk_manager.can_trade():
            logger.warning("[风控] 当日交易已受限，跳过决策")
            return

        decisions = self._spread_engine.on_tick(self._contract_cache)

        for decision in decisions:
            self._execute_decision(decision)

    def _execute_decision(self, decision: SwitchDecision):
        """执行换仓指令"""
        self.write_log(
            f"[执行] {decision.action.value}: {decision.from_symbol} → {decision.to_symbol} "
            f"z={decision.zscore:.2f if decision.zscore else 'N/A'} "
            f"basis={decision.basis_rate:.2f if decision.basis_rate else 'N/A'}%"
        )

        product = decision.product
        exec_cfg = self._config.get("execution", {})
        max_vol = exec_cfg.get("max_switch_volume", 10)

        state = self._spread_engine.get_state(product)
        if not state:
            return

        contracts = self._contract_cache.get(product, {})

        # --- 平仓 ---
        if decision.action == SwitchAction.CLOSE:
            self._close_position(product, decision)
            return

        # --- 开仓（近月或远月）---
        if decision.action in (SwitchAction.OPEN_NEAR, SwitchAction.OPEN_FAR):
            symbol = decision.to_symbol
            leg = "near" if decision.action == SwitchAction.OPEN_NEAR else "far"
            info = contracts.get(leg)
            if not info:
                self.write_log(f"[执行] {symbol} 行情不存在，跳过", level=logging.WARNING)
                return
            vn_dir = DIRECTION_MAP.get(decision.direction, VnDirection.LONG)
            volume = min(max_vol, state.position_volume if state.position_volume > 0 else max_vol)
            price = info.ask if decision.direction == Direction.LONG else info.bid
            order_ids = self.buy(symbol, price, volume) \
                if decision.direction == Direction.LONG \
                else self.short(symbol, price, volume)
            for oid in order_ids:
                self._pending_switches[oid] = decision
            return

        # --- 换仓（先开新腿，再平旧腿）---
        if decision.action in (
            SwitchAction.SWITCH_TO_FAR,
            SwitchAction.SWITCH_TO_NEAR,
            SwitchAction.ROLLOVER,
        ):
            self._two_leg_switch(decision, contracts, max_vol)

    def _two_leg_switch(
        self,
        decision: SwitchDecision,
        contracts: Dict[str, Dict[str, ContractInfo]],
        max_vol: int,
    ):
        """双腿换仓：先开新腿，新腿成交后平旧腿"""
        state = self._spread_engine.get_state(decision.product)
        if not state:
            return

        new_leg = "far" if decision.action in (
            SwitchAction.SWITCH_TO_FAR, SwitchAction.ROLLOVER
        ) else "near"
        old_leg = "near" if new_leg == "far" else "far"

        product_contracts = contracts.get(decision.product, {})
        new_info = product_contracts.get(new_leg)
        old_info = product_contracts.get(old_leg)

        if not new_info or not old_info:
            self.write_log(
                f"[换仓] {decision.product} 行情数据缺失，跳过",
                level=logging.WARNING
            )
            return

        volume = min(max_vol, abs(state.position_volume))
        if volume == 0:
            volume = 1  # 至少 1 手

        # 开新腿
        if decision.direction == Direction.LONG:
            price = new_info.ask
            order_ids = self.buy(new_info.symbol, price, volume)
        else:
            price = new_info.bid
            order_ids = self.short(new_info.symbol, price, volume)

        # 记录挂单，等成交后再平旧腿
        for oid in order_ids:
            self._new_leg_filled[oid] = False
            # 把旧腿信息打包进 metadata 待用
            decision.metadata = {
                "old_symbol": old_info.symbol,
                "old_direction": decision.direction,
                "volume": volume,
            }
            self._pending_switches[oid] = decision

        self.write_log(
            f"[双腿换仓] {decision.product}: 开{new_info.symbol} {volume}手 "
            f"@ {price:.1f}，待成交后平{old_info.symbol}"
        )

    def _close_position(self, product: str, decision: SwitchDecision):
        """平仓"""
        state = self._spread_engine.get_state(product)
        if not state or state.position_volume == 0:
            return
        contracts = self._contract_cache.get(product, {})
        leg = "near" if state.holding.name == "NEAR" else "far"
        info = contracts.get(leg)
        if not info:
            return

        volume = abs(state.position_volume)
        if decision.direction == Direction.LONG:
            price = info.bid
            order_ids = self.sell(info.symbol, price, volume)
        else:
            price = info.ask
            order_ids = self.cover(info.symbol, price, volume)

        for oid in order_ids:
            self._pending_switches[oid] = decision
        self.write_log(f"[平仓] {product} {info.symbol} {volume}手 @ {price:.1f}")

    # ------------------------------------------------------------------
    # 成交回调
    # ------------------------------------------------------------------

    def on_trade(self, trade: TradeData):
        """成交后：如果是换仓的新腿成交，立即触发平旧腿"""
        order_id = trade.orderid
        if order_id not in self._pending_switches:
            return

        decision = self._pending_switches.pop(order_id)

        # 风控记录
        if self._risk_manager:
            pnl = self._estimate_trade_pnl(trade)
            self._risk_manager.on_trade(pnl)

        # 如果是换仓新腿，触发平旧腿
        if decision.action in (
            SwitchAction.SWITCH_TO_FAR,
            SwitchAction.SWITCH_TO_NEAR,
            SwitchAction.ROLLOVER,
        ) and "old_symbol" in decision.metadata:
            old_symbol = decision.metadata["old_symbol"]
            old_dir = decision.metadata["old_direction"]
            volume = decision.metadata["volume"]

            # 平旧腿
            if old_dir == Direction.LONG:
                old_info = self._find_contract_by_symbol(decision.product, old_symbol)
                price = old_info.bid if old_info else trade.price * 0.999
                self.sell(old_symbol, price, volume)
            else:
                old_info = self._find_contract_by_symbol(decision.product, old_symbol)
                price = old_info.ask if old_info else trade.price * 1.001
                self.cover(old_symbol, price, volume)

            self.write_log(
                f"[双腿换仓] 新腿{trade.symbol}成交，触发平旧腿{old_symbol} {volume}手"
            )

        # 更新引擎持仓状态
        if self._spread_engine:
            net_vol = (trade.volume if trade.direction == VnDirection.LONG
                       else -trade.volume)
            self._spread_engine.update_position(
                decision.product, trade.symbol, net_vol
            )

    def on_order(self, order: OrderData):
        """委托回调：撤单处理"""
        if order.status in (Status.CANCELLED, Status.REJECTED):
            oid = order.orderid
            if oid in self._pending_switches:
                decision = self._pending_switches.pop(oid)
                self.write_log(
                    f"[委托] {order.symbol} 委托{order.status.value}，"
                    f"换仓任务取消: {decision.reason}",
                    level=logging.WARNING
                )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _subscribe_all(self):
        """订阅所有配置品种的近远月行情"""
        instruments = self._config.get("instruments", {})
        for product, cfg in instruments.items():
            if cfg.get("enabled", False):
                near = cfg["near_symbol"]
                far = cfg["far_symbol"]
                self.subscribe_data(near)
                self.subscribe_data(far)
                self.write_log(f"[订阅] {product}: {near}, {far}")

    def subscribe_data(self, symbol: str):
        """封装 VeighNa 行情订阅"""
        try:
            # VeighNa 5.x 行情订阅方式
            self.cta_engine.main_engine.subscribe(
                symbol, self.cta_engine.gateway_name
            )
        except Exception as e:
            self.write_log(f"[订阅] {symbol} 失败: {e}", level=logging.WARNING)

    def _parse_symbol(self, vt_symbol: str):
        """从合约代码解析品种和近远月标记"""
        symbol = vt_symbol.split(".")[0]
        for product in self._config.get("instruments", {}):
            cfg = self._config["instruments"][product]
            if symbol == cfg.get("near_symbol"):
                return product, "near"
            if symbol == cfg.get("far_symbol"):
                return product, "far"
        return None, None

    def _get_expiry(self, symbol: str) -> "date":
        """从合约代码推算交割日（第三个周五）"""
        from datetime import date
        try:
            # 解析合约代码，如 IF2506 -> 2025年6月
            year = int("20" + symbol[2:4])
            month = int(symbol[4:6])
            return self._third_friday(year, month)
        except Exception:
            return date.today()

    @staticmethod
    def _third_friday(year: int, month: int) -> "date":
        """计算某月第三个周五（股指期货交割日）"""
        from datetime import date
        d = date(year, month, 1)
        # 找到第一个周五
        days_to_friday = (4 - d.weekday()) % 7
        first_friday = d.replace(day=1 + days_to_friday)
        # 第三个周五
        return first_friday.replace(day=first_friday.day + 14)

    def _find_contract_by_symbol(self, product: str, symbol: str) -> Optional[ContractInfo]:
        contracts = self._contract_cache.get(product, {})
        for info in contracts.values():
            if info.symbol == symbol:
                return info
        return None

    def _estimate_trade_pnl(self, trade: TradeData) -> float:
        """粗估成交盈亏（用于风控记录）"""
        product = trade.symbol[:2]
        multiplier = (
            self._config.get("instruments", {})
            .get(product, {})
            .get("multiplier", 300)
        )
        sign = 1 if trade.direction == VnDirection.LONG else -1
        return sign * trade.volume * multiplier * trade.price

    def _sync_positions(self):
        """启动时从账户同步持仓到引擎（防状态不一致）"""
        try:
            positions = self.cta_engine.main_engine.get_all_positions()
            for pos in positions:
                product = pos.symbol[:2]
                if product in self._config.get("instruments", {}):
                    net = pos.volume - pos.short_volume if hasattr(pos, "short_volume") else pos.volume
                    if self._spread_engine:
                        self._spread_engine.update_position(product, pos.symbol, net)
            self.write_log("[初始化] 账户持仓同步完成")
        except Exception as e:
            self.write_log(f"[初始化] 持仓同步失败: {e}", level=logging.WARNING)

    def _load_historical_basis(self):
        """加载历史基差数据（如有持久化文件）"""
        # 预留接口：可从 CSV/数据库 加载历史快照初始化统计量
        pass

    def _setup_logging(self):
        log_cfg = self._config.get("logging", {})
        level = getattr(logging, log_cfg.get("level", "INFO"), logging.INFO)
        log_dir = log_cfg.get("log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(
            os.path.join(log_dir, f"strategy_{datetime.now():%Y%m%d}.log"),
            encoding="utf-8"
        )
        handler.setLevel(level)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(fmt)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(level)
