"""
risk_manager.py
风控模块

职责：
  - 单日最大亏损限制
  - 最大回撤保护
  - 总保证金上限
  - 临近交割自动换月前置检查
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)


class RiskManager:
    """
    风控管理器。

    所有指令在执行前必须经过风控检查：
        if not risk_manager.can_trade():
            return  # 禁止交易
    """

    def __init__(self, config: dict):
        self.max_drawdown_pct: float   = config.get("max_drawdown_pct", 5.0)
        self.max_loss_per_day: float   = config.get("max_loss_per_day", 50000)
        self.max_pos_per_inst: int     = config.get("max_position_per_instrument", 20)
        self.total_margin_limit: float = config.get("total_margin_limit", 2_000_000)

        # 运行时统计
        self._daily_pnl: float = 0.0
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        self._trading_date: date = date.today()
        self._halted: bool = False          # 是否已触发限制
        self._halt_reason: str = ""

    # ------------------------------------------------------------------
    # 状态更新
    # ------------------------------------------------------------------

    def on_new_day(self, new_date: date) -> None:
        """每日开盘前重置日内统计"""
        if new_date != self._trading_date:
            logger.info(f"[风控] 新交易日 {new_date}，重置日内统计")
            self._daily_pnl = 0.0
            self._trading_date = new_date
            self._halted = False
            self._halt_reason = ""

    def on_trade(self, realized_pnl: float) -> None:
        """成交后更新日内盈亏"""
        self._daily_pnl += realized_pnl
        self._current_equity += realized_pnl
        self._peak_equity = max(self._peak_equity, self._current_equity)

        # 检查日内亏损
        if self._daily_pnl < -self.max_loss_per_day:
            self._halt("日内亏损超限",
                       f"日内亏损 {self._daily_pnl:.0f} > 限制 {self.max_loss_per_day:.0f}")

        # 检查最大回撤
        if self._peak_equity > 0:
            drawdown_pct = (self._peak_equity - self._current_equity) / self._peak_equity * 100
            if drawdown_pct > self.max_drawdown_pct:
                self._halt("最大回撤超限",
                           f"回撤 {drawdown_pct:.2f}% > 限制 {self.max_drawdown_pct:.2f}%")

    def on_margin_update(self, total_margin: float) -> None:
        """保证金变化时检查"""
        if total_margin > self.total_margin_limit:
            self._halt("保证金超限",
                       f"总保证金 {total_margin:.0f} > 限制 {self.total_margin_limit:.0f}")

    def set_equity(self, equity: float) -> None:
        """从账户同步当前权益"""
        self._current_equity = equity
        self._peak_equity = max(self._peak_equity, equity)

    # ------------------------------------------------------------------
    # 检查接口
    # ------------------------------------------------------------------

    def can_trade(self) -> bool:
        """是否允许继续交易"""
        if self._halted:
            logger.warning(f"[风控] 交易被限制: {self._halt_reason}")
        return not self._halted

    def check_position_limit(self, product: str, new_volume: int) -> bool:
        """检查是否超出单品种持仓限制"""
        if abs(new_volume) > self.max_pos_per_inst:
            logger.warning(
                f"[风控] {product} 持仓 {new_volume}手 超过限制 {self.max_pos_per_inst}手"
            )
            return False
        return True

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _halt(self, reason: str, detail: str = "") -> None:
        if not self._halted:
            self._halted = True
            self._halt_reason = reason
            logger.error(f"[风控] 触发限制: {reason} | {detail}")

    def reset_halt(self) -> None:
        """手动解除限制（谨慎使用）"""
        self._halted = False
        self._halt_reason = ""
        logger.info("[风控] 手动解除限制")

    @property
    def status_str(self) -> str:
        return (
            f"日内PnL={self._daily_pnl:+.0f} "
            f"权益={self._current_equity:.0f} "
            f"峰值={self._peak_equity:.0f} "
            f"{'【限制中】' + self._halt_reason if self._halted else '正常'}"
        )
