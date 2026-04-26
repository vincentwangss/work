"""
direction_signal.py
方向信号接口（插件化设计）

设计原则：
  - 核心抽象：DirectionProvider（接口），返回 {品种: DirectionSignal}
  - 内置实现：ExternalFileProvider（读 JSON 文件）、MAProvider（均线）
  - 可扩展：继承 DirectionProvider 接入任意信号源（HTTP API、数据库等）
  - 降级机制：外部文件超时/不存在时自动降级到内置均线
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 信号枚举
# ------------------------------------------------------------------

class Direction(IntEnum):
    LONG   =  1    # 看多，持有多头仓位
    SHORT  = -1    # 看空，持有空头仓位
    FLAT   =  0    # 中性，不持仓


@dataclass
class DirectionSignal:
    """单品种方向信号"""
    product: str
    direction: Direction
    confidence: float = 1.0        # 信心度 [0, 1]，可用于调仓比例
    source: str = ""               # 信号来源说明
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)  # 附加信息（如具体指标值）

    def is_valid(self, ttl_minutes: int = 60) -> bool:
        """检查信号是否在有效期内"""
        return (datetime.now() - self.timestamp).total_seconds() < ttl_minutes * 60

    def __str__(self):
        return (
            f"[{self.product}] {self.direction.name} "
            f"conf={self.confidence:.2f} @ {self.timestamp:%H:%M:%S} ({self.source})"
        )


# ------------------------------------------------------------------
# 抽象接口
# ------------------------------------------------------------------

class DirectionProvider(ABC):
    """
    方向信号提供者基类。
    子类只需实现 fetch() 方法。
    """

    @abstractmethod
    def fetch(self, products: List[str]) -> Dict[str, DirectionSignal]:
        """
        获取指定品种列表的方向信号。

        Args:
            products: 品种代码列表，如 ["IF", "IH", "IC"]

        Returns:
            {product: DirectionSignal}，缺失的品种返回 FLAT 信号
        """
        ...

    def fetch_one(self, product: str) -> DirectionSignal:
        return self.fetch([product]).get(
            product,
            DirectionSignal(product=product, direction=Direction.FLAT, source="missing")
        )


# ------------------------------------------------------------------
# 实现 1：外部 JSON 文件（你的择时策略写入这个文件）
# ------------------------------------------------------------------

class ExternalFileProvider(DirectionProvider):
    """
    从 JSON 文件读取方向信号。

    你的择时策略只需把信号写成如下格式，本系统自动读取：

    {
        "timestamp": "2026-04-25T17:00:00",
        "signals": {
            "IF": {"direction": 1,  "confidence": 0.85, "reason": "MA金叉"},
            "IH": {"direction": 1,  "confidence": 0.70, "reason": "动量正"},
            "IC": {"direction": 0,  "confidence": 0.5,  "reason": "震荡"},
            "IM": {"direction": -1, "confidence": 0.60, "reason": "破位"}
        }
    }

    direction: 1=多, -1=空, 0=中性
    """

    def __init__(self, file_path: str, ttl_minutes: int = 60):
        self.file_path = file_path
        self.ttl_minutes = ttl_minutes
        self._cache: Optional[Dict] = None
        self._cache_mtime: float = 0.0

    def fetch(self, products: List[str]) -> Dict[str, DirectionSignal]:
        raw = self._load()
        if raw is None:
            logger.warning(f"[信号] 外部文件不可用，返回全 FLAT")
            return {p: DirectionSignal(p, Direction.FLAT, source="file_unavailable")
                    for p in products}

        ts_str = raw.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            ts = datetime.now()
            logger.warning(f"[信号] 无法解析时间戳 '{ts_str}'，使用当前时间")

        # 检查有效期
        age = (datetime.now() - ts).total_seconds() / 60.0
        if age > self.ttl_minutes:
            logger.warning(
                f"[信号] 外部信号已过期 ({age:.0f}分钟 > {self.ttl_minutes}分钟)，降级为 FLAT"
            )
            return {p: DirectionSignal(p, Direction.FLAT, source="signal_expired")
                    for p in products}

        raw_signals = raw.get("signals", {})
        result = {}
        for product in products:
            if product in raw_signals:
                s = raw_signals[product]
                d = Direction(int(s.get("direction", 0)))
                conf = float(s.get("confidence", 1.0))
                reason = s.get("reason", "")
                result[product] = DirectionSignal(
                    product=product,
                    direction=d,
                    confidence=conf,
                    source=f"external_file:{reason}",
                    timestamp=ts,
                    metadata=s,
                )
            else:
                result[product] = DirectionSignal(
                    product=product, direction=Direction.FLAT,
                    source="not_in_file", timestamp=ts
                )
        return result

    def _load(self) -> Optional[dict]:
        if not os.path.exists(self.file_path):
            logger.warning(f"[信号] 外部文件不存在: {self.file_path}")
            return None
        try:
            mtime = os.path.getmtime(self.file_path)
            if mtime != self._cache_mtime:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                self._cache_mtime = mtime
                logger.info(f"[信号] 重新加载外部信号文件: {self.file_path}")
            return self._cache
        except Exception as e:
            logger.error(f"[信号] 读取外部文件失败: {e}")
            return None

    def write_signal(self, signals: Dict[str, DirectionSignal]) -> None:
        """
        便捷方法：将信号写入文件（你的择时策略可直接调用这个方法）

        Usage:
            provider = ExternalFileProvider("path/to/direction.json")
            provider.write_signal({
                "IF": DirectionSignal("IF", Direction.LONG, 0.8, "MA金叉"),
                "IH": DirectionSignal("IH", Direction.LONG, 0.7, "动量正"),
            })
        """
        data = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "signals": {
                p: {
                    "direction": int(s.direction),
                    "confidence": s.confidence,
                    "reason": s.source,
                }
                for p, s in signals.items()
            },
        }
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"[信号] 已写入信号文件: {self.file_path}")


# ------------------------------------------------------------------
# 实现 2：内置均线方向（降级用）
# ------------------------------------------------------------------

class MAProvider(DirectionProvider):
    """
    基于均线穿越的简单方向判断（作为外部信号不可用时的 fallback）。
    数据通过 vnpy 的 BarManager 注入。
    """

    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        self.fast_period = fast_period
        self.slow_period = slow_period
        # {product: [close_prices]}
        self._price_buffer: Dict[str, List[float]] = {}

    def push_close(self, product: str, close: float) -> None:
        """推入最新收盘价（每根 K 线调用一次）"""
        if product not in self._price_buffer:
            self._price_buffer[product] = []
        buf = self._price_buffer[product]
        buf.append(close)
        # 保留足够历史
        max_len = self.slow_period * 3
        if len(buf) > max_len:
            self._price_buffer[product] = buf[-max_len:]

    def fetch(self, products: List[str]) -> Dict[str, DirectionSignal]:
        result = {}
        for product in products:
            buf = self._price_buffer.get(product, [])
            if len(buf) < self.slow_period:
                result[product] = DirectionSignal(
                    product=product, direction=Direction.FLAT,
                    source=f"ma_insufficient_data({len(buf)}/{self.slow_period})"
                )
                continue

            fast_ma = sum(buf[-self.fast_period:]) / self.fast_period
            slow_ma = sum(buf[-self.slow_period:]) / self.slow_period
            prev_fast = sum(buf[-(self.fast_period+1):-1]) / self.fast_period
            prev_slow = sum(buf[-(self.slow_period+1):-1]) / self.slow_period

            # 金叉/死叉
            if prev_fast <= prev_slow and fast_ma > slow_ma:
                direction = Direction.LONG
                reason = f"MA金叉 fast={fast_ma:.1f} slow={slow_ma:.1f}"
            elif prev_fast >= prev_slow and fast_ma < slow_ma:
                direction = Direction.SHORT
                reason = f"MA死叉 fast={fast_ma:.1f} slow={slow_ma:.1f}"
            elif fast_ma > slow_ma:
                direction = Direction.LONG
                reason = f"MA多头排列 fast={fast_ma:.1f} slow={slow_ma:.1f}"
            else:
                direction = Direction.SHORT
                reason = f"MA空头排列 fast={fast_ma:.1f} slow={slow_ma:.1f}"

            result[product] = DirectionSignal(
                product=product,
                direction=direction,
                confidence=abs(fast_ma - slow_ma) / slow_ma,  # 偏离度作为信心度
                source=reason,
            )
        return result


# ------------------------------------------------------------------
# 组合提供者：外部信号 + 自动降级到均线
# ------------------------------------------------------------------

class CompositeProvider(DirectionProvider):
    """
    组合信号提供者：优先使用外部文件，不可用时降级到均线。
    """

    def __init__(
        self,
        primary: DirectionProvider,
        fallback: DirectionProvider,
        ttl_minutes: int = 60,
    ):
        self.primary = primary
        self.fallback = fallback
        self.ttl_minutes = ttl_minutes

    def fetch(self, products: List[str]) -> Dict[str, DirectionSignal]:
        primary_signals = self.primary.fetch(products)

        result = {}
        fallback_needed = []

        for product in products:
            sig = primary_signals.get(product)
            if sig and sig.is_valid(self.ttl_minutes) and sig.direction != Direction.FLAT:
                result[product] = sig
            else:
                fallback_needed.append(product)

        if fallback_needed:
            logger.info(f"[信号] 降级到均线: {fallback_needed}")
            fallback_signals = self.fallback.fetch(fallback_needed)
            result.update(fallback_signals)

        return result


# ------------------------------------------------------------------
# 工厂函数
# ------------------------------------------------------------------

def build_provider_from_config(config: dict) -> DirectionProvider:
    """
    根据配置构建方向信号提供者。

    Args:
        config: config.yaml 中的 direction 段

    Returns:
        DirectionProvider 实例
    """
    source = config.get("signal_source", "external")
    ttl = config.get("signal_ttl_minutes", 60)

    if source == "external":
        file_path = config["external_signal_file"]
        return ExternalFileProvider(file_path, ttl_minutes=ttl)

    elif source == "ma":
        ma_cfg = config.get("fallback_ma", {})
        return MAProvider(
            fast_period=ma_cfg.get("fast_period", 5),
            slow_period=ma_cfg.get("slow_period", 20),
        )

    elif source == "composite":
        file_path = config["external_signal_file"]
        ma_cfg = config.get("fallback_ma", {})
        primary = ExternalFileProvider(file_path, ttl_minutes=ttl)
        fallback = MAProvider(
            fast_period=ma_cfg.get("fast_period", 5),
            slow_period=ma_cfg.get("slow_period", 20),
        )
        return CompositeProvider(primary, fallback, ttl_minutes=ttl)

    else:
        raise ValueError(f"未知 signal_source: {source}，支持 external/ma/composite")
