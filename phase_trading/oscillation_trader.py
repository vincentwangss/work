"""
Oscillation 做T (Day Trading) Strategy.

A dedicated strategy for stocks identified as being in a range-bound (震荡)
state.  Generates low-buy / high-sell signals within the oscillation channel,
suitable for repeated 做T (T+0 day trading using existing base position).

Core concept:
  A stock oscillates between a lower bound (support) and upper bound (resistance).
  Buy when price approaches the lower bound and shows reversal signs.
  Sell when price approaches the upper bound and shows exhaustion.
  Multiple round-trips can be executed within one oscillation phase.

Key parameter — oscillation_strength:
  Controls how strictly the oscillation is defined.
    Low  (0.5):  tight channel, more sensitive signals, more trades
    Med  (1.0):  standard Bollinger-like bands
    High (2.0+): wide channel, only extreme moves trigger, fewer but higher quality trades
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


class Signal(IntEnum):
    """Trading signal types."""
    NONE = 0
    BUY = 1       # 买入 (开多/加仓)
    SELL = -1     # 卖出 (平多/减仓)


# ──────────────────────────────────────────────
#  Oscillation Channel Computation
# ──────────────────────────────────────────────

@dataclass
class OscillationConfig:
    """Configuration for oscillation detection and trading.

    Attributes:
        oscillation_strength: Core parameter (0.5–5.0).
            Scales channel width, bounce requirements, and entry/exit thresholds.
        lookback: Rolling window for channel calculation (bars).
        channel_type: "bollinger" | "percentile" | "zscore"
        channel_std: Base std-dev multiplier for Bollinger bands.
        range_percentile_high / _low: Percentile bounds.
        min_bounces: Min number of channel touches to confirm oscillation.
        bounce_window: Rolling window to count bounces within (bars).
    """
    # ── 震荡强参数 (核心) ──
    oscillation_strength: float = 1.0   # 0.5 ~ 5.0

    # ── 通道参数 ──
    lookback: int = 60
    channel_type: str = "bollinger"    # bollinger / percentile / zscore
    channel_std: float = 2.0
    range_percentile_high: float = 90.0
    range_percentile_low: float = 10.0

    # ── 震荡确认参数 ──
    min_bounces: int = 3
    bounce_window: int = 30

    # ── 入场参数 ──
    entry_zone: float = 0.3           # 底部区域阈值 (0~1, 0=下轨, 1=上轨)
    require_oscillation_confirm: bool = True  # 是否要求震荡确认后再入场
    require_volume_contraction: bool = True   # 要求缩量
    volume_std_threshold: float = 1.5         # 成交量 < 均值+1.5倍标准差
    require_reversal_candle: bool = True      # 要求反转K线确认
    max_consecutive_losses: int = 3           # 最大连续止损次数, 超过则冷却

    # ── 出场参数 ──
    exit_zone: float = 0.75           # 顶部区域阈值
    take_profit_atr: float = 2.0      # ATR止盈倍数
    stop_loss_atr: float = 1.5        # ATR止损倍数
    max_holding_bars: int = 12        # 时间止损 (12根30min≈0.5交易日)
    channel_break_exit: bool = True   # 通道突破出场
    channel_break_bars: int = 2       # 连续突破几根K线算有效突破

    # ── 冷却机制 ──
    cooldown_bars: int = 3
    phase_stable_bars: int = 5

    # ── 做T模式 ──
    t_mode: str = "long_only"         # long_only / both

    @property
    def effective_std(self) -> float:
        """Effective channel width scaled by oscillation strength."""
        return self.channel_std * (self.oscillation_strength ** 0.5)

    @property
    def effective_min_bounces(self) -> int:
        """Minimum bounces required, scaled by strength."""
        return max(2, int(np.ceil(self.min_bounces * self.oscillation_strength / 2)))

    @property
    def effective_entry_zscore(self) -> float:
        """Z-score entry threshold: lower threshold for higher strength."""
        return -1.0 * self.oscillation_strength

    @property
    def effective_exit_zscore(self) -> float:
        """Z-score exit threshold: higher threshold for higher strength."""
        return 0.8 * self.oscillation_strength

    @classmethod
    def from_dict(cls, cfg: dict) -> "OscillationConfig":
        """Create config from a dictionary (from config.yaml)."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in cfg.items() if k in valid_keys}
        return cls(**filtered)


class OscillationChannel:
    """Compute oscillation channels and indicators.

    Supports three channel algorithms:
      - bollinger  : Bollinger Bands (mean ± std * k)
      - percentile : Rolling percentile-based channel
      - zscore     : Z-score of price within lookback window
    """

    def __init__(self, config: OscillationConfig):
        self.cfg = config

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute oscillation channel columns on the DataFrame.

        Adds columns:
          osc_upper, osc_lower, osc_mid, osc_range_pct
          osc_zscore, osc_position (0=lower, 1=upper)
          osc_is_bounce_upper, osc_is_bounce_lower
          osc_bounce_count
          osc_is_oscillating (boolean)
          osc_volume_ratio (current vol / mean vol)
        """
        df = df.copy()
        close = df["close"]

        if self.cfg.channel_type == "bollinger":
            self._compute_bollinger(df, close)
        elif self.cfg.channel_type == "percentile":
            self._compute_percentile(df, close)
        elif self.cfg.channel_type == "zscore":
            self._compute_zscore_channel(df, close)

        # ── Z-score (always computed) ──
        roll_mean = close.rolling(self.cfg.lookback).mean()
        roll_std = close.rolling(self.cfg.lookback).std().replace(0, np.nan)
        df["osc_zscore"] = (close - roll_mean) / roll_std

        # ── Position within channel (0 = lower, 1 = upper) ──
        width = (df["osc_upper"] - df["osc_lower"]).replace(0, np.nan)
        df["osc_position"] = (close - df["osc_lower"]) / width
        df["osc_position"] = df["osc_position"].clip(0, 1)

        # ── Range as % of mid-price ──
        mid = (df["osc_upper"] + df["osc_lower"]) / 2
        df["osc_range_pct"] = (df["osc_upper"] - df["osc_lower"]) / mid.replace(0, np.nan) * 100

        # ── Detect bounces ──
        self._detect_bounces(df)

        # ── Volume ratio ──
        if "volume" in df.columns:
            vol_mean = df["volume"].rolling(20).mean().replace(0, np.nan)
            df["osc_volume_ratio"] = df["volume"] / vol_mean
        else:
            df["osc_volume_ratio"] = 1.0

        # ── Oscillation state per bar (vectorized) ──
        # Use rolling conditions to check if each bar was oscillating
        bounces_ok = df["osc_bounce_count"] >= self.cfg.effective_min_bounces
        within_channel = (df["close"] <= df["osc_upper"] * 1.05) & (df["close"] >= df["osc_lower"] * 0.95)
        data_ok = df["osc_upper"].notna() & df["osc_lower"].notna()
        df["osc_is_oscillating"] = bounces_ok & within_channel & data_ok

        return df

    def _compute_bollinger(self, df: pd.DataFrame, close: pd.Series):
        """Bollinger Band channel."""
        mid = close.rolling(self.cfg.lookback).mean()
        std = close.rolling(self.cfg.lookback).std().replace(0, np.nan)
        k = self.cfg.effective_std
        df["osc_upper"] = mid + k * std
        df["osc_lower"] = mid - k * std
        df["osc_mid"] = mid

    def _compute_percentile(self, df: pd.DataFrame, close: pd.Series):
        """Percentile-based channel."""
        hi_q = self.cfg.range_percentile_high / 100
        lo_q = self.cfg.range_percentile_low / 100

        # Apply strength: wider percentile spread for stronger oscillation
        spread = hi_q - lo_q
        strength_factor = self.cfg.oscillation_strength ** 0.5
        adj_spread = min(spread * strength_factor, 0.95)
        adj_hi = 0.5 + adj_spread / 2
        adj_lo = 0.5 - adj_spread / 2

        df["osc_upper"] = close.rolling(self.cfg.lookback).quantile(adj_hi)
        df["osc_lower"] = close.rolling(self.cfg.lookback).quantile(adj_lo)
        df["osc_mid"] = (df["osc_upper"] + df["osc_lower"]) / 2

    def _compute_zscore_channel(self, df: pd.DataFrame, close: pd.Series):
        """Z-score based channel (reconstructs price levels from z-score)."""
        roll_mean = close.rolling(self.cfg.lookback).mean()
        roll_std = close.rolling(self.cfg.lookback).std().replace(0, np.nan)
        k = self.cfg.effective_std
        df["osc_upper"] = roll_mean + k * roll_std
        df["osc_lower"] = roll_mean - k * roll_std
        df["osc_mid"] = roll_mean

    def _detect_bounces(self, df: pd.DataFrame):
        """Detect touches of upper/lower bands that reverse (bounces).

        Uses Z-score to detect when price is at an extreme relative to its
        recent mean, then checks if the next bar reverses direction.

        Two-tier detection:
          1. Primary: price at extreme Z-score + next-bar reversal
          2. Secondary (position-based): price at channel edge + next-bar reversal
             (works better for percentile and fixed-width channels)
        """
        n = len(df)

        touches_upper = np.full(n, False)
        touches_lower = np.full(n, False)

        for i in range(1, n - 1):
            zscore = df["osc_zscore"].iloc[i]
            pos = df["osc_position"].iloc[i]

            if pd.isna(zscore) or pd.isna(pos):
                continue

            # Use Z-score as primary trigger (|z| > 1.2 ≈ near band edge)
            near_upper_z = zscore > 1.2
            near_lower_z = zscore < -1.2

            # Also use position as secondary (for wide-Bollinger cases)
            near_upper_pos = pos > 0.80
            near_lower_pos = pos < 0.20

            near_upper = near_upper_z or near_upper_pos
            near_lower = near_lower_z or near_lower_pos

            # Check next-bar reversal: direction changes toward middle
            if near_upper and df["close"].iloc[i + 1] < df["close"].iloc[i]:
                touches_upper[i] = True
            if near_lower and df["close"].iloc[i + 1] > df["close"].iloc[i]:
                touches_lower[i] = True

        df["osc_bounce_upper"] = touches_upper
        df["osc_bounce_lower"] = touches_lower
        df["osc_bounce_any"] = touches_upper | touches_lower

        # Rolling count of bounces
        bw = self.cfg.bounce_window
        df["osc_bounce_count"] = df["osc_bounce_any"].rolling(bw, min_periods=bw // 2).sum()

    def detect_is_oscillating(self, df: pd.DataFrame, i: int) -> bool:
        """Check if the stock is currently in a valid oscillating state.

        Conditions (all must be met):
          1. Sufficient bounce count in recent window
          2. Price staying within the channel (not breaking out)
          3. Range is reasonable (not too wide, not too narrow)
        """
        row = df.iloc[i]

        # Need enough data
        bounces = row.get("osc_bounce_count")
        if pd.isna(bounces) or bounces < self.cfg.effective_min_bounces:
            return False

        # Price must be within the channel bounds
        close = row["close"]
        upper = row.get("osc_upper")
        lower = row.get("osc_lower")
        if pd.isna(upper) or pd.isna(lower):
            return False

        # Allow slight overshoot (5%), but not full runaway breakout
        overshoot_upper = (close - upper) / upper
        overshoot_lower = (lower - close) / close

        if overshoot_upper > 0.05:
            return False
        if overshoot_lower > 0.05:
            return False

        return True

        return True

    def get_current_state(self, df: pd.DataFrame, i: int) -> dict:
        """Get current oscillation state summary at bar i."""
        row = df.iloc[i]
        return {
            "is_oscillating": self.detect_is_oscillating(df, i),
            "zscore": row.get("osc_zscore", np.nan),
            "position": row.get("osc_position", np.nan),
            "range_pct": row.get("osc_range_pct", np.nan),
            "bounce_count": row.get("osc_bounce_count", 0),
            "at_upper_band": row.get("osc_position", 0) > 0.85,
            "at_lower_band": row.get("osc_position", 0) < 0.15,
            "volume_ratio": row.get("osc_volume_ratio", 1.0),
        }


# ──────────────────────────────────────────────
#  Signal Logic
# ──────────────────────────────────────────────

def detect_bullish_reversal(df: pd.DataFrame, i: int) -> bool:
    """Detect bullish reversal pattern at bar i.

    Checks:
      1. Bullish engulfing
      2. Hammer with long lower wick
      3. Piercing pattern
    """
    if i < 1 or i >= len(df):
        return False
    row = df.iloc[i]
    prev = df.iloc[i - 1]

    # ── Bullish engulfing ──
    if prev["close"] < prev["open"] and row["close"] > row["open"]:
        if row["close"] > prev["open"] and row["open"] < prev["close"]:
            return True

    # ── Hammer ──
    body = abs(row["close"] - row["open"])
    lower_wick = min(row["open"], row["close"]) - row["low"]
    upper_wick = row["high"] - max(row["open"], row["close"])
    if body > 0 and lower_wick > 2 * body and upper_wick < body * 0.5:
        if row["close"] > row["open"]:  # green hammer
            return True

    # ── Piercing line ──
    if prev["close"] < prev["open"] and row["close"] > row["open"]:
        mid_prev = (prev["open"] + prev["close"]) / 2
        if row["close"] > mid_prev and row["open"] < prev["close"]:
            return True

    return False


def detect_bearish_exhaustion(df: pd.DataFrame, i: int) -> bool:
    """Detect bearish exhaustion / topping pattern at bar i.

    Checks:
      1. Bearish engulfing
      2. Shooting star
      3. Doji after uptrend
    """
    if i < 1 or i >= len(df):
        return False
    row = df.iloc[i]
    prev = df.iloc[i - 1]

    # ── Bearish engulfing ──
    if prev["close"] > prev["open"] and row["close"] < row["open"]:
        if row["close"] < prev["open"] and row["open"] > prev["close"]:
            return True

    # ── Shooting star ──
    body = abs(row["close"] - row["open"])
    upper_wick = row["high"] - max(row["open"], row["close"])
    if body > 0 and upper_wick > 2 * body and row["close"] < row["open"]:
        return True

    # ── Doji at top ──
    if body / (row["high"] - row["low"] + 1e-10) < 0.1:
        if prev["close"] > prev["open"] and row["high"] > prev["high"]:
            return True

    return False


# ──────────────────────────────────────────────
#  Oscillation 做T Strategy
# ──────────────────────────────────────────────

class OscillationTrader:
    """震荡做T策略主逻辑

    Entry rules:
      1. Phase = 震荡 (or self-detected oscillation)
      2. Price in lower zone (osc_position < entry_zone)
      3. Volume contraction (optional)
      4. Reversal candle confirmation (optional)
      5. Not in cooldown

    Exit rules:
      1. Price reaches upper zone (osc_position > exit_zone)
      2. Take profit / stop loss hit
      3. Channel breakout (price exits channel)
      4. Time stop (max_holding_bars exceeded)
      5. Phase change away from oscillation
    """

    def __init__(self, config: OscillationConfig):
        self.cfg = config
        self.channel = OscillationChannel(config)

    # ── Entry ──

    def evaluate_entry(self, df: pd.DataFrame, i: int, phase: int) -> Signal:
        """Evaluate if we should enter a long position.

        Args:
            df: DataFrame with oscillation indicators computed.
            i: Current bar index.
            phase: Current market phase (0 = oscillation, -1/1 = trend).

        Returns:
            Signal.BUY or Signal.NONE
        """
        if i < self.cfg.lookback:
            return Signal.NONE

        row = df.iloc[i]

        # ── Phase check ──
        if not self._is_valid_phase(df, i, phase):
            return Signal.NONE

        # Oscillation confirmation (can be relaxed for more trades)
        if self.cfg.require_oscillation_confirm:
            if not self.channel.detect_is_oscillating(df, i):
                return Signal.NONE

        # ── Position check: price in lower zone ──
        pos = row.get("osc_position")
        if pd.isna(pos) or pos > self.cfg.entry_zone:
            return Signal.NONE

        # ── Volume contraction check ──
        if self.cfg.require_volume_contraction:
            vol_ratio = row.get("osc_volume_ratio", 1.0)
            if vol_ratio > self.cfg.volume_std_threshold:
                return Signal.NONE

        # ── Reversal candle check ──
        if self.cfg.require_reversal_candle:
            if not detect_bullish_reversal(df, i):
                # Allow: if position is extremely low (< 0.1), accept any green candle
                if not (pos < 0.1 and row["close"] > row["open"]):
                    return Signal.NONE

        # ── Z-score confirmation ──
        zscore = row.get("osc_zscore", 0)
        if pd.notna(zscore) and zscore > self.cfg.effective_entry_zscore:
            # Not extreme enough — skip unless position is at absolute bottom
            if pos > 0.05:
                return Signal.NONE

        return Signal.BUY

    # ── Exit ──

    def evaluate_exit(
        self,
        df: pd.DataFrame,
        i: int,
        entry_idx: int,
        entry_price: float,
        phase: int,
    ) -> bool:
        """Evaluate if we should exit the current position.

        Args:
            df: DataFrame with oscillation indicators.
            i: Current bar index.
            entry_idx: Entry bar index.
            entry_price: Entry fill price.
            phase: Current market phase.

        Returns:
            True if should exit.
        """
        row = df.iloc[i]
        holding_bars = i - entry_idx
        price = row["close"]

        # ── 1. Phase change → exit ──
        if not self._is_valid_phase(df, i, phase):
            return True

        # ── 2. Time stop ──
        if holding_bars >= self.cfg.max_holding_bars:
            return True

        # ── 3. Stop loss ──
        if self.cfg.stop_loss_atr > 0:
            atr = row.get("ATR", np.nan)
            if pd.notna(atr) and atr > 0:
                loss_pct = (entry_price - price) / entry_price
                if loss_pct > self.cfg.stop_loss_atr * atr / entry_price:
                    return True

        # ── 4. Take profit ──
        if self.cfg.take_profit_atr > 0:
            atr = row.get("ATR", np.nan)
            if pd.notna(atr) and atr > 0:
                profit_pct = (price - entry_price) / entry_price
                if profit_pct > self.cfg.take_profit_atr * atr / entry_price:
                    return True

        # ── 5. Reached upper zone ──
        pos = row.get("osc_position", 0)
        if pd.notna(pos) and pos > self.cfg.exit_zone:
            # Confirm with bearish candle or overbought
            zscore = row.get("osc_zscore", 0)
            overbought = pd.notna(zscore) and zscore > self.cfg.effective_exit_zscore
            bearish = detect_bearish_exhaustion(df, i)
            if overbought or bearish:
                return True

        # ── 6. Channel breakout exit ──
        if self.cfg.channel_break_exit:
            if self._is_channel_breakout(df, i):
                return True

        return False

    # ── Internal helpers ──

    def _is_valid_phase(self, df: pd.DataFrame, i: int, phase: int) -> bool:
        """Check if trading is valid in current phase.

        If oscillation confirm is OFF, always allow entry —
        position + z-score + SL/TP provide the risk control.
        If ON:
          - External phase 0 (震荡) is OK.
          - Non-zero phase is OK only if self-detected oscillation.
        """
        if not self.cfg.require_oscillation_confirm:
            return True
        if phase == 0:
            return True
        # Allow in mild uptrend/downtrend if self-detect says oscillating
        if self.channel.detect_is_oscillating(df, i):
            return True
        return False

    def _is_channel_breakout(self, df: pd.DataFrame, i: int) -> bool:
        """Detect if price has broken out of the oscillation channel.

        Considers consecutive closes outside the channel.
        """
        if i < self.cfg.channel_break_bars:
            return False

        upper = df["osc_upper"].iloc[i]
        lower = df["osc_lower"].iloc[i]
        if pd.isna(upper) or pd.isna(lower):
            return False

        # Check consecutive closes outside
        above_count = 0
        below_count = 0
        for j in range(i - self.cfg.channel_break_bars + 1, i + 1):
            c = df["close"].iloc[j]
            if c > upper:
                above_count += 1
            elif c < lower:
                below_count += 1

        needed = self.cfg.channel_break_bars
        return above_count >= needed or below_count >= needed


# ──────────────────────────────────────────────
#  Standalone Oscillation Scanner / Backtest
# ──────────────────────────────────────────────

def compute_oscillation_metrics(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """Convenience function: compute all oscillation indicators on a DataFrame.

    Args:
        df: OHLCV DataFrame (columns: open, high, low, close, volume).
        config: Optional config dict for OscillationConfig.from_dict().

    Returns:
        DataFrame with all oscillation columns added.
    """
    if config:
        cfg = OscillationConfig.from_dict(config.get("oscillation_trading", {}))
    else:
        cfg = OscillationConfig()

    channel = OscillationChannel(cfg)
    return channel.compute(df)


def find_oscillation_stocks(
    df_dict: dict[str, pd.DataFrame],
    config: Optional[dict] = None,
) -> list[dict]:
    """Scan multiple stocks and find those currently oscillating.

    Args:
        df_dict: Dict mapping symbol → DataFrame.
        config: Optional config dict.

    Returns:
        List of dicts with symbol and oscillation metrics, sorted by quality.
    """
    from dataclasses import asdict

    if config:
        cfg = OscillationConfig.from_dict(config.get("oscillation_trading", {}))
    else:
        cfg = OscillationConfig()

    channel = OscillationChannel(cfg)
    results = []

    for symbol, df in df_dict.items():
        try:
            df_out = channel.compute(df)
            state = channel.get_current_state(df_out, len(df_out) - 1)
            i = len(df_out) - 1

            if not state["is_oscillating"]:
                continue

            # Quality score: higher = better oscillation candidate
            score = (
                min(state["bounce_count"] / cfg.effective_min_bounces, 3.0) * 0.3
                + (1.0 - abs(state["position"] - 0.5) * 2) * 0.3   # near mid = stable
                + min(15.0 / state["range_pct"], 1.5) * 0.2        # moderate range
                + (1.0 - min(abs(state["zscore"]), 3.0) / 3.0) * 0.2   # not extreme
            )

            # Most recent touch direction
            recent_dir = "neutral"
            if df_out["osc_bounce_upper"].iloc[-5:].any():
                recent_dir = "near_upper"
            elif df_out["osc_bounce_lower"].iloc[-5:].any():
                recent_dir = "near_lower"

            results.append({
                "symbol": symbol,
                "score": round(score, 3),
                "range_pct": round(state["range_pct"], 2),
                "zscore": round(state["zscore"], 2),
                "position": round(state["position"], 3),
                "bounce_count": state["bounce_count"],
                "recent_bias": recent_dir,
                "volume_ratio": round(state["volume_ratio"], 2),
            })
        except Exception as e:
            results.append({"symbol": symbol, "error": str(e)})

    results.sort(key=lambda r: r.get("score", 0), reverse=True)
    return results


# ──────────────────────────────────────────────
#  Rolling Parameter Optimizer
# ──────────────────────────────────────────────

@dataclass
class StrengthCandidate:
    """Result for one candidate strength value."""
    strength: float
    total_return_pct: float
    annual_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    win_rate_pct: float
    profit_factor: float


class StrengthOptimizer:
    """Rolling parameter optimizer for oscillation_strength.

    Scans multiple strength values over a recent historical window,
    picks the one that maximized the target metric,
    then returns the best value for forward trading.

    Uses an inline lightweight backtest (no trade log) for speed,
    avoiding circular imports with backtest.py.
    """

    def __init__(self, base_config: Optional[OscillationConfig] = None):
        self.base_config = base_config or OscillationConfig()

    def optimize(
        self,
        df: pd.DataFrame,
        strength_candidates: Optional[list[float]] = None,
        validation_bars: int = 120,
        metric: str = "sharpe_ratio",
        min_trades: int = 2,
        initial_capital: float = 100000.0,
    ) -> tuple[float, list[StrengthCandidate]]:
        """Run optimization and return (best_strength, candidate_results).

        The last `validation_bars` of df are used as the validation window.
        Each candidate is scored via a lightweight backtest.

        Args:
            df: Full OHLCV DataFrame (use the most recent portion for validation).
            strength_candidates: List of strength values to try.
                Default: [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0].
            validation_bars: Number of recent bars to validate on.
            metric: Which metric to maximize ('sharpe_ratio', 'total_return_pct',
                    'profit_factor', or 'composite').
            min_trades: Minimum trades required for a candidate to be valid.
            initial_capital: Starting capital for the inline backtest.

        Returns:
            (best_strength, candidate_results)
        """
        candidates = strength_candidates or [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]

        if len(df) < validation_bars + self.base_config.lookback + 30:
            # Not enough data — return default
            return self.base_config.oscillation_strength, []

        # Slice the validation window from the END of df
        val_start = len(df) - validation_bars
        df_val = df.iloc[max(0, val_start - self.base_config.lookback):].copy()

        results: list[StrengthCandidate] = []

        for strength in candidates:
            cfg = copy_config_with_strength(self.base_config, strength)
            stats = self._quick_backtest(df_val, cfg, initial_capital)
            if stats is None:
                continue
            if stats["total_trades"] < min_trades:
                continue
            results.append(StrengthCandidate(strength=strength, **stats))

        if not results:
            return self.base_config.oscillation_strength, []

        # Score and pick best
        scored = self._score_candidates(results, metric)
        best = scored[0]

        return best.strength, results

    def _quick_backtest(
        self,
        df: pd.DataFrame,
        cfg: OscillationConfig,
        capital: float,
    ) -> Optional[dict]:
        """Fast inline backtest that only computes aggregate metrics.

        Returns None if the backtest fails or produces no trades.
        """
        try:
            channel = OscillationChannel(cfg)
            trader = OscillationTrader(cfg)
            df_out = channel.compute(df)
        except Exception:
            return None

        n = len(df_out)
        if n < cfg.lookback + 10:
            return None

        cash = float(capital)
        position = 0
        entry_price = 0.0
        entry_idx = 0
        equities = []
        n_trades = 0
        n_wins = 0
        total_profit = 0.0
        total_loss = 0.0
        max_equity = float(capital)
        peak = float(capital)
        max_drawdown = 0.0

        for i in range(cfg.lookback, n):
            row = df_out.iloc[i]
            phase = row.get("phase", 0)

            if position == 0:
                sig = trader.evaluate_entry(df_out, i, phase)
                if sig == Signal.BUY:
                    price = row["close"] * 1.001  # 0.1% slippage
                    allocated = cash * 0.95
                    shares = int(allocated / (price * 100)) * 100
                    if shares < 100:
                        continue
                    cost = shares * price * 1.0003  # 0.03% commission
                    if cost > cash:
                        continue
                    cash -= cost
                    position = shares
                    entry_price = price
                    entry_idx = i
            else:
                should_exit = trader.evaluate_exit(df_out, i, entry_idx, entry_price, phase)
                if should_exit:
                    price = row["close"] * 0.999  # 0.1% slippage
                    proceeds = position * price * 0.9997  # 0.03% commission
                    gross_pnl = proceeds - (position * entry_price)
                    cash += proceeds

                    n_trades += 1
                    if gross_pnl > 0:
                        n_wins += 1
                        total_profit += gross_pnl
                    else:
                        total_loss += abs(gross_pnl)

                    position = 0

            # Record equity
            equity = cash + position * row["close"]
            equities.append(equity)
            peak = max(peak, equity)
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            max_drawdown = max(max_drawdown, dd)

        # Final equity (close any open position)
        if position > 0:
            price = df_out.iloc[-1]["close"]
            proceeds = position * price * 0.9997
            cash += proceeds
            equities[-1] = cash

        final_equity = cash
        total_return = (final_equity / capital - 1) * 100

        if n_trades == 0:
            return None

        # Daily returns from equity curve for Sharpe
        eq_series = pd.Series(equities, index=df_out.index[cfg.lookback:])
        daily_ret = eq_series.pct_change().dropna()

        if len(daily_ret) > 1 and daily_ret.std() > 0:
            excess = daily_ret - 0.03 / 252
            sharpe = np.sqrt(252) * excess.mean() / daily_ret.std()
        else:
            sharpe = 0.0

        n_days = len(eq_series)
        years = max(n_days / 252, 0.01)
        ann_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100

        win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else (
            float("inf") if total_profit > 0 else 0
        )

        return {
            "total_return_pct": round(total_return, 2),
            "annual_return_pct": round(ann_return, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "total_trades": n_trades,
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
        }

    def _score_candidates(
        self,
        candidates: list[StrengthCandidate],
        metric: str,
    ) -> list[StrengthCandidate]:
        """Sort candidates by the target metric (descending)."""
        if metric == "composite":
            # Composite score: normalize and combine multiple metrics
            def _composite(c: StrengthCandidate) -> float:
                sharpe_score = max(0, min(c.sharpe_ratio / 3, 1))
                return_score = max(0, min(c.total_return_pct / 100, 1))
                dd_penalty = max(0, c.max_drawdown_pct / 50)
                trade_score = min(c.total_trades / 10, 1) if c.total_trades >= 3 else -1
                return (sharpe_score * 0.4 + return_score * 0.3 - dd_penalty * 0.2
                        + trade_score * 0.1)

            candidates.sort(key=_composite, reverse=True)
        elif metric == "profit_factor":
            candidates.sort(
                key=lambda c: c.profit_factor if c.profit_factor != float("inf") else 999,
                reverse=True,
            )
        else:
            candidates.sort(
                key=lambda c: getattr(c, metric, 0),
                reverse=True,
            )
        return candidates

    def summary(self, results: list[StrengthCandidate], best_strength: float) -> str:
        """Format optimization results as a printable table."""
        if not results:
            return "  (no valid candidates)"

        lines = []
        labels = {
            "strength": "Strength",
            "total_return_pct": "Return%",
            "sharpe_ratio": "Sharpe",
            "max_drawdown_pct": "MaxDD%",
            "total_trades": "Trades",
            "win_rate_pct": "Win%",
            "profit_factor": "PFactor",
        }

        header = f"  {'*':>3s}  " + "  ".join(f"{v:>8s}" for v in labels.values())
        sep = "  " + "-" * (5 + len(labels) * 10)
        lines.append(header)
        lines.append(sep)

        for c in sorted(results, key=lambda x: x.strength):
            marker = ">" if abs(c.strength - best_strength) < 0.001 else " "
            pf = c.profit_factor
            pf_str = f"{pf:.2f}" if pf != float("inf") else "  inf"
            lines.append(
                f"  {marker:>3s}  "
                f"{c.strength:>8.1f}  "
                f"{c.total_return_pct:>7.2f}%  "
                f"{c.sharpe_ratio:>8.2f}  "
                f"{c.max_drawdown_pct:>7.2f}%  "
                f"{c.total_trades:>7d}  "
                f"{c.win_rate_pct:>7.1f}%  "
                f"{pf_str:>8s}"
            )
        lines.append(sep)

        return "\n".join(lines)


def copy_config_with_strength(base: OscillationConfig, strength: float) -> OscillationConfig:
    """Create a copy of OscillationConfig with a different oscillation_strength."""
    from dataclasses import replace
    return replace(base, oscillation_strength=round(strength, 4))
