"""Phase-specific strategy logic.

Generates entry/exit signals based on current market phase:
  - 主升 (Uptrend): buy on pullbacks to MA support
  - 震荡 (Range):   buy low / sell high within range
  - 主跌 (Downtrend): no position
"""

import numpy as np
import pandas as pd


class Signal:
    """Trading signal types."""
    NONE = 0
    BUY = 1
    SELL = -1
    CLOSE_LONG = -2  # exit long position (for range trading)


def detect_candle_pattern(df, i):
    """Detect simple bullish reversal candle patterns at index i."""
    if i < 1 or i >= len(df):
        return False
    row = df.iloc[i]
    prev = df.iloc[i - 1]

    # Bullish engulfing
    if prev["close"] < prev["open"] and row["close"] > row["open"]:
        if row["close"] > prev["open"] and row["open"] < prev["close"]:
            return True

    # Hammer / doji after decline
    body = abs(row["close"] - row["open"])
    lower_wick = min(row["open"], row["close"]) - row["low"]
    if body > 0 and lower_wick > 2 * body and row["close"] > row["open"]:
        return True

    # Piercing pattern
    if prev["close"] < prev["open"] and row["close"] > row["open"]:
        mid = (prev["open"] + prev["close"]) / 2
        if row["close"] > mid and row["open"] < prev["close"]:
            return True

    return False


def detect_bearish_candle(df, i):
    """Detect bearish reversal candle patterns at index i."""
    if i < 1 or i >= len(df):
        return False
    row = df.iloc[i]
    prev = df.iloc[i - 1]

    # Bearish engulfing
    if prev["close"] > prev["open"] and row["close"] < row["open"]:
        if row["close"] < prev["open"] and row["open"] > prev["close"]:
            return True

    # Shooting star
    body = abs(row["close"] - row["open"])
    upper_wick = row["high"] - max(row["open"], row["close"])
    if body > 0 and upper_wick > 2 * body and row["close"] < row["open"]:
        return True

    return False


class UptrendStrategy:
    """Strategy for 主升 (Primary Uptrend) phase.

    Entry: price pullback to short-term MA (MA3/MA5/MA10) with bounce confirmation.
    Exit:  phase → 震荡/主跌, or trend stalling (3-day no new high),
           or MA10 breakdown, or consolidation range forming.
    """

    def __init__(self, config):
        cfg = config.get("uptrend", {})
        self.pullback_to_ma3 = cfg.get("pullback_to_ma3", True)
        self.pullback_to_ma5 = cfg.get("pullback_to_ma5", True)
        self.pullback_to_ma10 = cfg.get("pullback_to_ma10", True)
        self.ma_touch_pct = cfg.get("ma_touch_pct", 0.015)
        self.exit_on_phase_change = cfg.get("exit_on_phase_change", True)
        self.exit_no_new_high_days = cfg.get("exit_no_new_high_days", 3)
        self.exit_on_ma_break = cfg.get("exit_on_ma_break", True)
        self.exit_on_consolidation = cfg.get("exit_on_consolidation", True)
        self.hard_stop_pct = cfg.get("hard_stop_pct", -0.08)

    # --- Entry logic ---

    def _ma_pullback_entry(self, df, i, ma_col):
        """Check if lowest price touched the MA and bounced.

        Conditions:
          - Low of current bar ≤ MA (intraday touch/pierce of the MA line)
          - Close > MA (bounce confirmed by end of day)
          - Bullish candle preferred
        """
        if i < 1:
            return False
        row = df.iloc[i]
        ma_val = row.get(ma_col)
        if pd.isna(ma_val):
            return False

        low_touched_ma = row["low"] <= ma_val * (1 + self.ma_touch_pct)
        close_above_ma = row["close"] > ma_val
        bullish = row["close"] > row["open"]

        return low_touched_ma and close_above_ma and bullish

    def evaluate_entry(self, df, i):
        """Check if we should enter in uptrend phase."""
        if i < 3:
            return False

        row = df.iloc[i]
        phase = row.get("phase", 0)

        if phase != 1:
            return False
        if pd.isna(row.get("MA3")):
            return False

        # Try entries at progressively deeper pullbacks (MA3 → MA5 → MA10)
        if self.pullback_to_ma3 and self._ma_pullback_entry(df, i, "MA3"):
            return True
        if self.pullback_to_ma5 and self._ma_pullback_entry(df, i, "MA5"):
            return True
        if self.pullback_to_ma10 and self._ma_pullback_entry(df, i, "MA10"):
            return True

        return False

    # --- Exit logic ---

    def _days_since_high(self, df, entry_idx, i):
        """Count bars since the highest close from entry to current bar."""
        window = df.iloc[entry_idx : i + 1]
        if len(window) == 0:
            return 999
        max_pos = window["close"].values.argmax()
        return len(window) - 1 - max_pos

    def _is_consolidating(self, df, entry_idx, i):
        """Detect if price is forming a narrow consolidation range.

        Conditions: narrow range + MA flattening in last 5 bars.
        """
        recent = df.iloc[max(entry_idx, i - 5) : i + 1]
        if len(recent) < 3:
            return False
        price_range = (recent["high"].max() - recent["low"].min()) / recent["close"].iloc[-1]
        if price_range > 0.025:
            return False
        ma3_slope = recent["MA3_slope"].iloc[-1] if "MA3_slope" in recent.columns else 0
        if abs(ma3_slope) > 0.05:
            return False
        return True

    def evaluate_exit(self, df, i, entry_idx, entry_price, current_stop=None):
        """Check if we should exit the position.

        Exits (any one triggers):
          1. Phase changed from 主升 → 震荡/主跌
          2. No new high in N days (trend stalling after pullback)
          3. Close below MA10 (trend broken)
          4. Consolidation range detected (narrow range + MA flattening)
          5. Hard stop: -8% from entry price (safety backstop)
        """
        row = df.iloc[i]
        phase = row.get("phase", 0)

        # 0. Hard stop from entry (safety net)
        if entry_price > 0:
            loss_pct = (row["close"] - entry_price) / entry_price
            if loss_pct < self.hard_stop_pct:
                return True

        # 1. Phase change away from 主升
        if self.exit_on_phase_change and phase != 1:
            return True

        # 2. No new high in N days (trend stalling)
        if self.exit_no_new_high_days > 0:
            days_since = self._days_since_high(df, entry_idx, i)
            if days_since >= self.exit_no_new_high_days:
                return True

        # 3. Close below MA10 (trend break)
        if self.exit_on_ma_break:
            if not pd.isna(row.get("MA10")) and row["close"] < row["MA10"]:
                # First bar closing below MA10 (not already below previously)
                prev_close = df.iloc[i - 1]["close"] if i > entry_idx else row["close"]
                prev_ma10 = df.iloc[i - 1].get("MA10", float("inf")) if i > entry_idx else float("inf")
                if not pd.isna(prev_ma10) and prev_close >= prev_ma10:
                    return True

        # 4. Consolidation range forming
        if self.exit_on_consolidation:
            if self._is_consolidating(df, entry_idx, i):
                return True

        return False


class RangeStrategy:
    """Strategy for 震荡 (Range-bound / Consolidation) phase.

    Entry pattern (short-term, ~6-10 bar lookback):
      1. 三五天没创新高 — recent bars show stalling (no new high)
      2. 三五天形成低位 — a low formed in the pullback
      3. 开始反弹 — current bar bouncing from that low
    Exit:  price reaches recent high (resistance), or overbought, or stop loss.
    """

    def __init__(self, config):
        cfg = config.get("rangetrading", {})
        self.support_bounce = cfg.get("support_bounce", True)
        self.oversold_bounce = cfg.get("oversold_bounce", True)
        self.rsi_oversold = cfg.get("rsi_oversold", 30)
        self.rsi_neutral = cfg.get("rsi_neutral", 40)
        self.swing_window = cfg.get("swing_window", 5)
        self.exit_on_resistance = cfg.get("exit_on_resistance", True)
        self.exit_on_overbought = cfg.get("exit_on_overbought", True)
        self.rsi_overbought = cfg.get("rsi_overbought", 70)
        self.profit_target_atr = cfg.get("profit_target_atr", 2.0)
        self.stop_loss_atr = cfg.get("stop_loss_atr", 1.5)
        self.max_holding_bars = cfg.get("max_holding_bars", 30)

    def evaluate_entry(self, df, i):
        """Check if we should enter in range phase.

        Looks back ~8 bars to detect:
          - A low has formed (lowest low in recent bars)
          - Price is now bouncing from that low
        """
        lookback = 8
        if i < lookback:
            return False

        row = df.iloc[i]
        atr = row.get("ATR", np.nan)
        phase = row.get("phase", 0)

        if phase != 0:
            return False
        if pd.isna(atr) or atr <= 0:
            return False

        recent = df.iloc[i - lookback : i + 1]

        # Find the recent low (the consolidation/pullback low)
        low_idx = recent["low"].idxmin()
        low_pos = recent.index.get_loc(low_idx)  # 0-based within recent window
        bars_since_low = lookback - low_pos

        # The low should have formed recently (within last 0-2 bars)
        # We want to catch the bounce as it happens
        if bars_since_low > 2:
            return False

        consolidation_low = recent["low"].min()

        # Price should be near that low (bouncing from it)
        from_low_pct = (row["close"] - consolidation_low) / max(consolidation_low, 0.01)
        near_low = from_low_pct < 0.03  # within 3% of the low

        # Bounce confirmation: today is a bounce from the low
        # Either: close > open (bullish) and close > previous close
        bouncing = (
            row["close"] > row["open"]
            and (i < 1 or row["close"] > df.iloc[i - 1]["close"])
        )

        # Also acceptable: hammer/doji at the low (intraday low touched, close recovered)
        candle_body = abs(row["close"] - row["open"])
        lower_wick = min(row["open"], row["close"]) - row["low"]
        hammer = candle_body > 0 and lower_wick > 2 * candle_body

        if not (bouncing or hammer):
            return False

        # Additionally check: was there no big rally before the low?
        # (we want the pattern: stalling → low → bounce, not: already rallied → small dip)
        recent_high = recent["high"].max()
        recent_high_close = recent.loc[recent["high"].idxmax(), "close"]
        if recent_high_close > row["close"] * 1.08:
            # Recent high was 8%+ above current — this might be a trend reversal, not range
            return False

        rsi = row.get("RSI", 50)

        # Condition 1: basic bounce from consolidation low
        if self.support_bounce and near_low and bouncing:
            return True

        # Condition 2: RSI oversold bounce
        if self.oversold_bounce:
            recent_rsi = recent["RSI"]
            was_oversold = (recent_rsi < self.rsi_oversold).any()
            if was_oversold and rsi > self.rsi_neutral and near_low:
                return True
            if rsi < self.rsi_oversold + 5 and hammer and near_low:
                return True

        return False

    def evaluate_exit(self, df, i, entry_idx, entry_price):
        """Check if we should exit the range position.

        Exits on: stop loss, take profit, phase change, time stop,
        overbought RSI, or resistance proximity.
        """
        row = df.iloc[i]
        phase = row.get("phase", 0)
        atr = row.get("ATR", np.nan)

        if phase != 0:
            return True
        if i - entry_idx >= self.max_holding_bars:
            return True
        if pd.isna(atr) or atr <= 0:
            return False

        # Stop loss
        loss_pct = (entry_price - row["close"]) / entry_price
        if loss_pct > self.stop_loss_atr * atr / entry_price:
            return True

        # Take profit
        profit_pct = (row["close"] - entry_price) / entry_price
        target = self.profit_target_atr * atr / entry_price
        if profit_pct > target:
            return True

        # Exit on overbought RSI
        if self.exit_on_overbought and row.get("RSI", 50) > self.rsi_overbought:
            return True

        # Exit near recent swing high (resistance)
        if self.exit_on_resistance:
            recent40 = df.iloc[max(0, i - 40) : i + 1]
            swing_highs = recent40["swing_high"].dropna()
            if len(swing_highs) > 0:
                resistance = swing_highs.iloc[-1]
                if abs(row["close"] - resistance) / row["close"] < 0.01:
                    return True

        return False


class DowntrendStrategy:
    """Strategy for 主跌 (Primary Downtrend) phase.

    No entries — stay in cash. Any existing position should be closed.
    """

    def __init__(self, config):
        _ = config  # unused currently

    def evaluate_entry(self, df, i):
        return False

    def evaluate_exit(self, df, i, entry_idx, entry_price):
        return True  # Always exit in downtrend
