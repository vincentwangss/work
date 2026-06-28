"""Market Phase Detector.

Identifies three market phases:
  -  1 : 主升 (Primary Uptrend)
  -  0 : 震荡 (Range-bound / Consolidation)
  - -1 : 主跌 (Primary Downtrend)

Uses ADX for trend strength and MA alignment for direction.
"""

import numpy as np
import pandas as pd


class PhaseDetector:
    """Detect market phase using technical indicators."""

    def __init__(self, config=None):
        cfg = config or {}
        pd_cfg = cfg.get("phase_detector", {})
        self.adx_period = pd_cfg.get("adx_period", 14)
        self.adx_threshold = pd_cfg.get("adx_threshold", 25)
        self.ma_fast = pd_cfg.get("ma_fast", 20)
        self.ma_slow = pd_cfg.get("ma_slow", 60)
        self.ma_slope_period = pd_cfg.get("ma_slope_period", 5)

    def compute_indicators(self, df):
        """Compute all indicators needed for phase detection."""
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # --- Moving Averages ---
        df["MA_fast"] = close.rolling(window=self.ma_fast).mean()
        df["MA_slow"] = close.rolling(window=self.ma_slow).mean()

        # Short-term MAs for strategy entry timing
        df["MA3"] = close.rolling(window=3).mean()
        df["MA5"] = close.rolling(window=5).mean()
        df["MA10"] = close.rolling(window=10).mean()
        df["MA3_slope"] = df["MA3"].pct_change(1) * 100
        df["MA5_slope"] = df["MA5"].pct_change(1) * 100
        df["MA10_slope"] = df["MA10"].pct_change(1) * 100
        # MA10 vs MA3 relationship (for trend strength)
        df["MA10_above_MA3"] = df["MA10"] > df["MA3"]

        # MA slopes (using linear regression slope over ma_slope_period)
        def _slope(series):
            y = series.dropna().values
            if len(y) < 2:
                return np.nan
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0] / y.mean() * 100  # normalized slope %

        df["MA_fast_slope"] = (
            df["MA_fast"].rolling(window=self.ma_slope_period).apply(_slope, raw=False)
        )
        df["MA_slow_slope"] = (
            df["MA_slow"].rolling(window=self.ma_slope_period).apply(_slope, raw=False)
        )

        # MA cross state
        df["MA_fast_above_slow"] = df["MA_fast"] > df["MA_slow"]

        # Price relative to MAs
        df["price_above_MA_fast"] = close > df["MA_fast"]
        df["price_above_MA_slow"] = close > df["MA_slow"]

        # --- ADX ---
        tr = np.maximum(
            high - low,
            np.maximum(
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ),
        )
        atr = tr.rolling(window=self.adx_period).mean()

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_di = 100 * (
            pd.Series(plus_dm, index=df.index).rolling(self.adx_period).sum()
            / tr.rolling(self.adx_period).sum()
        )
        minus_di = 100 * (
            pd.Series(minus_dm, index=df.index).rolling(self.adx_period).sum()
            / tr.rolling(self.adx_period).sum()
        )

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df["ADX"] = dx.rolling(window=self.adx_period).mean()
        df["ATR"] = atr
        df["+DI"] = plus_di
        df["-DI"] = minus_di

        # --- MACD ---
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        df["MACD"] = exp12 - exp26
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

        # --- RSI ---
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        # --- Bollinger Bands ---
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df["BB_upper"] = bb_mid + 2 * bb_std
        df["BB_lower"] = bb_mid - 2 * bb_std
        df["BB_position"] = (close - bb_mid) / (bb_std * 2)

        return df

    def detect(self, df):
        """Detect market phase for each bar.

        Uses a two-tier approach:
          1. Strong trend: ADX > threshold + MA alignment
          2. MA trend: clear MA alignment even with moderate ADX
          Otherwise: range/consolidation

        Returns:
            DataFrame with added 'phase' column:
                1  = 主升 (Primary Uptrend)
                0  = 震荡 (Range-bound / Consolidation)
                -1 = 主跌 (Primary Downtrend)
        """
        df = self.compute_indicators(df).copy()

        trending = df["ADX"] > self.adx_threshold

        # --- MA alignment conditions (used in both tiers) ---
        # Uptrend: price above fast MA, fast MA above slow MA, fast MA sloping up
        ma_uptrend = (
            df["price_above_MA_fast"]
            & df["MA_fast_above_slow"]
            & (df["MA_fast_slope"] > -0.05)
        )
        # Downtrend: price below fast MA, fast MA below slow MA, fast MA sloping down
        ma_downtrend = (
            ~df["price_above_MA_fast"]
            & ~df["MA_fast_above_slow"]
            & (df["MA_fast_slope"] < 0.05)
        )

        # Tier 1: Strong trend (high ADX + MA alignment)
        strong_uptrend = trending & ma_uptrend
        strong_downtrend = trending & ma_downtrend

        # Tier 2: MA-confirmed trend (clear MA alignment, regardless of ADX)
        # Price must be well above/below both MAs for conviction
        above_both_ma = df["price_above_MA_fast"] & df["price_above_MA_slow"]
        below_both_ma = ~df["price_above_MA_fast"] & ~df["price_above_MA_slow"]

        ma_uptrend_confirmed = (
            ma_uptrend
            & above_both_ma
            & (df["MA_fast"].notna())
            & (df["MA_slow"].notna())
        )
        ma_downtrend_confirmed = (
            ma_downtrend
            & below_both_ma
            & (df["MA_fast"].notna())
            & (df["MA_slow"].notna())
        )

        # Combine: strong takes priority, then MA-confirmed
        is_uptrend = strong_uptrend | ma_uptrend_confirmed
        is_downtrend = strong_downtrend | ma_downtrend_confirmed

        df["phase"] = np.select(
            [is_uptrend, is_downtrend],
            [1, -1],
            default=0,
        )

        # --- Consolidation override ---
        # When price is oscillating in a tight range AND ADX is declining,
        # force 震荡 even if MA alignment says 主升.
        # This catches extended sideways periods after a strong run-up
        # where ADX stays >25 but the trend has clearly gone flat
        # (e.g. 利通电子 2.13→4.8, ADX 81→27 but never <25).
        #
        # Uses close-price range (not high-low) to avoid inflation from
        # single-bar intraday spikes.  A stock can have 6% daily range +
        # +19% spike bars yet still show a tight close-to-close oscillation.
        # Combined with Bollinger Band position (not riding band edge = not
        # in strong trend) and ADX decline (momentum fading).
        lookback = 10
        recent_close_high = df["close"].rolling(lookback).max()
        recent_close_low = df["close"].rolling(lookback).min()
        close_range_pct = (
            (recent_close_high - recent_close_low) / df["close"].replace(0, np.nan)
        )

        # Price near middle of Bollinger Band (not hugging upper/lower edge)
        near_mid_band = df["BB_position"].abs() < 0.85

        # Momentum/volatility fading — either ADX declining (period 1 type)
        # or BB width contracting (period 3 type where ADX stays elevated
        # but volatility narrows as price stalls near highs).
        adx_declining = df["ADX"] < df["ADX"].shift(5)

        bb_width = (
            (df["BB_upper"] - df["BB_lower"]) / df["close"].rolling(20).mean()
        )
        bb_width_contracting = bb_width < bb_width.shift(5)

        momentum_fading = adx_declining | bb_width_contracting

        consolidate = (
            (close_range_pct < 0.28)                  # close-to-close oscillation
            & (near_mid_band)                         # not riding a band edge
            & (momentum_fading)                       # momentum or vol fading
            & (df["ADX"] > 0)
            & (df["close"].notna())
        )
        df.loc[consolidate, "phase"] = 0

        return df

    def get_current_phase(self, df):
        """Get the latest phase value."""
        result = self.detect(df)
        return int(result["phase"].iloc[-1])

    def get_phase_name(self, phase_val):
        return {1: "主升", -1: "主跌", 0: "震荡"}.get(phase_val, "未知")


class SwingDetector:
    """Detect swing highs and lows for support/resistance levels."""

    def __init__(self, window=5):
        self.window = window

    def detect(self, df):
        """Add swing_high and swing_low columns."""
        df = df.copy()
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)
        w = self.window

        swing_highs = np.full(n, np.nan)
        swing_lows = np.full(n, np.nan)

        for i in range(w, n - w):
            if all(highs[i] >= highs[i - w : i]) and all(
                highs[i] >= highs[i + 1 : i + w + 1]
            ):
                swing_highs[i] = highs[i]
            if all(lows[i] <= lows[i - w : i]) and all(
                lows[i] <= lows[i + 1 : i + w + 1]
            ):
                swing_lows[i] = lows[i]

        df["swing_high"] = swing_highs
        df["swing_low"] = swing_lows

        # Forward-fill last known levels
        df["last_swing_high"] = df["swing_high"].ffill()
        df["last_swing_low"] = df["swing_low"].ffill()
        return df
