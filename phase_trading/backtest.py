"""Backtesting engine for the phase-based trading system.

Simulates trading on historical data and produces performance metrics.
"""

import os
import json
import numpy as np
import pandas as pd

from phase_detector import PhaseDetector, SwingDetector
from strategies import UptrendStrategy, RangeStrategy, DowntrendStrategy, Signal
from oscillation_trader import (
    OscillationConfig,
    OscillationChannel,
    OscillationTrader,
    Signal as OscSignal,
)


class BacktestEngine:
    """Event-driven backtesting engine."""

    def __init__(self, config):
        self.config = config
        bt_cfg = config.get("backtest", {})

        self.initial_capital = bt_cfg.get("initial_capital", 100000.0)
        self.commission_pct = bt_cfg.get("commission_pct", 0.0003)
        self.slippage_pct = bt_cfg.get("slippage_pct", 0.001)
        self.min_volume = bt_cfg.get("min_volume", 0)
        self.phase_stable_bars = bt_cfg.get("phase_stable_bars", 5)
        self.cooldown_bars = bt_cfg.get("cooldown_bars", 5)

        # Strategy instances
        self.phase_detector = PhaseDetector(config)
        self.swing_detector = SwingDetector(
            config.get("rangetrading", {}).get("swing_window", 5)
        )
        self.uptrend = UptrendStrategy(config)
        self.range_stg = RangeStrategy(config)
        self.downtrend = DowntrendStrategy(config)

        # State
        self.reset()

    def reset(self):
        """Reset backtest state."""
        self.cash = self.initial_capital
        self.position = 0  # current shares held
        self.entry_price = 0.0
        self.entry_idx = 0
        self.entry_bar = 0  # bar index of entry
        self.current_stop = None
        self.trades = []
        self.equity_curve = []
        self.phase_history = []
        self.current_phase = 0
        self.consecutive_phase = 0  # bars since last phase change
        self.exit_bar = -999  # bar index of last exit (for cooldown)

    def run(self, df, symbol="unknown"):
        """Run backtest on historical data.

        Args:
            df: DataFrame with OHLCV data
            symbol: symbol name for reporting

        Returns:
            dict with backtest results
        """
        self.reset()
        df = df.copy()

        # Compute all indicators
        df = self.phase_detector.detect(df)
        df = self.swing_detector.detect(df)

        n = len(df)
        max_pos_pct = self.config.get("risk", {}).get("max_position_pct", 0.95)

        for i in range(60, n):  # Start after warmup
            row = df.iloc[i]
            phase = row["phase"]
            self.current_phase = phase
            self.phase_history.append(phase)

            # Track phase stability (consecutive same-phase count)
            if i > 60:
                self.consecutive_phase = (
                    self.consecutive_phase + 1
                    if phase == self.phase_history[-2]
                    else 1
                )

            atr = row.get("ATR", np.nan)

            if self.position == 0:
                # --- Not in a position: check entry ---
                signal = self._check_entry(df, i, phase)
                if signal == Signal.BUY:
                    price = row["close"] * (1 + self.slippage_pct)
                    # Position sizing: use % of capital, round to 100 shares
                    allocated = self.cash * max_pos_pct
                    shares = int(allocated / (price * 100)) * 100
                    if shares < 100:
                        continue
                    cost = shares * price
                    commission = cost * self.commission_pct
                    if cost + commission > self.cash:
                        shares = int((self.cash - commission) / (price * 100)) * 100
                        if shares < 100:
                            continue
                        cost = shares * price
                        commission = cost * self.commission_pct

                    self.cash -= cost + commission
                    self.position = shares
                    self.entry_price = price
                    self.entry_idx = i
                    self.entry_bar = len(self.equity_curve)

                    self.trades.append({
                        "entry_date": df.index[i],
                        "entry_price": round(price, 3),
                        "shares": shares,
                        "entry_phase": int(phase),
                        "phase_name": self.phase_detector.get_phase_name(phase),
                    })

            else:
                # --- In a position: check exit ---
                should_exit = self._check_exit(df, i, phase)

                if should_exit:
                    price = row["close"] * (1 - self.slippage_pct)
                    proceeds = self.position * price
                    commission = proceeds * self.commission_pct
                    self.cash += proceeds - commission

                    trade = self.trades[-1]
                    trade["exit_date"] = df.index[i]
                    trade["exit_price"] = round(price, 3)
                    trade["exit_phase"] = int(phase)
                    trade["exit_phase_name"] = self.phase_detector.get_phase_name(phase)
                    gross_pnl = proceeds - commission - (
                        trade["shares"] * trade["entry_price"]
                    )
                    trade_cost = trade["shares"] * trade["entry_price"] * self.commission_pct
                    trade["gross_pnl"] = round(gross_pnl - trade_cost, 2)
                    trade["return_pct"] = round(
                        gross_pnl / (trade["shares"] * trade["entry_price"]) * 100, 2
                    )
                    trade["holding_bars"] = i - self.entry_idx

                    self.position = 0
                    self.entry_price = 0.0
                    self.current_stop = None
                    self.exit_bar = i  # track exit for cooldown

            # Record equity
            equity = self.cash + self.position * row["close"]
            self.equity_curve.append({
                "date": df.index[i],
                "equity": round(equity, 2),
                "cash": round(self.cash, 2),
                "position": self.position,
                "close": round(row["close"], 3),
                "phase": int(phase),
                "phase_name": self.phase_detector.get_phase_name(phase),
            })

        # Close any remaining position at end
        if self.position > 0:
            final_price = df.iloc[-1]["close"]
            proceeds = self.position * final_price
            commission = proceeds * self.commission_pct
            self.cash += proceeds - commission
            trade = self.trades[-1]
            trade["exit_date"] = df.index[-1]
            trade["exit_price"] = round(final_price, 3)
            trade["exit_phase"] = int(df.iloc[-1]["phase"])
            trade["exit_phase_name"] = self.phase_detector.get_phase_name(df.iloc[-1]["phase"])
            gross_pnl = proceeds - commission - (trade["shares"] * trade["entry_price"])
            trade_cost = trade["shares"] * trade["entry_price"] * self.commission_pct
            trade["gross_pnl"] = round(gross_pnl - trade_cost, 2)
            trade["return_pct"] = round(
                gross_pnl / (trade["shares"] * trade["entry_price"]) * 100, 2
            )
            trade["holding_bars"] = n - 1 - self.entry_idx
            self.position = 0

        return self._compute_results(symbol, df)

    def _check_entry(self, df, i, phase):
        """Check entry signals based on current phase.

        Guards:
          - Phase stability: require N consecutive same-phase bars
          - Cooldown:       wait N bars after last exit (avoid whipsaw)
        """
        if self.consecutive_phase < self.phase_stable_bars:
            return Signal.NONE
        if i - self.exit_bar < self.cooldown_bars:
            return Signal.NONE

        if phase == 1:
            if self.uptrend.evaluate_entry(df, i):
                return Signal.BUY
        elif phase == 0:
            if self.range_stg.evaluate_entry(df, i):
                return Signal.BUY
        # phase == -1: no entry in downtrend
        return Signal.NONE

    def _check_exit(self, df, i, phase):
        """Check exit signals based on entry phase strategy."""
        # First check: downtrend should close any position
        if phase == -1:
            return True

        entry_phase = int(self.trades[-1]["entry_phase"])

        if entry_phase == 1:
            return self.uptrend.evaluate_exit(
                df, i, self.entry_idx, self.entry_price, self.current_stop
            )
        elif entry_phase == 0:
            return self.range_stg.evaluate_exit(df, i, self.entry_idx, self.entry_price)
        return True

    def _compute_results(self, symbol, df):
        """Compute performance metrics from backtest results."""
        eq_df = pd.DataFrame(self.equity_curve)
        if eq_df.empty:
            return {"symbol": symbol, "error": "no data in equity curve"}

        eq_df = eq_df.set_index("date")
        eq_df["daily_return"] = eq_df["equity"].pct_change()
        eq_df["cumulative_return"] = eq_df["equity"] / self.initial_capital - 1

        # Metrics
        total_return = (self.cash / self.initial_capital - 1) * 100
        n_days = len(eq_df)
        years = max(n_days / 252, 0.01)
        ann_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100

        # Sharpe ratio (using 3% risk-free rate)
        daily_returns = eq_df["daily_return"].dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            excess_returns = daily_returns - 0.03 / 252
            sharpe = np.sqrt(252) * excess_returns.mean() / daily_returns.std()
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        # Trade stats
        n_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.get("gross_pnl", 0) > 0]
        losing_trades = [t for t in self.trades if t.get("gross_pnl", 0) <= 0]
        win_rate = len(winning_trades) / n_trades * 100 if n_trades > 0 else 0.0

        total_profit = sum(t.get("gross_pnl", 0) for t in winning_trades)
        total_loss = abs(sum(t.get("gross_pnl", 0) for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        avg_profit = total_profit / len(winning_trades) if winning_trades else 0.0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0.0
        expectancy = (
            (win_rate / 100) * avg_profit + (1 - win_rate / 100) * (-avg_loss)
        ) if n_trades > 0 else 0.0

        # Phase distribution
        phase_counts = pd.Series(self.phase_history).value_counts().to_dict()
        phase_dist = {
            self.phase_detector.get_phase_name(k): v
            for k, v in sorted(phase_counts.items())
        }

        results = {
            "symbol": symbol,
            "date_range": f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}",
            "total_return_pct": round(total_return, 2),
            "annual_return_pct": round(ann_return, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "total_trades": n_trades,
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "avg_profit": round(avg_profit, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy": round(expectancy, 2),
            "final_capital": round(self.cash, 2),
            "phase_distribution": phase_dist,
            "trades": self.trades,
            "equity_curve": eq_df.reset_index().to_dict("records"),
        }
        return results

    def summary(self, results):
        """Print a formatted summary of backtest results."""
        if "error" in results:
            print(f"  Error: {results['error']}")
            return

        print(f"\n{'='*60}")
        print(f"  Backtest Results: {results['symbol']}")
        print(f"  Period: {results['date_range']}")
        print(f"{'='*60}")
        print(f"  Total Return:      {results['total_return_pct']:>8.2f}%")
        print(f"  Annual Return:     {results['annual_return_pct']:>8.2f}%")
        print(f"  Sharpe Ratio:      {results['sharpe_ratio']:>8.2f}")
        print(f"  Max Drawdown:      {results['max_drawdown_pct']:>8.2f}%")
        print(f"  Final Capital:     {results['final_capital']:>10.2f}")
        print(f"  Total Trades:      {results['total_trades']:>8d}")
        print(f"  Win Rate:          {results['win_rate_pct']:>8.1f}%")
        print(f"  Profit Factor:     {results['profit_factor']:>8.2f}")
        print(f"  Avg Profit:        {results['avg_profit']:>8.2f}")
        print(f"  Avg Loss:          {results['avg_loss']:>8.2f}")
        print(f"  Expectancy:        {results['expectancy']:>8.2f}")
        print(f"  Phase Distribution: {results.get('phase_distribution', {})}")
        print(f"{'='*60}")


class OscillationBacktestEngine:
    """Dedicated backtesting engine for the oscillation 做T strategy.

    Unlike the general BacktestEngine, this one:
      - Always runs the oscillation strategy (regardless of phase)
      - Tracks oscillation-specific metrics (channel touches, bounce count, etc.)
      - Tests different oscillation_strength values
    """

    def __init__(self, config):
        self.config = config
        bt_cfg = config.get("backtest", {})

        self.initial_capital = bt_cfg.get("initial_capital", 100000.0)
        self.commission_pct = bt_cfg.get("commission_pct", 0.0003)
        self.slippage_pct = bt_cfg.get("slippage_pct", 0.001)
        self.cooldown_bars = config.get("oscillation_trading", {}).get("cooldown_bars", 3)
        self.phase_stable_bars = config.get("oscillation_trading", {}).get("phase_stable_bars", 5)

        # Oscillation strategy instance
        osc_cfg = OscillationConfig.from_dict(config.get("oscillation_trading", {}))
        self.config_osc = osc_cfg
        self.channel = OscillationChannel(osc_cfg)
        self.trader = OscillationTrader(osc_cfg)

        # Phase detector (for reference)
        self.phase_detector = PhaseDetector(config)

        # State
        self.reset()

    def reset(self):
        self.cash = self.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.entry_idx = 0
        self.entry_bar = 0
        self.trades = []
        self.equity_curve = []
        self.phase_history = []
        self.current_phase = 0
        self.consecutive_phase = 0
        self.exit_bar = -999
        self.consecutive_losses = 0
        self._last_loss_bar = -999

    def run(self, df, symbol="unknown"):
        """Run oscillation backtest on historical data."""
        self.reset()
        df = df.copy()

        # Compute all indicators
        df = self.phase_detector.detect(df)
        df = self.channel.compute(df)

        n = len(df)
        max_pos_pct = self.config.get("risk", {}).get("max_position_pct", 0.95)
        lookback = self.config_osc.lookback

        for i in range(max(lookback, 60), n):
            row = df.iloc[i]
            phase = row["phase"]
            self.current_phase = phase
            self.phase_history.append(phase)

            # Phase stability
            if i > max(lookback, 60):
                self.consecutive_phase = (
                    self.consecutive_phase + 1
                    if phase == self.phase_history[-2]
                    else 1
                )

            if self.position == 0:
                # --- Check entry ---
                signal = self._check_entry(df, i, phase)
                if signal == OscSignal.BUY:
                    price = row["close"] * (1 + self.slippage_pct)
                    allocated = self.cash * max_pos_pct
                    shares = int(allocated / (price * 100)) * 100
                    if shares < 100:
                        continue
                    cost = shares * price
                    commission = cost * self.commission_pct
                    if cost + commission > self.cash:
                        shares = int((self.cash - commission) / (price * 100)) * 100
                        if shares < 100:
                            continue
                        cost = shares * price
                        commission = cost * self.commission_pct

                    self.cash -= cost + commission
                    self.position = shares
                    self.entry_price = price
                    self.entry_idx = i
                    self.entry_bar = len(self.equity_curve)

                    self.trades.append({
                        "entry_date": df.index[i],
                        "entry_price": round(price, 3),
                        "shares": shares,
                        "entry_phase": int(phase),
                        "phase_name": self.phase_detector.get_phase_name(phase),
                        "osc_position": round(row.get("osc_position", 0), 3),
                        "osc_zscore": round(row.get("osc_zscore", 0), 2),
                        "entry_osc_position": round(row.get("osc_position", 0), 3),
                        "entry_osc_zscore": round(row.get("osc_zscore", 0), 2),
                    })

            else:
                # --- Check exit ---
                should_exit = self._check_exit(df, i, phase)

                if should_exit:
                    price = row["close"] * (1 - self.slippage_pct)
                    proceeds = self.position * price
                    commission = proceeds * self.commission_pct
                    self.cash += proceeds - commission

                    trade = self.trades[-1]
                    trade["exit_date"] = df.index[i]
                    trade["exit_price"] = round(price, 3)
                    trade["exit_phase"] = int(phase)
                    trade["exit_phase_name"] = self.phase_detector.get_phase_name(phase)
                    trade["exit_osc_position"] = round(row.get("osc_position", 0), 3)
                    trade["exit_osc_zscore"] = round(row.get("osc_zscore", 0), 2)
                    gross_pnl = proceeds - commission - (
                        trade["shares"] * trade["entry_price"]
                    )
                    trade_cost = trade["shares"] * trade["entry_price"] * self.commission_pct
                    trade["gross_pnl"] = round(gross_pnl - trade_cost, 2)
                    trade["return_pct"] = round(
                        gross_pnl / (trade["shares"] * trade["entry_price"]) * 100, 2
                    )
                    trade["holding_bars"] = i - self.entry_idx

                    # Track consecutive losses
                    if trade["gross_pnl"] > 0:
                        self.consecutive_losses = 0
                    else:
                        self.consecutive_losses += 1
                        self._last_loss_bar = i

                    self.position = 0
                    self.entry_price = 0.0
                    self.exit_bar = i

            # Record equity
            equity = self.cash + self.position * row["close"]
            state = self.channel.get_current_state(df, i)
            self.equity_curve.append({
                "date": df.index[i],
                "equity": round(equity, 2),
                "cash": round(self.cash, 2),
                "position": self.position,
                "close": round(row["close"], 3),
                "phase": int(phase),
                "phase_name": self.phase_detector.get_phase_name(phase),
                "osc_position": round(state["position"], 3),
                "osc_zscore": round(state["zscore"], 2),
                "osc_is_oscillating": state["is_oscillating"],
            })

        # Close remaining position
        if self.position > 0:
            final_price = df.iloc[-1]["close"]
            proceeds = self.position * final_price
            commission = proceeds * self.commission_pct
            self.cash += proceeds - commission
            trade = self.trades[-1]
            trade["exit_date"] = df.index[-1]
            trade["exit_price"] = round(final_price, 3)
            trade["exit_phase"] = int(df.iloc[-1]["phase"])
            trade["exit_phase_name"] = self.phase_detector.get_phase_name(df.iloc[-1]["phase"])
            trade["exit_osc_position"] = round(df.iloc[-1].get("osc_position", 0), 3)
            trade["exit_osc_zscore"] = round(df.iloc[-1].get("osc_zscore", 0), 2)
            gross_pnl = proceeds - commission - (trade["shares"] * trade["entry_price"])
            trade_cost = trade["shares"] * trade["entry_price"] * self.commission_pct
            trade["gross_pnl"] = round(gross_pnl - trade_cost, 2)
            trade["return_pct"] = round(
                gross_pnl / (trade["shares"] * trade["entry_price"]) * 100, 2
            )
            trade["holding_bars"] = n - 1 - self.entry_idx
            self.position = 0

        return self._compute_results(symbol, df)

    def _check_entry(self, df, i, phase):
        """Check entry with guards."""
        # Phase stability: only relevant when not confirm=False
        if self.config.get("oscillation_trading", {}).get("require_oscillation_confirm", True):
            if self.consecutive_phase < self.phase_stable_bars:
                return Signal.NONE
        if i - self.exit_bar < self.cooldown_bars:
            return Signal.NONE

        # Consecutive loss throttle: after N losses, skip a window then resume
        max_loss = self.config.get("oscillation_trading", {}).get("max_consecutive_losses", 5)
        loss_penalty_bars = self.config.get("oscillation_trading", {}).get("loss_penalty_bars", 30)
        if self.consecutive_losses >= max_loss:
            bars_since_last_loss = i - self._last_loss_bar if hasattr(self, '_last_loss_bar') else 0
            if bars_since_last_loss < loss_penalty_bars:
                return Signal.NONE
            # After penalty window, reset the counter so trading can resume
            self.consecutive_losses = max_loss // 2
            self._last_loss_bar = i

        return self.trader.evaluate_entry(df, i, phase)

    def _check_exit(self, df, i, phase):
        """Check exit."""
        return self.trader.evaluate_exit(
            df, i, self.entry_idx, self.entry_price, phase
        )

    def _compute_results(self, symbol, df):
        """Compute performance metrics."""
        eq_df = pd.DataFrame(self.equity_curve)
        if eq_df.empty:
            return {"symbol": symbol, "error": "no data in equity curve"}

        eq_df = eq_df.set_index("date")
        eq_df["daily_return"] = eq_df["equity"].pct_change()
        eq_df["cumulative_return"] = eq_df["equity"] / self.initial_capital - 1

        # Basic metrics
        total_return = (self.cash / self.initial_capital - 1) * 100
        n_days = len(eq_df)
        years = max(n_days / 252, 0.01)
        ann_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100

        daily_returns = eq_df["daily_return"].dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            excess_returns = daily_returns - 0.03 / 252
            sharpe = np.sqrt(252) * excess_returns.mean() / daily_returns.std()
        else:
            sharpe = 0.0

        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        # Trade stats
        n_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.get("gross_pnl", 0) > 0]
        losing_trades = [t for t in self.trades if t.get("gross_pnl", 0) <= 0]
        win_rate = len(winning_trades) / n_trades * 100 if n_trades > 0 else 0.0

        total_profit = sum(t.get("gross_pnl", 0) for t in winning_trades)
        total_loss = abs(sum(t.get("gross_pnl", 0) for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        avg_profit = total_profit / len(winning_trades) if winning_trades else 0.0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0.0
        expectancy = (
            (win_rate / 100) * avg_profit + (1 - win_rate / 100) * (-avg_loss)
        ) if n_trades > 0 else 0.0

        # Oscillation-specific metrics
        osc_df = df.copy()
        osc_df["in_position"] = 0
        for t in self.trades:
            entry_d = t.get("entry_date")
            exit_d = t.get("exit_date")
            if entry_d and exit_d:
                mask = (osc_df.index >= entry_d) & (osc_df.index <= exit_d)
                osc_df.loc[mask, "in_position"] = 1

        entry_positions = [
            t.get("osc_position") for t in self.trades if t.get("osc_position") is not None
        ]
        exit_positions = [
            t.get("exit_osc_position") for t in self.trades if t.get("exit_osc_position") is not None
        ]

        phase_counts = pd.Series(self.phase_history).value_counts().to_dict()
        phase_dist = {
            self.phase_detector.get_phase_name(k): v
            for k, v in sorted(phase_counts.items())
        }

        # Oscillation stats
        total_bars = len(osc_df)
        oscillating_bars = osc_df.get("osc_is_oscillating", pd.Series([False] * total_bars)).sum()
        osc_pct = oscillating_bars / total_bars * 100 if total_bars > 0 else 0

        results = {
            "symbol": symbol,
            "strategy": "oscillation_做T",
            "oscillation_strength": self.config_osc.oscillation_strength,
            "date_range": (
                f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}"
            ),
            "total_return_pct": round(total_return, 2),
            "annual_return_pct": round(ann_return, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "total_trades": n_trades,
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "avg_profit": round(avg_profit, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy": round(expectancy, 2),
            "final_capital": round(self.cash, 2),
            "phase_distribution": phase_dist,
            "oscillation_metrics": {
                "oscillating_bar_pct": round(osc_pct, 1),
                "avg_entry_position": round(np.mean(entry_positions), 3) if entry_positions else 0,
                "avg_exit_position": round(np.mean(exit_positions), 3) if exit_positions else 0,
            },
            "trades": self.trades,
            "equity_curve": eq_df.reset_index().to_dict("records"),
        }
        return results

    def summary(self, results):
        """Print formatted summary."""
        if "error" in results:
            print(f"  Error: {results['error']}")
            return

        print(f"\n{'='*60}")
        print(f"  震荡做T Backtest: {results['symbol']}")
        print(f"  Oscillation Strength: {results.get('oscillation_strength', 'N/A')}")
        print(f"  Period: {results['date_range']}")
        print(f"{'='*60}")
        print(f"  Total Return:      {results['total_return_pct']:>8.2f}%")
        print(f"  Annual Return:     {results['annual_return_pct']:>8.2f}%")
        print(f"  Sharpe Ratio:      {results['sharpe_ratio']:>8.2f}")
        print(f"  Max Drawdown:      {results['max_drawdown_pct']:>8.2f}%")
        print(f"  Final Capital:     {results['final_capital']:>10.2f}")
        print(f"  Total Trades:      {results['total_trades']:>8d}")
        print(f"  Win Rate:          {results['win_rate_pct']:>8.1f}%")
        print(f"  Profit Factor:     {results['profit_factor']:>8.2f}")
        print(f"  Avg Profit:        {results['avg_profit']:>8.2f}")
        print(f"  Avg Loss:          {results['avg_loss']:>8.2f}")
        print(f"  Expectancy:        {results['expectancy']:>8.2f}")

        # Oscillation-specific
        osc_m = results.get("oscillation_metrics", {})
        if osc_m:
            print(f"  ── 震荡指标 ──")
            print(f"  Oscillating Bars:  {osc_m.get('oscillating_bar_pct', 0):>7.1f}%")
            print(f"  Avg Entry Pos:     {osc_m.get('avg_entry_position', 0):>8.3f}")
            print(f"  Avg Exit Pos:      {osc_m.get('avg_exit_position', 0):>8.3f}")

        print(f"  Phase Dist:        {results.get('phase_distribution', {})}")
        print(f"{'='*60}")
