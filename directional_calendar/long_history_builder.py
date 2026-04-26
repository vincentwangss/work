"""
Long-history backtest data builder.
Source: akshare futures_main_sina (daily, 2017~now) → synthetic 5min → basis
"""
import time
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import akshare as ak

warnings.filterwarnings("ignore")

DATA_DIR = r"C:\Users\wang\WorkBuddy\20260425111208\directional_calendar\data"
PRODUCTS = ["IF", "IH", "IC", "IM"]
MULTIPLIER = {"IF": 300, "IH": 300, "IC": 200, "IM": 200}
BASIS_MULTIPLIER = {"IF": 1.0, "IH": 0.7, "IC": 1.3, "IM": 1.6}


def fetch_futures_daily(product: str) -> pd.DataFrame | None:
    try:
        df = ak.futures_main_sina(symbol=f"{product}0")
        if df is not None and not df.empty:
            col_map = {
                "date": "日期", "open": "开盘价", "high": "最高价",
                "low": "最低价", "close": "收盘价", "volume": "成交量",
                "hold": "持仓量"
            }
            # Reverse map (Chinese→English)
            rename = {v: k for k, v in col_map.items() if v in df.columns}
            df = df.rename(columns=rename)
            for c in ["date"]:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c])
            for c in ["open", "high", "low", "close", "volume", "hold"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)
            return df
    except Exception as e:
        print(f"      [ERROR] {product}: {e}")
    return None


def expand_daily_to_5min(df_daily: pd.DataFrame, bars_per_day: int = 48) -> pd.DataFrame:
    """Expand daily OHLC to synthetic 5-min bars with realistic intraday path."""
    rng = np.random.RandomState(42)
    rows = []
    
    for _, row in df_daily.iterrows():
        dt_date = row["date"]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        v = float(row.get("volume", 0))
        hold_v = row.get("hold", 0)
        
        if pd.isna(o) or pd.isna(c) or o <= 0:
            continue
        
        total_range = max(h - l, abs(o) * 0.001)
        
        # Generate timestamps: morning 9:30-11:30 (24 bars), afternoon 13:00-15:00 (24 bars)
        times = []
        for i in range(48):
            if i < 24:
                m = 30 + i * 5
                hr, mn = 9 + m // 60, m % 60
            else:
                m = (i - 24) * 5
                hr, mn = 13 + m // 60, m % 60
            times.append(f"{hr:02d}:{mn:02d}")
        
        n = min(len(times), bars_per_day)
        
        # Brownian bridge to close with realistic volatility pattern
        prices = np.zeros(n + 1)
        prices[0] = o
        
        vol_pattern = np.ones(n)
        vol_pattern[:8] *= 1.5     # morning high vol
        vol_pattern[20:28] *= 0.5   # lunch low vol
        vol_pattern[-8:] *= 1.3     # closing vol
        vol_pattern /= vol_pattern.mean()
        
        increments = rng.randn(n) * vol_pattern * 0.25
        
        for i in range(n):
            progress = (i + 1) / n
            bridge = o + (c - o) * progress
            noise = (rng.randn() - 0.3 * progress) * total_range / 3
            prices[i + 1] = bridge + noise + np.cumsum([0] + list(increments[:i+1]))[-1] * total_range / (n ** 0.5) * 0.3
        
        prices[-1] = c
        prices = np.clip(prices, l - total_range * 0.05, h + total_range * 0.05)
        
        vol_per_bar = max(v / n, 1)
        
        for i in range(n):
            bo, bcl = prices[i], prices[i + 1]
            bh = max(bo, bcl) + abs(rng.randn()) * total_range * 0.08
            bl = min(bo, bcl) - abs(rng.randn()) * total_range * 0.08
            t_str = times[i] if i < len(times) else "15:00"
            
            rows.append({
                "datetime": f"{dt_date.strftime('%Y-%m-%d')} {t_str}:00",
                "open": round(bo, 1), "high": round(bh, 1),
                "low": round(bl, 1), "close": round(bcl, 1),
                "volume": int(vol_per_bar * (1 + 0.5 * rng.rand())),
                "hold": int(hold_v) if not pd.isna(hold_v) else 0,
            })
    
    result = pd.DataFrame(rows)
    if not result.empty:
        result["datetime"] = pd.to_datetime(result["datetime"])
        result = result.sort_values("datetime").reset_index(drop=True)
    return result


def add_basis_columns(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """
    Add synthetic basis columns that MATCH real basis characteristics.
    
    Real data stats (from actual 5min IF/IH/IC/IM):
      IF:  mu=-9.28%, sigma=0.42%
      IH:  mu=-7.81%, sigma=0.57%
      IC:  mu=-10.39%, sigma=0.75%
      IM:  mu=-13.35%, sigma=0.54%
    
    We generate a mean-reverting process with these parameters.
    """
    fut_c = df["close"].astype(float)
    
    # Target statistics (from real data)
    TARGET_STATS = {
        "IF": {"mu": -9.28, "sigma": 0.42},
        "IH": {"mu": -7.81, "sigma": 0.57},
        "IC": {"mu": -10.39, "sigma": 0.75},
        "IM": {"mu": -13.35, "sigma": 0.54},
    }
    target = TARGET_STATS.get(product, {"mu": -9.0, "sigma": 0.5})
    target_mu = target["mu"]   # % annualized
    target_sigma = target["sigma"]  # % annualized
    
    n = len(df)
    
    # Generate Ornstein-Uhlenbeck process for basis (mean-reverting)
    # dX = theta * (mu - X) * dt + sigma * dW
    rng = np.random.RandomState(42)
    theta = 0.01  # mean reversion speed (per bar)
    dt = 1.0     # time step per 5min bar
    
    # Scale sigma to per-bar: annual_sigma / sqrt(252*48)
    # But OU process reduces effective sigma by sqrt(theta/2) factor
    # So we need to INCREASE the input sigma to compensate
    bars_per_year = 252 * 48  # ~12096 bars/year
    ou_factor = np.sqrt(2.0 / theta)  # OU variance reduction factor
    sigma_per_bar = target_sigma * ou_factor / np.sqrt(bars_per_year)
    
    basis = np.zeros(n)
    basis[0] = target_mu  # start at long-term mean
    
    noise = rng.randn(n) * sigma_per_bar
    for i in range(1, n):
        drift = theta * (target_mu - basis[i-1]) * dt
        basis[i] = basis[i-1] + drift + noise[i]
    
    # Convert to annualized rate (already in % terms)
    raw_annual = pd.Series(basis, index=df.index)
    raw_basis = raw_annual / 100.0 * fut_c * 45/365  # approximate point basis
    
    mult = BASIS_MULTIPLIER.get(product, 1.0)
    
    df = df.copy()
    df["near_symbol"] = f"{product}MAIN"
    df["far_symbol"] = f"{product}FAR_M"
    df["product"] = product
    df["raw_basis"] = raw_basis
    df["raw_annualized_rate"] = raw_annual
    df["dividend_adjusted_basis"] = raw_basis
    df["adj_annualized_rate"] = raw_annual
    df["near_price"] = fut_c
    df["far_price"] = fut_c - abs(raw_basis) * mult  # far at discount
    df["near_close"] = fut_c
    df["far_close"] = df["far_price"]
    df["days_to_near_expiry"] = 45
    df["days_to_far_expiry"] = 135
    
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    today_str = datetime.now().strftime("%Y%m%d")
    
    print("=" * 70)
    print(f"  LONG-HISTORY DATA BUILDER ({today_str})")
    print(f"  Source: akshare futures_main_sina → synthetic 5min")
    print("=" * 70)
    
    results = {}
    
    for product in PRODUCTS:
        print(f"\n{'─'*60}")
        print(f"  Product: {product}")
        print(f"{'─'*60}")
        
        print(f"  [1/3] Fetching daily data...")
        df_daily = fetch_futures_daily(product)
        if df_daily is None or df_daily.empty:
            print(f"  SKIP: no data")
            continue
        print(f"       {len(df_daily)} days ({df_daily['date'].iloc[0].date()} ~ {df_daily['date'].iloc[-1].date()})")
        
        print(f"  [2/3] Expanding to 5min...")
        df_5min = expand_daily_to_5min(df_daily)
        print(f"       {len(df_5min)} bars")
        
        print(f"  [3/3] Computing basis...")
        df_basis = add_basis_columns(df_5min, product)
        
        # Save
        outfile = os.path.join(DATA_DIR, f"5min_basis_{product}_long_{today_str}.csv")
        df_basis.to_csv(outfile, index=False, encoding="utf-8-sig")
        
        span_days = (df_basis["datetime"].iloc[-1] - df_basis["datetime"].iloc[0]).total_seconds() / 86400
        mu = df_basis["adj_annualized_rate"].mean()
        sig = df_basis["adj_annualized_rate"].std()
        
        print(f"       SAVED: {outfile}")
        print(f"       Stats: {span_days:.0f} days ({span_days/30:.1f} mo) | mu={mu:+.3f}% sigma={sig:.3f}%")
        
        results[product] = df_basis
    
    print(f"\n\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for p, df in results.items():
        span = (df["datetime"].iloc[-1] - df["datetime"].iloc[0]).total_seconds() / 86400
        mu = df["adj_annualized_rate"].mean()
        sig = df["adj_annualized_rate"].std()
        print(f"  {p}: {len(df):>8,} bars | {span:>6.0f}d ({span/30:>5.1f}mo) | mu={mu:>+6.3f}% sigma={sig:>5.3f}%")


if __name__ == "__main__":
    main()
