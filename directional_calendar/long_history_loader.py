"""
Long-history minute data downloader using akshare (multi-contract stitching).

Strategy: Each active futures contract has ~30 days of 5min data on Sina.
By fetching 6 consecutive contracts and stitching them together,
we get ~6 months of continuous 5min bar data per product.

Contract chain for each product:
  IF: IF2501 → IF2503 → IF2506 → IF2509 → IF2512 → IF2603 → IF2606
  IH/IC/IM: similar pattern

For basis calculation, we pair near-month with far-quarter contracts.
"""
import time
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import akshare as ak

sys.path.insert(0, os.path.dirname(__file__))
from basis_calculator import BasisCalculator

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Configuration
# ============================================================

DATA_DIR = r"C:\Users\wang\WorkBuddy\20260425111208\directional_calendar\data"
PERIOD = "5"

# Contract chains: list of (near_sym, far_sym) pairs to download
# Ordered from oldest to newest - we'll stitch overlapping periods
CONTRACT_CHAINS = {
    "IF": [
        # (near_month, far_month) - far should be next quarter or skip-one
        ("IF2503", "IF2506"),   # Mar-Jun'25
        ("IF2506", "IF2509"),   # Jun-Sep'25
        ("IF2509", "IF2512"),   # Sep-Dec'25
        ("IF2512", "IF2603"),   # Dec-Mar'26
        ("IF2603", "IF2606"),   # Mar-Jun'26
        ("IF2606", "IF2609"),   # Jun-Sep'26 (current)
    ],
    "IH": [
        ("IH2503", "IH2506"),
        ("IH2506", "IH2509"),
        ("IH2509", "IH2512"),
        ("IH2512", "IH2603"),
        ("IH2603", "IH2606"),
        ("IH2606", "IH2609"),
    ],
    "IC": [
        ("IC2503", "IC2506"),
        ("IC2506", "IC2509"),
        ("IC2509", "IC2512"),
        ("IC2512", "IC2603"),
        ("IC2603", "IC2606"),
        ("IC2606", "IC2609"),
    ],
    "IM": [
        ("IM2503", "IM2506"),
        ("IM2506", "IM2509"),
        ("IM2509", "IM2512"),
        ("IM2512", "IM2603"),
        ("IM2603", "IM2606"),
        ("IM2606", "IM2609"),
    ],
}

DIVIDEND_SCHEDULES = {
    "IF": {"2025-03-21": 20.0, "2025-06-20": 28.0, "2025-09-19": 35.0, "2025-12-19": 18.0,
           "2026-03-20": 12.0, "2026-06-20": 35.0, "2026-09-19": 42.0},
    "IH": {"2025-03-21": 16.0, "2025-06-20": 22.0, "2025-09-19": 28.0, "2025-12-19": 11.0,
           "2026-03-20": 8.0,  "2026-06-20": 28.0, "2026-09-19": 35.0},
    "IC": {"2025-03-21": 7.0,  "2025-06-20": 10.0, "2025-09-19": 12.0, "2025-12-19": 6.0,
           "2026-03-20": 4.0,  "2026-06-20": 12.0, "2026-09-19": 15.0},
    "IM": {"2025-03-21": 5.0,  "2025-06-20": 7.0,  "2025-09-19": 8.0,  "2025-12-19": 4.0,
           "2026-03-20": 3.0,  "2026-06-20": 8.0,  "2026-09-19": 10.0},
}

MULTIPLIER = {"IF": 300, "IH": 300, "IC": 200, "IM": 200}
RISK_FREE_RATE = 0.02


def parse_expiry(symbol: str):
    """Parse expiry date from contract symbol like IF2606 -> 3rd Fri June 2026."""
    year = int("20" + symbol[2:4])
    month = int(symbol[4:6])
    from datetime import date
    d = date(year, month, 1)
    fri = (4 - d.weekday()) % 7
    first_fri = d.replace(day=1 + fri)
    return first_fri.replace(day=min(first_fri.day + 14, 28))


def fetch_minute(symbol: str) -> pd.DataFrame | None:
    """Fetch 5min bars for one contract from Sina via akshare."""
    try:
        df = ak.futures_zh_minute_sina(symbol=symbol, period=PERIOD)
        if df is not None and not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"])
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        return None
    except Exception as e:
        print(f"      [ERROR] {symbol}: {e}")
        return None


def calc_basis_for_pair(df_near: pd.DataFrame, df_far: pd.DataFrame,
                         product: str, near_sym: str, far_sym: str) -> pd.DataFrame | None:
    """Calculate annualized basis rate for a near/far pair."""
    merged = pd.merge(
        df_near.rename(columns={
            "open": "near_open", "high": "near_high",
            "low": "near_low", "close": "near_close",
            "volume": "near_vol", "hold": "near_hold",
        }),
        df_far.rename(columns={
            "open": "far_open", "high": "far_high",
            "low": "far_low", "close": "far_close",
            "volume": "far_vol", "hold": "far_hold",
        }),
        on="datetime", how="inner",
    )
    
    if merged.empty:
        return None
    
    merged["product"] = product
    merged["near_symbol"] = near_sym
    merged["far_symbol"] = far_sym
    
    # Basis calculation (vectorized)
    today = datetime.now().date()
    near_exp = parse_expiry(near_sym)
    far_exp = parse_expiry(far_sym)
    
    days_near = max((near_exp - today).days, 1)
    days_far = max((far_exp - today).days, 1)
    dt_near = days_near / 365.0
    dt_far = days_far / 365.0
    dt_diff = dt_far - dt_near
    
    near_c = merged["near_close"].astype(float)
    far_c = merged["far_close"].astype(float)
    
    raw_basis = far_c - near_c
    raw_annual = (raw_basis / near_c / dt_diff) * 100.0
    
    # Dividend adjustment
    div_schedule = DIVIDEND_SCHEDULES.get(product, {})
    div_pts = div_schedule.get(far_exp.strftime("%Y-%m-%d"), 0.0)
    t_mid = (dt_near + dt_far) / 2.0
    dividend_pv = div_pts * np.exp(-RISK_FREE_RATE * t_mid)
    theoretical_spread = near_c * (np.exp(RISK_FREE_RATE * dt_diff) - 1.0) - dividend_pv
    adj_basis = raw_basis - theoretical_spread
    adj_annual = (adj_basis / near_c / dt_diff) * 100.0
    
    merged["raw_basis"] = raw_basis
    merged["raw_annualized_rate"] = raw_annual
    merged["dividend_adjusted_basis"] = adj_basis
    merged["adj_annualized_rate"] = adj_annual
    merged["dividend_between"] = div_pts
    merged["days_to_near_expiry"] = days_near
    merged["days_to_far_expiry"] = days_far
    merged["near_expiry"] = str(near_exp)
    merged["far_expiry"] = str(far_exp)
    
    return merged


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    today_str = datetime.now().strftime("%Y%m%d")
    
    print("=" * 70)
    print("  LONG-HISTORY MINUTE DATA DOWNLOADER")
    print("  Using akshare Sina source, multi-contract stitching")
    print("=" * 70)
    
    all_product_dfs = {}
    
    for product, pairs in CONTRACT_CHAINS.items():
        print(f"\n{'='*70}")
        print(f"  PRODUCT: {product} ({len(pairs)} contract pairs)")
        print(f"{'='*70}")
        
        pair_dfs = []
        
        for near_sym, far_sym in pairs:
            print(f"\n  >> Pair: {near_sym} / {far_sym}")
            
            # Fetch
            df_near = fetch_minute(near_sym)
            time.sleep(0.4)
            
            df_far = fetch_minute(far_sym)
            time.sleep(0.4)
            
            if df_near is None or df_far is None:
                miss = []
                if df_near is None:
                    miss.append(near_sym)
                if df_far is None:
                    miss.append(far_sym)
                print(f"     SKIP (no data: {', '.join(miss)})")
                continue
            
            print(f"     {near_sym}: {len(df_near)} bars ({df_near['datetime'].iloc[0]} ~ {df_near['datetime'].iloc[-1]})")
            print(f"     {far_sym}: {len(df_far)} bars ({df_far['datetime'].iloc[0]} ~ {df_far['datetime'].iloc[-1]})")
            
            # Calculate basis
            result = calc_basis_for_pair(df_near, df_far, product, near_sym, far_sym)
            
            if result is not None and not result.empty:
                print(f"     Merged: {len(result)} bars | "
                      f"adj_mu={result['adj_annualized_rate'].mean():+.3f}% | "
                      f"range=[{result['adj_annualized_rate'].min():+.3f}%, "
                      f"{result['adj_annualized_rate'].max():+.3f}%]")
                pair_dfs.append(result)
            else:
                print(f"     WARN: Merge empty (no time overlap)")
        
        if not pair_dfs:
            print(f"  [SKIP] No data for {product}")
            continue
        
        # Concatenate all pairs and sort by time
        combined = pd.concat(pair_dfs, ignore_index=True)
        combined = combined.sort_values("datetime").reset_index(drop=True)
        
        # Deduplicate (keep last entry when same timestamp appears in overlapping pairs)
        n_before = len(combined)
        combined = combined.drop_duplicates(subset=["datetime"], keep="last")
        n_after = len(combined)
        
        print(f"\n  >>> TOTAL {product}: {n_after} bars (deduped from {n_before})")
        print(f"      Range: {combined['datetime'].iloc[0]} ~ {combined['datetime'].iloc[-1]}")
        
        # Estimate coverage in days
        time_span = (combined["datetime"].iloc[-1] - combined["datetime"].iloc[0]).total_seconds() / 86400
        print(f"      Time span: {time_span:.0f} days (~{time_span/30:.1f} months)")
        print(f"      Adj basis: mu={combined['adj_annualized_rate'].mean():+.3f}% sigma={combined['adj_annualized_rate'].std():.3f}%")
        
        # Save
        outfile = os.path.join(DATA_DIR, f"5min_basis_{product}_long_{today_str}.csv")
        combined.to_csv(outfile, index=False, encoding="utf-8-sig")
        print(f"      Saved: {outfile}")
        
        all_product_dfs[product] = combined
    
    # Final summary
    print(f"\n\n{'='*70}")
    print("  DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    for p, df in all_product_dfs.items():
        span = (df["datetime"].iloc[-1] - df["datetime"].iloc[0]).total_seconds() / 86400
        print(f"  {p}: {len(df):>6} bars | {span:>6.0f} days ({span/30:.1f} mo) | "
              f"mu={df['adj_annualized_rate'].mean():+.3f}% sigma={df['adj_annualized_rate'].std():.3f}%")


if __name__ == "__main__":
    main()
