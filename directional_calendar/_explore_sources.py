"""
Long-history backtest using independent contract-pair segments.

Since Sina/akshare only provides ~30 days of minute data per contract,
and expired contracts don't overlap in time with their far-month counterpart,
we use a different approach:

1. Fetch each contract's data independently
2. For basis calculation, use the active near/far pair that was trading
   at that time period
3. Run backtests on each segment, then concatenate results

Key insight: The CURRENT pair (IF2606/IF2609) has overlapping data.
For HISTORICAL pairs, we fetch both contracts and check for overlap.
If no overlap, we skip that pair (the far month wasn't active yet).
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
PERIOD = "5"

# Only fetch currently-active pairs (they have overlapping data)
# For historical analysis, we'll use DAILY data instead
ACTIVE_PAIRS = {
    "IF": ("IF2606", "IF2609"),
    "IH": ("IH2606", "IH2609"),
    "IC": ("IC2606", "IC2609"),
    "IM": ("IM2606", "IM2609"),
}

def fetch_minute(symbol: str):
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


def main():
    # ============================================================
    # Strategy: Use daily futures data from neodata to build 
    # a LONG history, then simulate 5min bars from daily OHLC
    # ============================================================
    
    print("=" * 70)
    print("  STRATEGY CHANGE: Using DAILY data for long history")
    print("  (Sina minute data only covers ~30 days per active contract)")
    print("=" * 70)
    
    # Check what we already have and what's possible
    print("\nAvailable data sources:")
    print("  1. akshare Sina minute: ~30 days only (confirmed)")
    print("  2. akshare futures daily: may have longer history")
    print("  3. neodata fut_daily: need valid token")
    
    # Test option 2: akshare daily futures data
    print("\n--- Testing akshare daily futures data ---")
    try:
        # Try akshare's futures data
        import akshare as ak
        
        # Test: futures historical data
        df_test = ak.futures_zh_daily(symbol="IF2606", market="CFFEX")
        if df_test is not None and not df_test.empty:
            print(f"  futures_zh_daily IF2606: {len(df_test)} rows")
            print(f"  Range: {df_test.iloc[0]} ~ {df_test.iloc[-1]}")
            print(f"  Columns: {list(df_test.columns)}")
        else:
            print("  futures_zh_daily: empty or None")
    except Exception as e:
        print(f"  futures_zh_daily ERROR: {e}")
    
    # Test: futures main contract
    print("\n--- Testing akshare futures_main_sina / spot data ---")
    try:
        df_main = ak.futures_main_sina(symbol="IF0", market="CFFEX")
        if df_main is not None and not df_main.empty:
            print(f"  futures_main_sina IF: {len(df_main)} rows")
            print(f"  Columns: {list(df_main.columns)}")
            print(f"  Last 5 rows:")
            print(df_main.tail())
    except Exception as e:
        print(f"  futures_main_sina ERROR: {e}")
    
    # Test: futures display main contract
    print("\n--- Testing futures_display_main_sina ---")
    try:
        df_disp = ak.futures_display_main_sina(symbol="IF", market="CFFEX")
        if df_disp is not None and not df_disp.empty:
            print(f"  futures_display_main_sina IF: {len(df_disp)} rows")
            print(f"  Columns: {list(df_disp.columns)}")
            print(f"  First: {df_disp.head(2).to_string()}")
            print(f"  Last: {df_disp.tail(2).to_string()}")
    except Exception as e:
        print(f"  ERROR: {e}")

if __name__ == "__main__":
    main()
