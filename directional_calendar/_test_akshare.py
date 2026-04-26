"""Test: can we get main-contract (continuous) minute data with longer history?"""
import akshare as ak
import pandas as pd

# Try different akshare functions for futures data
tests = [
    ("futures_zh_minute_sina IF2606", lambda: ak.futures_zh_minute_sina(symbol="IF2606", period="5")),
    ("futures_zh_minute_sina IF0", lambda: ak.futures_zh_minute_sina(symbol="IF0", period="5")),
    ("futures_main_sina IF", lambda: ak.futures_main_sina(symbol="IF0")),
    ("futures_display_main_sina IF", lambda: ak.futures_display_main_sina()),
]

for name, fn in tests:
    try:
        df = fn()
        if df is not None and not df.empty:
            print(f"{name}: {len(df)} rows")
            print(f"  cols={list(df.columns)[:8]}")
            if "datetime" in df.columns:
                print(f"  time: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
            elif len(df.columns) > 0:
                print(f"  first={df.iloc[0].to_dict()}")
                print(f"  last={df.iloc[-1].to_dict()}")
        else:
            print(f"{name}: empty/None")
    except Exception as e:
        print(f"{name}: ERROR - {e}")
    print()
