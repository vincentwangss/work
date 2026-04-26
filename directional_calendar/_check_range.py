"""Check: how much historical minute data can we get from akshare?"""
import akshare as ak
import pandas as pd

# Test IF2606 - how far back does 5min data go?
print("Testing akshare futures_zh_minute_sina historical depth...")
print()

for sym in ["IF2606", "IH2606", "IC2606", "IM2606"]:
    for period in ["5"]:
        try:
            df = ak.futures_zh_minute_sina(symbol=sym, period=period)
            if df is not None and not df.empty:
                first_dt = df["datetime"].iloc[0]
                last_dt = df["datetime"].iloc[-1]
                n = len(df)
                # Estimate days covered (assuming ~48 bars/day for 5min during trading hours)
                days_covered = n / 48
                print(f"  {sym} period={period}: {n} bars | {first_dt} ~ {last_dt} | ~{days_covered:.0f} days")
            else:
                print(f"  {sym} period={period}: EMPTY")
        except Exception as e:
            print(f"  {sym} period={period}: ERROR - {e}")
    print()
