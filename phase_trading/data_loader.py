"""Data loading module - supports akshare (Chinese stocks) and CSV."""

import os
import pandas as pd
import numpy as np


class DataLoader:
    """Load price data from various sources."""

    def __init__(self, config):
        self.config = config
        self.source = config["data"]["source"]
        self.adjust = config["data"].get("adjust", "qfq")
        self.cache_dir = config["data"].get("cache_dir", "data/cache")

    def load(self, symbol, start_date, end_date):
        """Load data for a given symbol and date range."""
        if self.source == "akshare":
            return self._load_akshare(symbol, start_date, end_date)
        elif self.source == "tushare":
            return self._load_tushare(symbol, start_date, end_date)
        elif self.source == "csv":
            return self._load_csv(symbol, start_date, end_date)
        else:
            raise ValueError(f"Unknown data source: {self.source}")

    def _load_akshare(self, symbol, start_date, end_date):
        """Load data from akshare (Chinese A-share market)."""
        try:
            import akshare as ak
        except ImportError:
            raise ImportError(
                "akshare not installed. Run: pip install akshare"
            )

        # Normalize date format
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")

        adjust_map = {"qfq": "qfq", "hfq": "hfq", None: ""}
        adj = adjust_map.get(self.adjust, "qfq")

        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust=adj,
        )

        if df.empty:
            raise ValueError(f"No data for symbol {symbol}")

        # Standardize column names
        col_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df

    def _load_tushare(self, symbol, start_date, end_date):
        """Load data from tushare pro."""
        try:
            import tushare as ts
        except ImportError:
            raise ImportError("tushare not installed. Run: pip install tushare")

        pro = ts.pro_api()
        # Normalize symbol: if just digits, treat as SZ for 00/30/20 codes, SH for 60
        if symbol.isdigit():
            prefix = symbol[:2]
            if prefix in ("60", "68"):
                ts_code = f"{symbol}.SH"
            elif prefix in ("00", "30", "20"):
                ts_code = f"{symbol}.SZ"
            else:
                ts_code = f"{symbol}.SZ"
        else:
            ts_code = symbol

        start = start_date.replace("-", "")
        end = end_date.replace("-", "")

        df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
        if df is None or df.empty:
            raise ValueError(f"No data for symbol {symbol} (ts_code={ts_code})")

        df = df.rename(columns={
            "trade_date": "date",
            "vol": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df

    def _load_csv(self, symbol, start_date, end_date):
        """Load data from CSV file."""
        path = symbol if os.path.exists(symbol) else os.path.join("data", f"{symbol}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")

        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df.sort_index()
        df = df.loc[start_date:end_date]
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        return df

    def list_available(self):
        """List available symbols."""
        if self.source in ("akshare", "tushare"):
            return None  # all A-share stocks available
        data_dir = "data"
        if not os.path.exists(data_dir):
            return []
        return [f.replace(".csv", "") for f in os.listdir(data_dir) if f.endswith(".csv")]


def compute_returns(df):
    """Add daily return columns to DataFrame."""
    df = df.copy()
    df["daily_return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df
