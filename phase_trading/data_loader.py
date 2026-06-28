"""Data loading module — akshare, baostock, tushare, CSV."""

import os
import pandas as pd
import numpy as np

# ── Baostock symbol mapping ──
def symbol_to_baostock(symbol: str) -> str:
    """Convert a plain symbol to baostock format (sh. / sz. prefix)."""
    s = symbol.strip()
    if s.startswith("sh.") or s.startswith("sz."):
        return s
    # Normalize: 00700, 600519 → sh.xxx; 000001, 300308 → sz.xxx
    if s == "00700":
        return "sh.00700"  # 腾讯控股 is in SH-HK
    prefix = s[:3] if len(s) >= 3 else s
    if prefix in ("600", "601", "603", "605", "688", "007"):
        return f"sh.{s}"
    else:
        return f"sz.{s}"


def _normalize_date_bs(d: str) -> str:
    """Convert YYYYMMDD or YYYY-MM-DD to YYYY-MM-DD (baostock format)."""
    d = d.strip().replace("-", "")
    if len(d) == 8:
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    return d


def symbol_to_akshare(symbol: str) -> str:
    """Normalize symbol for akshare A-share API (plain digits)."""
    return symbol.strip()


class DataLoader:
    """Load price data from various sources."""

    def __init__(self, config):
        self.config = config
        self.source = config["data"]["source"]
        self.adjust = config["data"].get("adjust", "qfq")
        self.cache_dir = config["data"].get("cache_dir", "data/cache")

    def load(self, symbol, start_date, end_date):
        """Load data for a given symbol and date range (daily)."""
        if self.source == "akshare":
            return self._load_akshare(symbol, start_date, end_date)
        elif self.source == "baostock":
            return self._load_baostock(symbol, start_date, end_date)
        elif self.source == "tushare":
            return self._load_tushare(symbol, start_date, end_date)
        elif self.source == "csv":
            return self._load_csv(symbol, start_date, end_date)
        else:
            raise ValueError(f"Unknown data source: {self.source}")

    def load_minute(self, symbol, start_date, end_date, freq="30"):
        """Load minute-level K-line data.

        Supported sources: baostock (preferred for long history).
        freq: minute frequency – '5', '15', '30', '60'.
        """
        if self.source == "baostock":
            return self._load_baostock_minute(symbol, start_date, end_date, freq)
        elif self.source == "akshare":
            return self._load_akshare_minute(symbol, start_date, end_date, freq)
        else:
            raise ValueError(f"Minute data not supported for source: {self.source}")

    # ──────────────── akshare ────────────────

    def _load_akshare(self, symbol, start_date, end_date):
        """Load daily data from akshare."""
        try:
            import akshare as ak
        except ImportError:
            raise ImportError("akshare not installed. Run: pip install akshare")

        start = start_date.replace("-", "")
        end = end_date.replace("-", "")
        adjust_map = {"qfq": "qfq", "hfq": "hfq", None: ""}
        adj = adjust_map.get(self.adjust, "qfq")

        df = ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=start, end_date=end, adjust=adj,
        )
        if df.empty:
            raise ValueError(f"No data for symbol {symbol}")

        return self._standardize_akshare_df(df,
            {"日期": "date", "开盘": "open", "收盘": "close",
             "最高": "high", "最低": "low", "成交量": "volume"})

    def _load_akshare_minute(self, symbol, start_date, end_date, freq="30"):
        """Load minute data from akshare (limited history ~1.5 months)."""
        try:
            import akshare as ak
        except ImportError:
            raise ImportError("akshare not installed. Run: pip install akshare")

        if symbol == "00700":
            df = ak.stock_hk_hist_min_em(
                symbol=symbol, period=freq,
                start_date=start_date, end_date=end_date, adjust="qfq",
            )
            # col order: 0=时间, 1=开盘, 2=收盘, 3=最高, 4=最低, 5=涨跌幅, 6=成交额, 7=成交量
            dates = pd.to_datetime(df.iloc[:, 0].values)
            out = pd.DataFrame({
                "open": df.iloc[:, 1].values.astype(float),
                "high": df.iloc[:, 3].values.astype(float),
                "low": df.iloc[:, 4].values.astype(float),
                "close": df.iloc[:, 2].values.astype(float),
                "volume": df.iloc[:, 7].values.astype(float),
            }, index=dates)
        else:
            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol, period=freq,
                start_date=start_date, end_date=end_date, adjust="qfq",
            )
            # col order: 0=时间, 1=开盘, 2=收盘, 3=最高, 4=最低, 5=涨跌幅, 6=涨跌额, 7=成交量, 8=成交额
            dates = pd.to_datetime(df.iloc[:, 0].values)
            out = pd.DataFrame({
                "open": df.iloc[:, 1].values.astype(float),
                "high": df.iloc[:, 3].values.astype(float),
                "low": df.iloc[:, 4].values.astype(float),
                "close": df.iloc[:, 2].values.astype(float),
                "volume": df.iloc[:, 7].values.astype(float),
            }, index=dates)

        out.index.name = "date"
        return out.sort_index()

    @staticmethod
    def _standardize_akshare_df(df, col_map):
        """Convert akshare DataFrame to standard OHLCV format."""
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    # ──────────────── baostock ────────────────

    def _load_baostock(self, symbol, start_date, end_date):
        """Load daily data from baostock (long history)."""
        bscode = symbol_to_baostock(symbol)
        try:
            import baostock as bs
        except ImportError:
            raise ImportError("baostock not installed. Run: pip install baostock")

        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock login failed: {lg.error_msg}")

        rs = bs.query_history_k_data_plus(
            bscode,
            "date,open,high,low,close,volume,amount",
            start_date=_normalize_date_bs(start_date),
            end_date=_normalize_date_bs(end_date),
            frequency="d", adjustflag="2",  # 2 = 前复权
        )
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        bs.logout()

        if not rows:
            raise ValueError(f"No data for {symbol} ({bscode})")

        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df[["open", "high", "low", "close", "volume"]]

    def _load_baostock_minute(self, symbol, start_date, end_date, freq="30"):
        """Load minute-level K-line from baostock (long history, ~6+ months).

        freq: '5', '15', '30', '60'.
        """
        bscode = symbol_to_baostock(symbol)
        try:
            import baostock as bs
        except ImportError:
            raise ImportError("baostock not installed. Run: pip install baostock")

        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock login failed: {lg.error_msg}")

        # Baostock expects date format YYYY-MM-DD
        start_clean = _normalize_date_bs(start_date)
        end_clean = _normalize_date_bs(end_date)
        rs = bs.query_history_k_data_plus(
            bscode,
            "date,time,open,high,low,close,volume,amount",
            start_date=start_clean,
            end_date=end_clean,
            frequency=freq, adjustflag="2",
        )
        if rs is None:
            bs.logout()
            raise ValueError(f"baostock query returned None for {symbol} ({bscode})")

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        bs.logout()

        if not rows:
            raise ValueError(f"No minute data for {symbol} ({bscode})")

        df = pd.DataFrame(rows, columns=["date", "time", "open", "high", "low", "close", "volume", "amount"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Build datetime index from date + time (format: 20260105100000000)
        t = df["time"].str[8:14]
        df["datetime"] = pd.to_datetime(
            df["date"] + " " + t.str[:2] + ":" + t.str[2:4] + ":" + t.str[4:6]
        )
        df = df.set_index("datetime").sort_index()
        return df[["open", "high", "low", "close", "volume"]]

    # ──────────────── tushare ────────────────

    def _load_tushare(self, symbol, start_date, end_date):
        """Load daily data from tushare pro."""
        try:
            import tushare as ts
        except ImportError:
            raise ImportError("tushare not installed. Run: pip install tushare")

        pro = ts.pro_api()
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

        df = df.rename(columns={"trade_date": "date", "vol": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df[["open", "high", "low", "close", "volume"]].astype(float)

    # ──────────────── CSV ────────────────

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
        if self.source in ("akshare", "tushare", "baostock"):
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
