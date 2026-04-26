"""
minute_data_loader.py
5 分钟级别历史基差数据回填

基于 akshare futures_zh_minute_sina 拉取近月+远月合约的 5 分钟 K 线，
逐根 Bar 计算分红调整后的年化基差率，保存为 CSV。

特点：
  - 支持多品种（IF/IH/IC/IM）
  - 自动按合约周期拼接（跨合约换月时自动切换）
  - 每根 5 分钟 Bar 计算一次基差快照
  - 数据量：单合约约 204 根/天 × N 天（新浪限制最近约 5 个交易日）
  
用法：
    # 回填 IF 近远月 5分钟基差
    python minute_data_loader.py --products IF

    # 多品种 + 自定义近远月合约
    python minute_data_loader.py --products IF IH IC --near IF2506 --far IF2609

    # 只拉取不计算，原始数据保存
    python minute_data_loader.py --raw-only --products IF

输出：
    data/min5_basis_IF_20260425.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from basis_calculator import BasisCalculator, ContractInfo, DividendRecord

logger = logging.getLogger(__name__)

# 默认 K 线周期
DEFAULT_PERIOD = "5"


def _fuzzy_get_dividend(div_table: Dict, target_date: date,
                        tolerance_days: int = 30) -> float:
    """
    在分红预测表中查找距目标日期最近的记录（容差 ±tolerance_days）。
    返回匹配到的预测分红点数，未找到返回 0.0。

    设计思路：券商研报的交割日与代码推算的交割日常差1~2天，
    只要偏差在 30 个自然日内就采用最近的预测值（合理下限估计）。
    """
    if not div_table:
        return 0.0

    # 精确匹配
    if target_date in div_table:
        return div_table[target_date].dividend_points

    # 最近可用匹配
    best_date = None
    best_delta = 999
    for d in div_table:
        delta = abs((d - target_date).days)
        if delta <= tolerance_days and delta < best_delta:
            best_date = d
            best_delta = delta

    if best_date is not None:
        rec = div_table[best_date]
        logger.info(
            f"[分红预测匹配] 目标{target_date} -> {best_date} "
            f"(差{best_delta}天, {rec.dividend_points:.2f}pt)"
        )
        return rec.dividend_points

    logger.warning(
        f"[分红] 缺少 {target_date} 附近(±{tolerance_days}天)的分红预测，"
        f"将使用 0.0（基差偏差可能达数十点）"
    )
    return 0.0


# ------------------------------------------------------------------
# 数据拉取
# ------------------------------------------------------------------

def fetch_minute_bars(symbol: str, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    """
    从新浪拉取期货 K 线（默认 5 分钟）。

    Args:
        symbol: 合约代码，如 "IF2606" (不带后缀)
        period: "1"/"5"/"15"/"30"/"60"

    Returns:
        DataFrame with columns [datetime, open, high, low, close, volume, hold]
        或 None
    """
    import akshare as ak

    try:
        df = ak.futures_zh_minute_sina(symbol=symbol, period=period)
        if df is not None and not df.empty:
            df["symbol"] = symbol
            return df
        logger.warning(f"[{period}min] {symbol} 返回空数据")
        return None
    except Exception as e:
        logger.error(f"[{period}min] {symbol} 拉取失败: {e}")
        return None


def fetch_pair_minute(
    near_sym: str,
    far_sym: str,
    product: str,
    period: str = DEFAULT_PERIOD,
) -> pd.DataFrame:
    """拉取一对近远月合约的 K 线数据并合并"""

    # 拉取时加间隔避免被限流
    df_near = fetch_minute_bars(near_sym, period=period)
    time.sleep(0.3)
    df_far = fetch_minute_bars(far_sym, period=period)

    if df_near is None or df_far is None or df_near.empty or df_far.empty:
        logger.warning(f"[{period}min] {near_sym} 或 {far_sym} 无数据")
        return pd.DataFrame()

    # 标准化列名
    for df in [df_near, df_far]:
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 按 datetime inner join（只保留两边都有的时间点）
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
        on="datetime",
        how="inner",
    )

    merged["product"] = product
    merged["near_symbol"] = near_sym
    merged["far_symbol"] = far_sym

    print(f"  合并后: {merged.shape[0]} 根{period}minBar "
          f"({merged['datetime'].iloc[0]} ~ {merged['datetime'].iloc[-1]})")

    return merged


# ------------------------------------------------------------------
# 基差计算（向量化）
# ------------------------------------------------------------------

def calc_basis_vectorized(
    df: pd.DataFrame,
    product: str,
    calc: BasisCalculator,
    near_sym: str,
    far_sym: str,
    near_expiry: date,
    far_expiry: date,
    dividend_between: float = 0.0,
    risk_free_rate: float = 0.020,
) -> pd.DataFrame:
    """
    向量化计算每根 Bar 的基差率。

    在 DataFrame 上直接操作，避免逐行循环，速度快 100x+
    """
    n = len(df)
    if n == 0:
        return df

    today = datetime.now().date()

    # 剩余天数
    days_near = max((near_expiry - today).days, 1)
    days_far = max((far_expiry - today).days, 1)
    dt_near = days_near / 365.0
    dt_far = days_far / 365.0
    dt_diff = dt_far - dt_near

    near_c = df["near_close"].astype(float)
    far_c = df["far_close"].astype(float)

    # 原始基差
    raw_basis = far_c - near_c
    raw_annual = (raw_basis / near_c / dt_diff) * 100.0

    # 分红调整
    t_mid = (dt_near + dt_far) / 2.0
    dividend_pv = dividend_between * np.exp(-risk_free_rate * t_mid)
    theoretical_spread = near_c * (np.exp(risk_free_rate * dt_diff) - 1.0) - dividend_pv
    adj_basis = raw_basis - theoretical_spread
    adj_annual = (adj_basis / near_c / dt_diff) * 100.0

    df["raw_basis"] = raw_basis
    df["raw_annualized_rate"] = raw_annual
    df["dividend_adjusted_basis"] = adj_basis
    df["adj_annualized_rate"] = adj_annual
    df["dividend_between"] = dividend_between
    df["days_to_near_expiry"] = days_near
    df["days_to_far_expiry"] = days_far
    df["near_expiry"] = str(near_expiry)
    df["far_expiry"] = str(far_expiry)

    # 同时存入计算器的历史序列（供 Z-score）
    for i in range(n):
        snap = type('Snap', (), {
            'timestamp': df['datetime'].iloc[i].to_pydatetime(),
            'product': product,
            'near_symbol': near_sym,
            'far_symbol': far_sym,
            'near_price': float(near_c.iloc[i]),
            'far_price': float(far_c.iloc[i]),
            'raw_basis': float(raw_basis.iloc[i]),
            'raw_annualized_rate': float(raw_annual.iloc[i]),
            'dividend_adjusted_basis': float(adj_basis.iloc[i]),
            'adj_annualized_rate': float(adj_annual.iloc[i]),
            'near_expiry': near_expiry,
            'far_expiry': far_expiry,
            'days_near': days_near,
            'days_far': days_far,
            'dividend_between': dividend_between,
        })()
        calc._push_history(product, snap)

    return df


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

def backfill_minute_data(
    config_path: str,
    products: List[str] = None,
    near_override: Dict[str, str] = None,
    far_override: Dict[str, str] = None,
    raw_only: bool = False,
    output_format: str = "csv",
    period: str = DEFAULT_PERIOD,
) -> Dict[str, pd.DataFrame]:
    """
    主函数：回填历史基差数据（默认 5 分钟）。

    Returns:
        {product: DataFrame}
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    instruments = config.get("instruments", {})
    if products is None:
        products = [p for p in instruments if instruments[p].get("enabled")]

    calc = BasisCalculator()
    results = {}

    output_dir = os.path.join(os.path.dirname(config_path), "data")
    os.makedirs(output_dir, exist_ok=True)

    today_str = datetime.now().strftime("%Y%m%d")
    period_label = f"{period}min"

    for product in products:
        cfg = instruments.get(product)
        if not cfg or not cfg.get("enabled"):
            continue

        near_sym = near_override.get(product, cfg["near_symbol"]) if near_override else cfg["near_symbol"]
        far_sym = far_override.get(product, cfg["far_symbol"]) if far_override else cfg["far_symbol"]

        print(f"\n{'='*60}")
        print(f"  品种: {product} ({cfg['name']}) | 周期: {period}min")
        print(f"  合约对: {near_sym} / {far_sym}")
        print(f"{'='*60}")

        # 加载分红
        div_schedule = cfg.get("dividend_schedule", {})
        div_pts = 0.0
        if div_schedule:
            calc.load_dividend_schedule(product, div_schedule, "config")

            def get_exp(s):
                y = int("20"+s[2:4]); m=int(s[4:6]); d=date(y,m,1); ff=(4-d.weekday())%7; return d.replace(day=min(1+ff+14,28))

            far_exp = get_exp(far_sym)

            # 使用模糊匹配（容差±3天）查找分红记录
            div_table = calc._dividend_table.get(product, {})
            div_pts = _fuzzy_get_dividend(div_table, far_exp)
            print(f"  分红贴水: {div_pts:.2f}点 (截止{far_exp})")

        # 拉取数据
        merged = fetch_pair_minute(near_sym, far_sym, product, period=period)
        if merged.empty:
            print(f"  [SKIP] no data for {product}")
            continue

        # 计算基差（如果不是 raw_only 模式）
        if not raw_only:
            def _get_exp(s):
                y=int("20"+s[2:4]); m=int(s[4:6]); d=date(y,m,1); ff=(4-d.weekday())%7; return d.replace(day=min(1+ff+14,28))

            near_exp = _get_exp(near_sym)
            far_exp = _get_exp(far_sym)

            result_df = calc_basis_vectorized(
                merged, product, calc, near_sym, far_sym, near_exp, far_exp, div_pts,
                risk_free_rate=0.0,
            )

            # 统计摘要
            adj_col = "adj_annualized_rate"
            raw_col = "raw_annualized_rate"
            print(f"\n  [STATS] {period_label} basis stats ({len(result_df)} bars):")
            print(f"     raw_annual: mean={result_df[raw_col].mean():+.3f}% std={result_df[raw_col].std():.3f}%")
            print(f"     adj_annual: mean={result_df[adj_col].mean():+.3f}% std={result_df[adj_col].std():.3f}%")
            print(f"     range: [{result_df[adj_col].min():+.3f}%, {result_df[adj_col].max():+.3f}%]")

            # Z-score 统计
            history = calc._history.get(product, [])
            if len(history) >= 10:
                vals = [s.adj_annualized_rate for s in history]
                arr = np.array(vals)
                print(f"     full mu={arr.mean():+.3f}% sigma={arr.std():.3f}%")
        else:
            result_df = merged

        # 保存
        ext = ".parquet" if output_format == "parquet" else ".csv"
        filename = f"{period_label}_basis_{product}_{today_str}{ext}"
        filepath = os.path.join(output_dir, filename)

        if output_format == "parquet":
            result_df.to_parquet(filepath, index=False)
        else:
            result_df.to_csv(filepath, index=False, encoding="utf-8-sig")

        results[product] = result_df
        print(f"  [OK] saved: {filepath} ({len(result_df)} rows)")

    # 输出汇总
    if results:
        print(f"\n\n{'='*60}")
        print(f"  {period_label} data backfill complete")
        print(f"{'='*60}")
        for p, df in results.items():
            print(f"  {p}: {df.shape[0]} rows | "
                  f"adj_mu={df.get('adj_annualized_rate', df['raw_basis']).mean():+.3f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"{DEFAULT_PERIOD} minute level basis data backfill tool")
    parser.add_argument("--config",
                        default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--products", nargs="+",
                        help="specify products, e.g., IF IH IC")
    parser.add_argument("--near", nargs="+",
                        help="override near-month contract code, paired with --products")
    parser.add_argument("--far", nargs="+",
                        help="override far-season contract code")
    parser.add_argument("--raw-only", action="store_true",
                        help="only fetch raw quotes, skip basis calculation")
    parser.add_argument("--format", choices=["parquet", "csv"],
                        default="csv", help="output format")
    parser.add_argument("--period", choices=["1", "5", "15", "30", "60"],
                        default=DEFAULT_PERIOD,
                        help=f"K-line period in minutes (default: {DEFAULT_PERIOD})")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    near_map = {}
    far_map = {}
    if args.near and args.products:
        for sym, p in zip(args.near, args.products):
            near_map[p] = sym
    if args.far and args.products:
        for sym, p in zip(args.far, args.products):
            far_map[p] = sym

    backfill_minute_data(
        config_path=args.config,
        products=args.products,
        near_override=near_map if near_map else None,
        far_override=far_map if far_map else None,
        raw_only=args.raw_only,
        output_format=args.format,
        period=args.period,
    )
