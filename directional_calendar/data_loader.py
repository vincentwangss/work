"""
data_loader.py
历史基差数据回填脚本

用 akshare 拉取股指期货近远月合约历史行情，
计算分红调整后的年化基差率，持久化到本地 CSV。
供系统启动时加载以初始化统计量。

用法：
    python data_loader.py                    # 默认回填最近90天
    python data_loader.py --days 180         # 回填180天
    python data_loader.py --products IF IH   # 只回填指定品种
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta

import akshare as ak
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from basis_calculator import (
    BasisCalculator, BasisSnapshot, ContractInfo, DividendRecord,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 合约代码生成器
# ------------------------------------------------------------------

def generate_contract_pairs(product: str, start_date: str, end_date: str) -> list:
    """
    生成 [start_date, end_date] 期间的近月/远月合约配对列表。

    股指期货合约规则：
      - 每月第三个周五交割
      - 近月 = 当前月或下个月
      - 远季 = 下个季度末（3/6/9/12月）
    
    Returns:
        [(near_symbol, far_symbol, near_expiry, far_expiry), ...]
    """
    pairs = []
    current = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    while current <= end:
        year = current.year
        month = current.month

        # 确定当前近月和远季合约
        # 近月：当月或次月（取更近的那个未交割的）
        near_expiry = _third_friday(year, month)
        if (current.date() - near_expiry).days > -5:
            # 当月已临近交割，近月跳到次月
            if month == 12:
                near_expiry = _third_friday(year + 1, 1)
            else:
                near_expiry = _third_friday(year, month + 1)

        # 远季：当前季度之后最近的季末合约（3/6/9/12）
        quarter_months = [3, 6, 9, 12]
        next_quarters = [q for q in quarter_months if (q > month) or (year < current.year)]
        if not next_quarters:
            far_year = year + 1
            far_month = 3
        else:
            far_year = year + (next_quarters[0] <= month and len(next_quarters) == 0)
            far_month = next_quarters[0]
            if far_month <= month:
                far_year += 1
        # 更精确的远季选择：取当前近月之后的下一个季月
        far_expiry = _find_next_quarter_expiry(current.date())

        # 避免近远月相同
        if far_expiry <= near_expiry:
            far_expiry = _quarter_after(near_expiry)

        near_sym = f"{product}{near_expiry.year % 100}{near_expiry.month:02d}"
        far_sym = f"{product}{far_expiry.year % 100}{far_expiry.month:02d}"

        pairs.append((near_sym, far_sym, near_expiry, far_expiry))
        current += timedelta(days=7)  # 每周一个快照点即可

    return pairs


def _third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    days_to_fri = (4 - d.weekday()) % 7
    first_fri = d.replace(day=1 + days_to_fri)
    return first_fri.replace(day=min(first_fri.day + 14, 28))


def _quarter_after(d: date) -> date:
    """给定日期后的第一个季末交割日"""
    year = d.year
    quarters = [(year, 6), (year, 9), (year, 12), (year + 1, 3)]
    for y, m in quarters:
        exp = _third_friday(y, m)
        if exp > d:
            return exp
    return _third_friday(year + 2, 3)


def _find_next_quarter_expiry(from_date: date) -> date:
    """从 from_date 往后找下一个季末交割日"""
    quarters = [3, 6, 9, 12]
    for offset in range(16):
        candidate = from_date + timedelta(days=offset * 30)
        for q in quarters:
            exp = _third_friday(candidate.year, q)
            if exp > from_date:
                return exp
    return _third_friday(from_date.year + 2, 12)


# ------------------------------------------------------------------
# 数据拉取
# ------------------------------------------------------------------

def fetch_futures_daily(symbol: str, adjust="") -> Optional[pd.DataFrame]:
    """
    用 akshare 拉取单只期货合约日线数据。
    
    symbol: 如 "IF2506"
    adjust: "" 不复权, "qfq" 前复权, "hfq" 后复权
    
    Returns DataFrame with columns: [日期, 开盘, 收盘, 最高, 最低, 成交量, ...]
             or None on failure.
    """
    try:
        df = ak.futures_main_sina(
            symbol=symbol,
            market="CFE",       # 中金所
        )
        return df
    except Exception as e:
        logger.warning(f"[akshare] {symbol} 拉取失败: {e}")
        return None


def fetch_futures_daily_ak(symbol: str) -> Optional[pd.DataFrame]:
    """
    用 akshare futures_zh_daily_sina 拉取日线数据。
    """
    try:
        df = ak.futures_zh_daily_sina(symbol=f"{symbol}.CFE")
        if df is not None and not df.empty:
            df.columns = ['datetime', 'open', 'high', 'low', 'close', 
                          'volume', 'hold', 'open_oi']
            df['date'] = pd.to_datetime(df['datetime']).dt.date
            df.set_index('date', inplace=True)
            return df[['open', 'high', 'low', 'close', 'volume']]
        return None
    except Exception as e:
        logger.warning(f"[akshare daily] {symbol}: {e}")
        return None


def fetch_spot_index(product_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    拉取对应现货指数日线（用于验证）。
    
    product_code: "sh000300"(沪深300), "sh000016"(上证50), "sh000905"(中证500)
    """
    try:
        df = ak.stock_zh_index_daily(symbol=product_code)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df = df[(df['date'] >= date.fromisoformat(start_date)) & 
                 (df['date'] <= date.fromisoformat(end_date))]
        return df.set_index('date')
    except Exception as e:
        logger.warning(f"[akshare index] {product_code}: {e}")
        return pd.DataFrame()


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

def backfill_basis_data(config_path: str, days_back: int = 90,
                        products: list = None) -> dict:
    """
    主函数：回填历史基差数据并保存到 CSV。

    Returns:
        {product: DataFrame of basis snapshots}
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    instruments = config.get("instruments", {})
    if products is None:
        products = [p for p, cfg in instruments.items() if cfg.get("enabled")]

    calc = BasisCalculator(risk_free_rate=0.0)
    all_results = {}

    for product in products:
        print(f"\n{'='*60}")
        print(f"处理品种: {product} ({instruments[product]['name']})")
        print(f"时间范围: {start_date} ~ {end_date}")

        # 加载分红日程
        div_schedule = instruments[product].get("dividend_schedule", {})
        for ds_str, pts in div_schedule.items():
            expiry = date.fromisoformat(ds_str)
            calc._dividend_table.setdefault(product, {})[expiry] = DividendRecord(
                expiry_date=expiry, dividend_points=float(pts), source="config"
            )

        # 生成合约对
        pairs = generate_contract_pairs(product, start_date, end_date)
        print(f"共 {len(pairs)} 个合约对周期需要拉取")

        snapshots = []

        # 对每个合约周期拉取数据
        processed_pairs = set()
        for near_sym, far_sym, near_exp, far_exp in pairs:
            pair_key = (near_sym, far_sym)
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            print(f"\n  拉取 {near_sym}/{far_sym} "
                  f"(交割{near_exp.strftime('%Y-%m-%d')}/{far_exp.strftime('%Y-%m-%d')})...")

            df_near = fetch_futures_daily_ak(near_sym)
            df_far = fetch_futures_daily_ak(far_sym)

            if df_near is None or df_far is None or df_near.empty or df_far.empty:
                print(f"    ⚠️ 缺少数据，跳过")
                continue

            # 合并两合约数据（按日期 inner join）
            merged = df_near.join(df_far, lsuffix='_near', rsuffix='_far',
                                  how='inner')
            if merged.empty:
                continue

            today_val = datetime.now().date()

            for idx, row in merged.iterrows():
                trade_date = idx if isinstance(idx, date) else row.get('date', today_val)
                
                try:
                    snap = calc.calc_snapshot(
                        ContractInfo(near_sym, product, near_exp, float(row['close_near'])),
                        ContractInfo(far_sym, product, far_exp, float(row['close_far'])),
                        product,
                        as_of=datetime.combine(trade_date, datetime.min.time()),
                    )
                    snapshots.append(snap)
                except Exception as e:
                    logger.debug(f"计算异常 {trade_date}: {e}")

            print(f"    ✅ 累计 {len(snapshots)} 条快照")

        # 导出为 DataFrame 并保存 CSV
        if snapshots:
            result_df = calc.export_history_df(product)
            output_dir = os.path.join(os.path.dirname(config_path), "data")
            os.makedirs(output_dir, exist_ok=True)
            
            csv_path = os.path.join(output_dir, f"basis_history_{product}.csv")
            result_df.to_csv(csv_path, encoding='utf-8-sig')
            
            all_results[product] = result_df
            
            # 打印统计摘要
            adj_col = 'adj_annualized_rate'
            raw_col = 'raw_annualized_rate'
            if adj_col in result_df.columns:
                print(f"\n  📊 统计 ({len(result_df)} 条):")
                print(f"     原始年化基差率 μ={result_df[raw_col].mean():+.2f}% σ={result_df[raw_col].std():.2f}%")
                print(f"     分红调整后 μ={result_df[adj_col].mean():+.2f}% σ={result_df[adj_col].std():.2f}%")
                print(f"     最小={result_df[adj_col].min():+.2f}% 最大={result_df[adj_col].max():+.2f}%")
                print(f"     已保存到: {csv_path}")
        else:
            print(f"\n  ⚠️ {product} 无有效数据")

    return all_results


def load_csv_into_calculator(calc: BasisCalculator, data_dir: str, 
                              products: list) -> int:
    """
    从已保存的 CSV 文件加载历史快照到计算器（系统启动时调用）。
    
    Returns: 总加载数量
    """
    total = 0
    for product in products:
        csv_path = os.path.join(data_dir, f"basis_history_{product}.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"历史文件不存在: {csv_path}")
            continue
        
        df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
        snapshots = []
        for _, row in df.iterrows():
            from basis_calculator import BasisSnapshot
            snap = BasisSnapshot(
                timestamp=pd.Timestamp(row.name).to_pydatetime(),
                product=product,
                near_symbol=str(row.get('near_symbol', '')),
                far_symbol=str(row.get('far_symbol', '')),
                near_price=row['near_price'],
                far_price=row['far_price'],
                raw_basis=row['raw_basis'],
                raw_annualized_rate=row['raw_annualized_rate'],
                dividend_adjusted_basis=row.get('adj_basis', row['raw_basis']),
                adj_annualized_rate=row.get('adj_annualized_rate', row['raw_annualized_rate']),
                near_expiry=date.fromisoformat(str(row['near_expiry'])) if 'near_expiry' in row else date.today(),
                far_expiry=date.fromisoformat(str(row['far_expiry'])) if 'far_expiry' in row else date.today(),
                days_near=int(row.get('days_near', 30)),
                days_far=int(row.get('days_far', 120)),
                dividend_between=row.get('dividend_between', 0),
            )
            snapshots.append(snap)
        
        calc.load_history_from_records(product, snapshots)
        total += len(snapshots)
        print(f"  加载 {product}: {len(snapshots)} 条历史记录")
    
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="历史基差数据回填工具")
    parser.add_argument("--config",
                        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
                        help="配置文件路径")
    parser.add_argument("--days", type=int, default=90,
                        help="回填天数（默认90）")
    parser.add_argument("--products", nargs="+", default=None,
                        help="指定品种（默认全部启用）")
    parser.add_argument("--load-only", action="store_true",
                        help="仅从已有CSV加载到内存测试（不重新拉取）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.load_only:
        calc = BasisCalculator()
        data_dir = os.path.join(os.path.dirname(args.config), "data")
        products = args.products or ["IF", "IH", "IC"]
        n = load_csv_into_calculator(calc, data_dir, products)
        print(f"\n总计加载 {n} 条记录")
        
        # 测试 Z-score 计算
        for p in products:
            mean, std, count = calc.get_stats(p, use_adjusted=True)
            print(f"{p}: mean={mean:.2f}% std={std:.2f}% count={count}")
    else:
        results = backfill_basis_data(args.config, args.days, args.products)
        
        # 输出汇总表
        if results:
            print(f"\n\n{'='*70}")
            print("回填完成 — 基差统计汇总")
            print(f"{'='*70}")
            print(f"{'品种':<6s} {'样本数':>6s} {'原始μ':>8s} {'原始σ':>8s} {'调整μ':>8s} {'调整σ':>8s}")
            print("-"*56)
            for product, df in results.items():
                raw = df['raw_annualized_rate']
                adj = df.get('adj_annualized_rate', raw)
                print(f"{product:<6s} {len(df):>6d} {raw.mean():>+8.2f}% {raw.std():>8.2f} "
                      f"{adj.mean():>+8.2f}% {adj.std():>8.2f}")
