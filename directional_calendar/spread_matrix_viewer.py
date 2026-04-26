"""
spread_matrix_viewer.py
股指期货价差矩阵可视化工具

功能：
  1. 读取 ccfx 5分钟基差数据（含分红调整）
  2. 构建四合约价差矩阵（C1-C2, C1-C3, C1-C4, C2-C3, C2-C4, C3-C4）
  3. 生成交互式 HTML 报告：
     - 价差矩阵热力图（原始 / 分红调整后）
     - 各合约对历史走势图
     - Z-Score 统计与偏离信号
     - 分红影响分解
  4. 支持实时数据更新接口（预留 API 接入点）

用法:
  python spread_matrix_viewer.py --products IF IH IC IM \
      --data-dir data/ --report reports/spread_matrix.html

  # 仅查看最新快照
  python spread_matrix_viewer.py --snapshot

  # 指定日期范围
  python spread_matrix_viewer.py --start 2026-01-01 --end 2026-04-25
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ============================================================
# 常量
# ============================================================
DATA_DIR = Path(__file__).parent / "data"
REPORT_DIR = Path(__file__).parent / "reports"
CONFIG_PATH = Path(__file__).parent / "config.yaml"

PRODUCT_INFO = {
    "IF": {"name": "沪深300", "multiplier": 300, "color": "#e74c3c"},
    "IH": {"name": "上证50", "multiplier": 300, "color": "#3498db"},
    "IC": {"name": "中证500", "multiplier": 200, "color": "#2ecc71"},
    "IM": {"name": "中证1000", "multiplier": 200, "color": "#9b59b6"},
}

# 热力图颜色映射：负值(贴水)=绿, 正值(升水)=红
def heatmap_color(val: float, vmin: float, vmax: float) -> str:
    """返回热力图颜色 (CSS rgba)"""
    if vmax == vmin:
        return "rgba(128,128,128,0.85)"
    t = (val - vmin) / (vmax - vmin)  # 0~1
    if t < 0.5:
        # 绿 -> 白
        r = int(255 * (t * 2))
        g = int(180 + 75 * (t * 2))
        b = int(150 + 105 * (t * 2))
    else:
        # 白 -> 红
        r = int(255)
        g = int(255 - 175 * ((t - 0.5) * 2))
        b = int(255 - 205 * ((t - 0.5) * 2))
    return f"rgba({r},{g},{b},0.85)"


def z_color(val: float) -> str:
    """Z-Score 颜色：正=红(高估), 负=绿(低估), 中=白"""
    if abs(val) <= 0.5:
        return "rgba(240,240,240,0.9)"
    elif val > 0:
        intensity = min(abs(val) / 3.0, 1.0)
        return f"rgba(231,76,60,{0.4 + intensity*0.5})"
    else:
        intensity = min(abs(val) / 3.0, 1.0)
        return f"rgba(46,204,113,{0.4 + intensity*0.5})"


# ============================================================
# 工具函数
# ============================================================

def get_exp_date(symbol: str) -> date:
    """根据合约代码推算交割日（当月第3个周五）"""
    y = int("20" + symbol[2:4])
    m = int(symbol[4:6])
    d = date(y, m, 1)
    # 当月第3个周五: 找到第1个周五, 再+14天
    first_friday_offset = (4 - d.weekday()) % 7
    third_friday = 1 + first_friday_offset + 14
    # 不超过当月最后一天
    return d.replace(day=min(third_friday, 28))


# 股指期货季月: 3, 6, 9, 12
QUARTERLY_MONTHS = {3, 6, 9, 12}


def identify_main_contracts(contracts: List[str],
                            reference_date: Optional[date] = None) -> List[str]:
    """
    从合约列表中识别四主力合约：C1(当月), C2(下月), C3(当季), C4(远季)。

    规则（中金所股指期货）：
      - 股指期货只有季月合约(3/6/9/12月) + 下月邻月
      - 例如当前4月，则主力为: 5月(C1/C2), 6月(C3), 9月(C4)
      - 如果当月==下月（如6月=季月），去重后取最近的4个不同合约
      - 核心逻辑：按交割日排序，取未来最近+未过期的4个

    Returns:
      [C1, C2, C3, C4] 合约代码列表，不足则返回实际能识别的
    """
    if not contracts:
        return []

    ref = reference_date or date.today()

    # 计算每个合约的交割日和距今天数
    contract_info = []
    for c in contracts:
        try:
            exp = get_exp_date(c)
            days_to_exp = (exp - ref).days
            m = int(c[4:6])
            is_quarterly = m in QUARTERLY_MONTHS
            contract_info.append({
                "symbol": c,
                "exp": exp,
                "days": days_to_exp,
                "month": m,
                "is_quarterly": is_quarterly,
            })
        except Exception:
            continue

    if not contract_info:
        return contracts[:4]

    # 按交割日升序排列
    contract_info.sort(key=lambda x: (x["days"], x["symbol"]))

    # 过滤已过期合约 (days < -5，给一点容差)
    active = [c for c in contract_info if c["days"] >= -5]
    if not active:
        active = contract_info[:4]

    # 股指期货策略：直接取交割日最近的4个未来合约
    future = [c for c in active if c["days"] >= 0]
    future.sort(key=lambda x: (x["days"], x["month"]))

    # 如果未来合约不足4个，用已过期但最接近的补
    result_symbols = [c["symbol"] for c in future[:4]]
    if len(result_symbols) < 4:
        remaining = [c for c in active
                     if c["symbol"] not in result_symbols]
        remaining.sort(key=lambda x: abs(x["days"]))
        for c in remaining:
            if len(result_symbols) >= 4:
                break
            result_symbols.append(c["symbol"])

    # 打印标签
    labels = ['C1', 'C2', 'C3', 'C4']
    label_str = ', '.join(
        f'{labels[i]}={result_symbols[i]}'
        for i in range(len(result_symbols))
    )
    print(f"  [四主力合约] {label_str}")
    return result_symbols


def load_dividend_config(config_path: Path) -> Dict[str, Dict[date, float]]:
    """
    从 config.yaml 加载各品种的分红预测表。
    
    Returns:
      {品种: {交割日: 分红点数}}
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    result: Dict[str, Dict[date, float]] = {}
    instruments = config.get("instruments", {})
    
    for product, cfg in instruments.items():
        schedule = cfg.get("dividend_schedule", {})
        if not schedule or not cfg.get("enabled"):
            continue
        
        div_table: Dict[date, float] = {}
        for date_str, points in schedule.items():
            exp_date = date.fromisoformat(date_str)
            div_table[exp_date] = float(points)
        
        if div_table:
            result[product] = div_table
    
    return result


def find_dividend(div_table: Dict[date, float],
                   target_date: date,
                   tolerance_days: int = 30) -> float:
    """
    模糊匹配分红预测（容差±tolerance_days天）。
    
    券商研报的交割日期与代码推算的交割日常差1~2天，
    只要偏差在30天内就采用最近的预测值。
    """
    if not div_table:
        return 0.0
    
    if target_date in div_table:
        return div_table[target_date]
    
    best_date = None
    best_delta = 999
    for d in div_table:
        delta = abs((d - target_date).days)
        if delta <= tolerance_days and delta < best_delta:
            best_date = d
            best_delta = delta
    
    if best_date is not None:
        pts = div_table[best_date]
        print(f"  [分红] 目标{target_date} -> 匹配{best_date} "
              f"(差{best_delta}天, {pts:.2f}pt)")
        return pts
    
    print(f"  [分红] 缺少{target_date}附近(±{tolerance_days}天)的分红预测")
    return 0.0


def recalc_dividend_adjusted(df: pd.DataFrame,
                              product: str,
                              div_table: Dict[date, float]) -> pd.DataFrame:
    """
    对 DataFrame 重新计算分红调整后的年化基差率。
    
    原始 CSV 中 dividend_between 可能全为0（旧数据未计算），
    此函数从 config.yaml 加载分红预测并重新计算。
    
    计算公式（同 basis_calculator）：
      理论价差 = 近月价 × (e^(r×ΔT) - 1) - 分红PV
      调整后基差 = 实际价差 - 理论价差
      年化调整基差率 = 调整后基差 / 近月价 / ΔT × 100%
    """
    if df.empty:
        return df
    
    # 向量化重新计算
    results: List[pd.DataFrame] = []
    
    for (near_sym, far_sym), group in df.groupby(["near_symbol", "far_symbol"]):
        # 直接从合约代码推算交割日
        try:
            far_exp = get_exp_date(far_sym)
            near_exp = get_exp_date(near_sym)
        except Exception:
            continue
        
        # 查找分红
        div_pts = find_dividend(div_table, far_exp) if div_table else 0.0
        
        today = datetime.now().date()
        
        days_near = max((near_exp - today).days, 1)
        days_far = max((far_exp - today).days, 1)
        dt_near = days_near / 365.0
        dt_far = days_far / 365.0
        dt_diff = dt_far - dt_near
        
        if dt_diff <= 0:
            results.append(group)
            continue
        
        near_c = group["near_close"].astype(float)
        far_c = group["far_close"].astype(float)
        
        raw_basis = far_c - near_c
        raw_annual = (raw_basis / near_c / dt_diff) * 100.0
        
        # 分红调整 (r=2%)
        t_mid = (dt_near + dt_far) / 2.0
        r = 0.02
        dividend_pv = div_pts * math.exp(-r * t_mid)
        theoretical_spread = near_c * (math.exp(r * dt_diff) - 1.0) - dividend_pv
        adj_basis = raw_basis - theoretical_spread
        adj_annual = (adj_basis / near_c / dt_diff) * 100.0
        
        g = group.copy()
        g["dividend_between"] = div_pts
        g["adj_annualized_rate"] = adj_annual.values
        g["raw_annualized_rate"] = raw_annual.values
        g["raw_basis"] = raw_basis.values
        
        # 打印摘要
        last_adj = adj_annual.iloc[-1] if len(adj_annual) > 0 else 0
        last_raw = raw_annual.iloc[-1] if len(raw_annual) > 0 else 0
        print(f"  [{near_sym}/{far_sym}] 分红={div_pts:.1f}pt, "
              f"原始年化={last_raw:.2f}%, 调整后={last_adj:.2f}%")
        
        results.append(g)
    
    if results:
        combined = pd.concat(results, ignore_index=True)
        combined = combined.sort_values("datetime").reset_index(drop=True)
        return combined
    
    return df


# ============================================================
# 数据加载器
# ============================================================
class SpreadMatrixDataLoader:
    """
    加载 ccfx 5分钟基差数据，构建价差矩阵。
    
    核心逻辑：
      - 每个 time slice，收集所有活跃合约对的价差数据
      - 构建 N×N 矩阵（上三角为实际价差）
      - 分红剔除：从 config.yaml 读取分红预测，重新计算 adj_annualized_rate
    """

    def __init__(self, data_dir: Path = DATA_DIR,
                 config_path: Path = CONFIG_PATH):
        self.data_dir = data_dir
        self._cache: Dict[str, pd.DataFrame] = {}
        # 加载分红配置
        self.dividend_config: Dict[str, Dict[date, float]] = \
            load_dividend_config(config_path)
        if self.dividend_config:
            print(f"[分红] 已加载 {len(self.dividend_config)} 个品种的分红预测")
            for p, tbl in self.dividend_config.items():
                print(f"  {p}: {[f'{d}={v:.0f}pt' for d,v in tbl.items()]}")

    def load_product(self, product: str,
                     start: Optional[str] = None,
                     end: Optional[str] = None) -> pd.DataFrame:
        """加载单个品种的所有合约对数据，并重新计算分红调整"""
        cache_key = f"{product}_{start}_{end}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        pattern = f"5min_basis_{product}_ccfx_*.csv"
        files = sorted(self.data_dir.glob(pattern))
        if not files:
            print(f"[WARN] 未找到 {product} 的 ccfx 数据文件")
            return pd.DataFrame()

        dfs = []
        for f in files:
            df = pd.read_csv(f, parse_dates=["datetime"])
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values("datetime").reset_index(drop=True)

        if start:
            df = df[df["datetime"] >= start]
        if end:
            df = df[df["datetime"] <= end]

        # 重新计算分红调整（原始CSV中dividend_between可能为0）
        div_table = self.dividend_config.get(product, {})
        if div_table and "adj_annualized_rate" in df.columns:
            old_adj_mean = df["adj_annualized_rate"].mean()
            df = recalc_dividend_adjusted(df, product, div_table)
            new_adj_mean = df["adj_annualized_rate"].mean()
            div_col_mean = df["dividend_between"].mean()
            print(f"\n  [分红重算] {product}: "
                  f"旧均值={old_adj_mean:.2f}% -> "
                  f"新均值={new_adj_mean:.2f}% (分红均值={div_col_mean:.1f}pt)")

        self._cache[cache_key] = df
        print(f"\n[加载] {product}: {len(df)} 行, "
              f"{df['datetime'].min()} ~ {df['datetime'].max()}, "
              f"{len(df.groupby(['near_symbol','far_symbol']))} 个合约对")
        return df

    def get_active_contracts(self, df: pd.DataFrame,
                             timestamp: pd.Timestamp) -> List[str]:
        """获取指定时刻所有活跃合约"""
        mask = df["datetime"] == timestamp
        contracts = set()
        for _, row in df[mask].iterrows():
            contracts.add(row["near_symbol"])
            contracts.add(row["far_symbol"])
        return sorted(contracts)

    def build_matrix_snapshot(self, df: pd.DataFrame,
                              timestamp: pd.Timestamp) -> dict:
        """
        在指定时刻构建价差矩阵快照。

        Returns:
          {
            "timestamp": "...",
            "contracts": ["IF2509","IF2512",...],
            "matrix_raw": [[0, -33.4, ...], ...],       # 原始价差(点数)
            "matrix_adj_rate": [[0, -3.30, ...], ...],   # 分红调整后年化(%)
            "matrix_raw_rate": [[0, -3.30, ...], ...],   # 原始年化率(%)
            "dividend": {("IF2509","IF2512"): 0.0, ...},
            "pair_details": [...],
            "contract_overview": [...]   # 每个单合约的信息
          }
        """
        snap = df[df["datetime"] == timestamp].copy()

        if snap.empty:
            return {}

        contracts = sorted(set(list(snap["near_symbol"].unique()) +
                               list(snap["far_symbol"].unique())))
        n = len(contracts)
        contract_idx = {c: i for i, c in enumerate(contracts)}

        matrix_raw = np.zeros((n, n))         # 原始价差(点数)
        matrix_adj = np.zeros((n, n))          # 分红调整后年化率(%)
        matrix_raw_rate = np.zeros((n, n))     # 原始年化率(%)
        dividend_map = {}                      # 合约对->分红(点)

        pair_details = []

        # ---- 收集每个单合约的价格信息 ----
        contract_prices: Dict[str, float] = {}  # symbol -> 最新价

        for _, row in snap.iterrows():
            near = row["near_symbol"]
            far = row["far_symbol"]
            ci = contract_idx[near]
            fi = contract_idx[far]

            raw_spread = row["raw_spread"]
            adj_rate = row["adj_annualized_rate"]
            raw_rate = row["raw_annualized_rate"]
            div = row.get("dividend_between", 0.0)

            # 记录合约价格（取最后一条即可，同一时刻同合约价格一致）
            contract_prices[near] = float(row["near_close"])
            contract_prices[far] = float(row["far_close"])

            # 上三角：far - near (远月减近月)
            matrix_raw[ci, fi] = raw_spread
            matrix_raw[fi, ci] = -raw_spread   # 下三角取反
            matrix_adj[ci, fi] = adj_rate
            matrix_adj[fi, ci] = -adj_rate
            matrix_raw_rate[ci, fi] = raw_rate
            matrix_raw_rate[fi, ci] = -raw_rate
            dividend_map[(near, far)] = float(div)

            pair_details.append({
                "pair": f"{near}/{far}",
                "near": near,
                "far": far,
                # 价格信息
                "near_close": round(float(row["near_close"]), 1),
                "far_close": round(float(row["far_close"]), 1),
                # 原始价差（点数）
                "raw_spread": round(float(raw_spread), 2),
                "raw_spread_pt": round(float(raw_spread), 2),
                # 原始年化基差率 (%)
                "raw_annualized": round(float(raw_rate), 2),
                # 分红影响
                "dividend": round(float(div), 2),
                "dividend_pt": round(float(div), 2),
                # 调整后年化基差率 (%)
                "adj_annualized": round(float(adj_rate), 2),
            })

        # ---- 构建单合约概览 ----
        contract_overview = []
        for c in contracts:
            price = contract_prices.get(c, 0)
            # 找到以该合约为 near 的所有对，汇总信息
            near_pairs = [p for p in pair_details if p["near"] == c]
            far_pairs = [p for p in pair_details if p["far"] == c]

            # 取最近月的基差信息（第一个 near_pair 通常是最临近的）
            if near_pairs:
                primary_pair = near_pairs[0]
                raw_basis = primary_pair["raw_spread"]
                raw_ann = primary_pair["raw_annualized"]
                dividend = primary_pair["dividend"]
                adj_ann = primary_pair["adj_annualized"]
            else:
                raw_basis = 0
                raw_ann = 0
                dividend = 0
                adj_ann = 0

            contract_overview.append({
                "symbol": c,
                "price": round(price, 1),
                "raw_spread": round(raw_basis, 2),
                "raw_annualized": round(raw_ann, 2),
                "dividend": round(dividend, 2),
                "adj_annualized": round(adj_ann, 2),
            })

        return {
            "timestamp": str(timestamp),
            "contracts": contracts,
            # 注意: main_contracts 由调用方注入，不在此处计算
            # (因为快照可能只含部分合约，需要全量数据来识别)
            "matrix_raw": matrix_raw.tolist(),
            "matrix_adj": matrix_adj.tolist(),
            "matrix_raw_rate": matrix_raw_rate.tolist(),
            "dividend_map": {f"{k[0]}|{k[1]}": v
                             for k, v in dividend_map.items()},
            "pair_details": pair_details,
            "contract_overview": contract_overview,
        }

    def compute_pair_statistics(self, df: pd.DataFrame,
                                window: int = 288) -> List[dict]:
        """
        计算各合约对的历史统计量（用于 Z-score）。
        
        Returns:
          List of {
            "pair", "mean", "std", "current", "zscore",
            "pct_25", "median", "pct_75", "min", "max", "count"
          }
        """
        results = []
        grouped = df.groupby(["near_symbol", "far_symbol"])

        for (near, far), group in grouped:
            adj_rates = group["adj_annualized_rate"].values
            raw_spreads = group["raw_spread"].values

            if len(adj_rates) < window:
                continue

            # rolling z-score (用最近 window 根K线统计)
            recent = adj_rates[-window:]
            mean_val = float(np.mean(recent))
            std_val = float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.01
            current = float(adj_rates[-1])

            if std_val < 1e-6:
                std_val = 0.01

            zscore = (current - mean_val) / std_val

            results.append({
                "pair": f"{near}/{far}",
                "near": near,
                "far": far,
                "mean": round(mean_val, 2),
                "std": round(std_val, 2),
                "current": round(current, 2),
                "zscore": round(zscore, 2),
                "pct_25": round(float(np.percentile(adj_rates, 25)), 2),
                "median": round(float(np.median(adj_rates)), 2),
                "pct_75": round(float(np.percentile(adj_rates, 75)), 2),
                "min_val": round(float(np.min(adj_rates)), 2),
                "max_val": round(float(np.max(adj_rates)), 2),
                "count": len(adj_rates),
                "current_raw": round(float(raw_spreads[-1]), 2),
            })

        # 按 |zscore| 排序
        results.sort(key=lambda x: abs(x["zscore"]), reverse=True)
        return results

    def get_time_series(self, df: pd.DataFrame,
                        near: str, far: str) -> List[dict]:
        """获取指定合约对的时间序列（降采样到每小时以减少数据量）"""
        mask = (df["near_symbol"] == near) & (df["far_symbol"] == far)
        pair_df = df[mask].copy()
        if pair_df.empty:
            return []

        # 降采样：每12根5min线取最后1根（约1小时）
        pair_df = pair_df.iloc[::12].reset_index(drop=True)

        rows = []
        for _, row in pair_df.iterrows():
            rows.append({
                "time": str(row["datetime"]),
                "raw": round(float(row["raw_spread"]), 2),
                "adj_rate": round(float(row["adj_annualized_rate"]), 2),
                "raw_rate": round(float(row["raw_annualized_rate"]), 2),
                "dividend": round(float(row.get("dividend_between", 0)), 2),
                "near_price": round(float(row["near_close"]), 1),
                "far_price": round(float(row["far_close"]), 1),
            })
        return rows


# ============================================================
# HTML 报告生成器
# ============================================================
class SpreadMatrixReportGenerator:
    """生成交互式 HTML 报告"""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(self, all_data: Dict[str, dict]) -> Path:
        """
        all_data: {product: {
            "snapshot": {...},
            "statistics": [...],
            "time_series": {("near","far"): [...]},
            "latest_time": "...",
            "contract_count": N
        }}
        """
        # 将所有产品数据序列化为 JSON
        json_data = json.dumps(all_data, ensure_ascii=False, default=str)

        html = self._build_html(json_data, all_data)
        self.output_path.write_text(html, encoding="utf-8")
        return self.output_path

    def _build_html(self, json_data: str,
                    all_data: Dict[str, dict]) -> str:
        products = list(all_data.keys())
        first_product = products[0] if products else ""

        # Tab 按钮（纯 Python 字符串拼接，不混 JS 模板）
        tab_buttons = []
        for i, p in enumerate(products):
            name = PRODUCT_INFO[p]["name"]
            active_cls = ' active' if i == 0 else ''
            btn = (f'<button class="tab-btn{active_cls}" '
                   f'onclick="switchTab(this,\'{p}\')" '
                   f'data-tab="{p}">{name}({p})</button>')
            tab_buttons.append(btn)
        product_tabs_html = "\n  ".join(tab_buttons)

        # 配置注入
        app_config = json.dumps({"firstProduct": first_product},
                                ensure_ascii=False)

        # 读取 JS 文件并内联到 HTML 中（解决 reports/ 目录下相对路径问题）
        js_path = Path(__file__).parent / "spread_matrix.js"
        js_code = js_path.read_text(encoding="utf-8")

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>股指期货价差矩阵监控</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0d1117;
    --card-bg: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --green: #3fb950;
    --red: #f85149;
    --blue: #58a6ff;
    --yellow: #d29922;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: -apple-system, 'Microsoft YaHei', sans-serif;
    background: var(--bg); color: var(--text);
    padding: 16px; min-height: 100vh;
  }}
  .header {{
    text-align:center; padding: 20px 0 16px;
    border-bottom: 1px solid var(--border); margin-bottom: 16px;
  }}
  .header h1 {{ font-size: 22px; font-weight:600; }}
  .header .subtitle {{ color:var(--text-muted); font-size:13px; margin-top:4px; }}
  .update-info {{ display:inline-flex; align-items:center; gap:6px;
                  margin-top:8px; color:var(--text-muted); font-size:12px; }}
  .update-dot {{ width:8px; height:8px; border-radius:50%; background:var(--green);
                 animation:pulse 2s infinite; }}
  @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.4}} }}

  /* Tabs */
  .tabs {{ display:flex; gap:4px; margin-bottom:16px; flex-wrap:wrap; }}
  .tab-btn {{
    padding:8px 18px; border:none; border-radius:8px; cursor:pointer;
    background:var(--card-bg); color:var(--text-muted); font-size:14px;
    font-weight:500; transition:all 0.2s; border:1px solid var(--border);
  }}
  .tab-btn.active {{ background:#1f2937; color:var(--text);
                     border-color:var(--blue); }}
  .tab-btn:hover:not(.active) {{ background:#21262d; color:var(--text); }}

  /* Grid */
  .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .grid-3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }}
  @media(max-width:1200px){{ .grid-2,.grid-3 {{ grid-template-columns:1fr; }} }}

  .card {{
    background:var(--card-bg); border:1px solid var(--border);
    border-radius:10px; padding:16px; margin-bottom:16px;
  }}
  .card-title {{
    font-size:15px; font-weight:600; margin-bottom:12px;
    display:flex; align-items:center; gap:8px;
  }}
  .card-title .icon {{ font-size:16px; }}

  /* Matrix Table */
  .matrix-wrapper {{ overflow-x:auto; }}
  table.matrix {{
    width:100%; border-collapse:collapse; font-size:13px;
  }}
  table.matrix th {{
    background:#21262d; padding:8px 12px; text-align:center;
    font-weight:500; color:var(--text-muted); border:1px solid var(--border);
    white-space:nowrap; font-size:12px;
  }}
  table.matrix td {{
    padding:10px 14px; text-align:center; border:1px solid var(--border);
    font-weight:600; font-family:'Consolas','Courier New',monospace;
    transition:background 0.15s; cursor:default; position:relative;
  }}
  table.matrix td:hover {{ filter:brightness(1.3); transform:scale(1.05);
                           z-index:2; box-shadow:0 0 12px rgba(0,0,0,0.5); }}
  table.matrix td.diagonal {{ background:#21262d; color:var(--text-muted); }}
  .matrix-label {{ font-weight:700; color:var(--blue); }}

  /* Stats */
  .stats-row {{
    display:flex; justify-content:space-between; align-items:center;
    padding:10px 14px; border-bottom:1px solid #21262d; font-size:13px;
  }}
  .stats-row:last-child {{ border-bottom:none; }}
  .stats-pair {{ font-weight:600; color:var(--blue); min-width:140px; }}
  .stats-z {{
    font-family:'Consolas',monospace; font-weight:700; font-size:14px;
    padding:2px 10px; border-radius:6px;
  }}
  .stats-val {{ color:var(--text-muted); font-family:'Consolas',monospace;
               font-size:12px; }}
  .signal-badge {{
    display:inline-block; padding:2px 8px; border-radius:4px;
    font-size:11px; font-weight:600; margin-left:8px;
  }}
  .signal-long {{ background:rgba(63,185,80,0.15); color:var(--green); }}
  .signal-short {{ background:rgba(248,81,73,0.15); color:var(--red); }}
  .signal-neutral {{ background:rgba(139,148,158,0.1); color:var(--text-muted); }}

  /* Chart */
  .chart-container {{ position:relative; height:320px; width:100%; }}

  /* Legend */
  .legend {{ display:flex; gap:20px; flex-wrap:wrap; font-size:12px;
             margin-top:10px; color:var(--text-muted); }}
  .legend-item {{ display:flex; align-items:center; gap:4px; }}
  .legend-color {{ width:14px; height:14px; border-radius:3px; }}

  /* Dividend info */
  .div-card {{
    background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
    border:1px solid #30363d; border-radius:8px; padding:12px; margin:6px 0;
  }}
  .div-header {{ display:flex; justify-content:space-between; font-size:13px;
                 margin-bottom:6px; }}
  .div-pair {{ font-weight:600; color:var(--blue); }}
  .div-amount {{ font-weight:700; font-size:16px; color:var(--yellow);
                 font-family:'Consolas',monospace; }}
  .div-detail {{ font-size:11px; color:var(--text-muted); }}

  /* Toggle switch */
  .toggle-group {{ display:flex; gap:0; margin-bottom:12px; }}
  .toggle-btn {{
    padding:6px 14px; border:1px solid var(--border); cursor:pointer;
    font-size:12px; background:transparent; color:var(--text-muted);
    transition:all 0.2s;
  }}
  .toggle-btn:first-child {{ border-radius:6px 0 0 6px; }}
  .toggle-btn:last-child {{ border-radius:0 6px 6px 0; }}
  .toggle-btn.active {{ background:var(--blue); color:#fff;
                         border-color:var(--blue); }}

  .panel-content {{ display:none; }}
  .panel-content.active {{ display:block; }}

  .footer {{ text-align:center; color:var(--text-muted);
             font-size:11px; padding:20px 0; border-top:1px solid var(--border);
             margin-top:20px; }}
</style>
</head>
<body>

<div class="header">
  <h1>📊 股指期货价差矩阵监控</h1>
  <div class="subtitle">四合约跨期价差 · 分红调整后年化基差率 · Z-Score偏离信号</div>
  <div class="update-info">
    <span class="update-dot"></span>
    <span id="updateTime">数据加载中...</span>
    <span style="margin-left:12px;color:var(--text-muted)">
      | 切换品种查看各合约间价差关系 |
    </span>
  </div>
</div>

<div class="tabs" id="productTabs">
  {product_tabs_html}
</div>

<!-- ========== 主面板 ========== -->
<div id="mainContent"></div>

<!-- ========== 数据注入（由 Python 生成） ========== -->
<script id="appData" type="application/json">
{json_data}
</script>

<script id="appConfig" type="application/json">
{app_config}
</script>

<!-- ========== 前端逻辑（内联 JS，避免路径问题） ========== -->
<script>
{js_code}
</script>

<div class="footer">
  股指期货价差矩阵监控工具 · 分红调整基于 config.yaml 分红预测表 ·
  数据来源: ccfx 5分钟行情 · 仅供研究参考
</div>

</body>
</html>"""
        return html


# ============================================================
# Main Entry
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="股指期货价差矩阵可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--products", nargs="+", default=["IF", "IH", "IC", "IM"],
        help="品种列表 (默认: IF IH IC IM)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(DATA_DIR),
        help="数据目录路径",
    )
    parser.add_argument(
        "--report", type=str, default=str(REPORT_DIR / "spread_matrix.html"),
        help="输出 HTML 报告路径",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="起始日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="截止日期 (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    loader = SpreadMatrixDataLoader(data_dir)
    report_gen = ReportGenerator(Path(args.report))

    all_product_data = {}

    for product in args.products:
        print(f"\n{'='*60}")
        print(f"处理品种: {product} ({PRODUCT_INFO[product]['name']})")
        print(f"{'='*60}")

        # 1. 加载数据
        df = loader.load_product(product, start=args.start, end=args.end)
        if df.empty:
            print(f"[SKIP] {product} 无数据")
            continue

        # 2. 从全量数据识别四主力合约（基于所有历史出现的合约）
        all_historical_contracts = sorted(
            set(df["near_symbol"].unique().tolist() +
                df["far_symbol"].unique().tolist())
        )
        main_contracts = identify_main_contracts(all_historical_contracts)

        # 3. 最新矩阵快照
        latest_time = df["datetime"].max()
        snapshot = loader.build_matrix_snapshot(df, latest_time)
        print(f"\n[快照] {latest_time}")
        print(f"  活跃合约: {snapshot.get('contracts', [])}")
        # 把主力合约信息也注入快照
        snapshot["main_contracts"] = main_contracts

        # 3. 各合约对统计
        statistics = loader.compute_pair_statistics(df, window=288)
        if statistics:
            top3 = statistics[:3]
            print(f"\n[TOP3偏离]")
            for s in top3:
                sig = ""
                if s["zscore"] > 1.5:
                    sig = " [高估]"
                elif s["zscore"] < -1.5:
                    sig = " [低估]"
                print(f"  {s['pair']:20s}  Z={s['zscore']:+.2f}  "
                      f"cur={s['current']:.2f}%  mu={s['mean']:.2f}%  "
                      f"sigma={s['std']:.2f}{sig}")

        # 4. 所有合约对时间序列
        ts_map = {}
        for (near, far), group in df.groupby(["near_symbol", "far_symbol"]):
            key = f"{near}|{far}"
            ts_map[key] = loader.get_time_series(df, near, far)

        all_product_data[product] = {
            "snapshot": snapshot,
            "statistics": statistics,
            "time_series": ts_map,
            "latest_time": str(latest_time),
            "contract_count": len(snapshot.get("contracts", [])),
        }

    # 5. 生成 HTML 报告
    if not all_product_data:
        print("[ERROR] 无有效数据，无法生成报告")
        sys.exit(1)

    report_path = report_gen.generate(all_product_data)
    print(f"\n{'='*60}")
    print(f"[OK] report generated: {report_path}")
    print(f"{'='*60}")


class ReportGenerator(SpreadMatrixReportGenerator):
    pass


if __name__ == "__main__":
    main()
