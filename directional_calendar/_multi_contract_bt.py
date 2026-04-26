"""
multi_contract_backtest.py
多合约跨期价差策略回测引擎

核心改进：
  - 不再只看 near/far 两个合约，而是同时跟踪每个品种的全部活跃合约
  - 每根5min bar计算所有 C(4,2)=6 个合约对的年化基差率
  - 策略自动选择最优合约对持仓：
    * 做多方向：选择年化基差最高的对（远月相对最贵 → 升水最大）
    * 做空方向：选择年化基差最低的对（远月相对最便宜 → 贴水最深）
  - 支持同一品种内的任意两合约之间切换（不仅是近/远）

数据源：CCFX tick数据（已解压处理后的CSV）
"""

from __future__ import annotations

import os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import zipfile
import io
import warnings
from datetime import date, datetime, timedelta, time as dtime

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import yaml
from basis_calculator import BasisCalculator

warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================

BASE = r'D:\BaiduNetdiskDownload\ccfx'
OUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
REPORT_DIR = os.path.join(os.path.dirname(__file__), 'reports')

PRODUCTS = ['IF', 'IH', 'IC', 'IM']

MULTIPLIER = {"IF": 300, "IH": 300, "IC": 200, "IM": 200}

# 合约到期日计算（第三周五）
def get_expiry(symbol):
    try:
        y = int("20" + symbol[2:4])
        m = int(symbol[4:6])
        d = date(y, m, 1)
        ff = (4 - d.weekday()) % 7
        return d.replace(day=min(1 + ff + 14, 28))
    except:
        return date.today() + timedelta(days=55)

# ============================================================
# Phase 1: 从CCFX zip读取tick数据，生成每品种4合约的5min K线
# ============================================================

def load_ccfx_day(day_dir):
    """从一天的zip文件中读取并返回原始DataFrame"""
    zfiles = [f for f in os.listdir(day_dir) if f.endswith('.zip')]
    if not zfiles:
        return None
    zf = zipfile.ZipFile(os.path.join(day_dir, zfiles[0]))
    cn = zf.namelist()[0]
    df = pd.read_csv(io.BytesIO(zf.read(cn)))
    zf.close()
    return df


def filter_trading_time(df):
    """过滤到交易时间"""
    df['_dt'] = pd.to_datetime(df['ActionDay'].astype(str) + ' ' + df['UpdateTime'].astype(str))
    t = df['_dt'].dt.time
    am = (t >= dtime(9, 15)) & (t <= dtime(11, 30))
    pm = (t >= dtime(13, 0)) & (t <= dtime(15, 0))
    return df[am | pm].copy()


def resample_5min(df_group):
    """将单合约tick数据resample为5min OHLC"""
    df = df_group.sort_values('_dt').set_index('_dt')
    ohlc = df['LastPrice'].resample('5min').ohlc()
    vol = df['Volume'].resample('5min').last()  # 累计成交量
    
    result = pd.DataFrame({
        'open': ohlc['open'], 'high': ohlc['high'],
        'low': ohlc['low'], 'close': ohlc['close'],
        'volume': vol,
    })
    result = result.dropna(subset=['close'])
    out = result.reset_index()
    # rename index column to datetime
    if out.columns[0] != 'datetime':
        out.rename(columns={out.columns[0]: 'datetime'}, inplace=True)
    return out


def process_all_days():
    """遍历所有天，生成每品种每合约的5min K线字典"""
    print("=" * 70)
    print("  Phase 1: CCFX Tick -> 5min OHLC (全合约)")
    print("=" * 70)

    all_days = sorted([d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d))])

    # 存储结构: {product: {symbol: DataFrame}}
    contract_bars = {p: {} for p in PRODUCTS}
    total_days = 0
    skipped = 0

    for day_dir_name in all_days:
        day_dir = os.path.join(BASE, day_dir_name)
        df_raw = load_ccfx_day(day_dir)
        if df_raw is None or len(df_raw) < 1000:
            skipped += 1
            continue
        
        total_days += 1
        df = filter_trading_time(df_raw)

        for prod in PRODUCTS:
            for sym in sorted(df[df['InstruID'].str.startswith(prod)]['InstruID'].unique()):
                sub_df = df[df['InstruID'] == sym].copy()
                if len(sub_df) < 100:
                    continue
                bars = resample_5min(sub_df)
                if sym not in contract_bars[prod]:
                    contract_bars[prod][sym] = []
                contract_bars[prod][sym].append(bars)

        if total_days % 30 == 0:
            print(f"  已处理 {total_days} 天... ({day_dir_name})")

    # 合并每天的数据
    print(f"\n  处理完毕: {total_days} 天有效, {skipped} 天跳过")
    
    merged = {}
    for prod in PRODUCTS:
        merged[prod] = {}
        for sym, bar_list in contract_bars[prod].items():
            if bar_list:
                full_df = pd.concat(bar_list, ignore_index=True)
                full_df = full_df.sort_values('datetime').reset_index(drop=True)
                # 去重（同一时间点保留最后一条）
                full_df.drop_duplicates(subset=['datetime'], keep='last', inplace=True)
                merged[prod][sym] = full_df
                print(f"    {prod}/{sym}: {len(full_df)} bars, "
                      f"{full_df['datetime'].iloc[0]} ~ {full_df['datetime'].iloc[-1]}")

    return merged


# ============================================================
# Phase 2: 计算多合约基差矩阵
# ============================================================

class MultiContractBasisCalculator:
    """
    多合约基差计算器
    
    对每个品种的4个合约，每根bar计算所有 C(4,2)=6 个合约对的：
      - 价差 (far_price - near_price)
      - 年化基差率 raw_annualized (无分红调整)
      - 分红调整后年化基差率 adj_annualized_rate (R=0)
    """
    
    def __init__(self):
        self.calc = BasisCalculator(risk_free_rate=0.0)
        self.div_schedule = {}
        
        cfg_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            for prod in PRODUCTS:
                if prod in cfg.get('instruments', {}):
                    ds = cfg['instruments'][prod].get('dividend_schedule', {})
                    if ds:
                        self.div_schedule[prod] = ds
                        self.calc.load_dividend_schedule(prod, ds, 'config')
    
    def compute_basis_matrix(self, prod, contracts_data, dt):
        """
        给定时间点 dt 和该品种的所有合约5min bar，
        返回所有合约对的基差信息。
        
        Returns:
            list of dict, 每个 dict 是一个合约对的基差信息
        """
        symbols = sorted(contracts_data.keys())
        pairs_info = []
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym_near = symbols[i]
                sym_far = symbols[j]
                
                dn = contracts_data[sym_near]
                df = contracts_data[sym_far]
                
                # 找到这两根合约在 dt 或最接近 dt 的价格
                # 先尝试前向填充（最近的历史价）
                rn = dn[dn['datetime'] <= dt]
                rf = df[df['datetime'] <= dt]
                
                if len(rn) == 0:
                    # 如果没有历史数据（dt是第一天），用后向填充
                    rn = dn[dn['datetime'] >= dt]
                if len(rf) == 0:
                    rf = df[df['datetime'] >= dt]
                    
                if len(rn) == 0 or len(rf) == 0:
                    continue
                
                near_price = float(rn['close'].iloc[0 if len(dn[dn['datetime'] <= dt]) == 0 else -1])
                far_price = float(rf['close'].iloc[0 if len(df[df['datetime'] <= dt]) == 0 else -1])
                
                if near_price <= 0 or far_price <= 0:
                    continue
                
                spread = far_price - near_price
                
                exp_near = get_expiry(sym_near)
                exp_far = get_expiry(sym_far)
                
                try:
                    as_of = dt.date() if hasattr(dt, 'date') else date.today()
                except:
                    as_of = date.today()
                
                d_near = max((exp_near - as_of).days, 1)
                d_far = max((exp_far - as_of).days, 1)
                t_span = max(d_far - d_near, 30) / 365.0
                
                if t_span <= 0:
                    t_span = 90 / 365.0
                
                # 分红
                div_pts = self.calc.get_dividend_between(prod, exp_near, exp_far)
                
                raw_rate = spread / near_price / t_span * 100.0
                adj_rate = (spread + div_pts) / near_price / t_span * 100.0
                
                pairs_info.append({
                    'near_symbol': sym_near,
                    'far_symbol': sym_far,
                    'near_close': near_price,
                    'far_close': far_price,
                    'spread': spread,
                    'days_near': d_near,
                    'days_far': d_far,
                    't_years': round(t_span, 4),
                    'raw_annualized_rate': round(raw_rate, 4),
                    'dividend_between': round(div_pts, 2),
                    'adj_annualized_rate': round(adj_rate, 4),
                    'product': prod,
                    'datetime': dt,
                })
        
        return pairs_info


# ============================================================
# Phase 3: 多合约回测引擎
# ============================================================

class MultiContractBacktestEngine:
    """
    多合约回测引擎
    
    核心思路：
    - 每根bar，根据当前方向信号和所有可用合约对的基差，
      选择最优的合约对来持有或切换。
    - 做多时：选 adj_annualized_rate 最高（升水最大）的对
    - 做空时：选 adj_annualized_rate 最低（贴水最深）的对
    - 持有期间持续监控，如果有更好的对出现就切换
    """
    
    def __init__(self, 
                 sigma_entry=1.0, sigma_exit=0.3,
                 cooldown_bars=48,  # 4小时冷却(48*5min)
                 rollover_days=5,
                 initial_capital=1_000_000,
                 volume=1,
                 commission_rate=0.000023,
                 slippage_ticks=0.5,
                 min_spread_threshold=0.3,   # 最低年化基差才考虑切换(%)
                 ):
        self.sigma_entry = sigma_entry
        self.sigma_exit = sigma_exit
        self.cooldown_bars = cooldown_bars
        self.rollover_days = rollover_days
        self.initial_capital = initial_capital
        self.volume = volume
        self.commission_rate = commission_rate
        self.slippage_ticks = slippage_ticks
        self.min_spread_threshold = min_spread_threshold
        
        self.trades = []
        self.equity_curve = []
        self.equity_times = []

    def run(self, all_contracts_data, direction_signal_map, product, basis_calc):
        """
        执行单品种多合约回测
        
        Args:
            all_contracts_data: dict {symbol: DataFrame} 该品种所有合约的5min数据
            direction_signal_map: dict {datetime: signal_value} 方向信号
            product: str 品种代码
            basis_calc: MultiContractBasisCalculator 实例
        """
        print(f"\n{'='*70}")
        print(f"  Multi-Contract Backtest: {product}")
        print(f"{'='*70}")
        
        symbols = sorted(all_contracts_data.keys())
        n_symbols = len(symbols)
        n_pairs = n_symbols * (n_symbols - 1) // 2
        print(f"  可用合约({n_symbols}): {' '.join(symbols)}")
        print(f"  可用合约对数: {n_pairs}")
        
        # 获取全局时间轴（用所有合约的时间并集）
        all_times_set = set()
        for df in all_contracts_data.values():
            all_times_set.update(df['datetime'].unique())
        
        common_times = sorted(all_times_set)
        
        if len(common_times) == 0:
            print(f"  无公共时间点，跳过")
            return None
        print(f"  公共时间点: {len(common_times)} bars")
        print(f"  时间范围: {common_times[0]} ~ {common_times[-1]}")
        
        # ---- 优化: 预构建价格查找表（不用每次遍历DataFrame）----
        # price_lookup[symbol] = sorted list of (datetime, close_price)
        # 用二分查找加速价格查询
        print("\n  构建价格查找表...")
        price_lookup = {}
        for sym in symbols:
            df = all_contracts_data[sym]
            # 排序并去重
            sub = df[['datetime', 'close']].drop_duplicates('datetime').sort_values('datetime')
            price_lookup[sym] = (sub['datetime'].values, sub['close'].values)
        
        # 预计算合约到期日
        expiry_map = {sym: get_expiry(sym) for sym in symbols}
        
        # 预生成所有合约对列表
        pair_list = []
        for i in range(n_symbols):
            for j in range(i + 1, n_symbols):
                pair_list.append((symbols[i], symbols[j]))
        
        import bisect
        
        def get_price(sym, dt):
            """用bisect快速查找sym在dt时刻的最近价格，返回(price, is_real)"""
            times_arr, prices_arr = price_lookup[sym]
            idx = bisect.bisect_right(times_arr, dt) - 1
            if idx < 0:
                # dt早于该合约第一个数据点——返回None表示不可用
                return None, False
            if idx >= len(prices_arr):
                idx = len(prices_arr) - 1
            # 检查时间差距是否太大（超过1天 = 数据不存在）
            time_gap = (dt - times_arr[idx]).total_seconds() / 86400
            if time_gap > 1:  # 超过1天没有数据
                return None, False
            return float(prices_arr[idx]), True
        
        def compute_pairs_at(dt):
            """快速计算dt时刻所有合约对的基差"""
            results = []
            as_of_date = dt.date() if hasattr(dt, 'date') else date.today()
            
            for sym_near, sym_far in pair_list:
                near_p, near_ok = get_price(sym_near, dt)
                far_p, far_ok = get_price(sym_far, dt)
                
                # 两个合约都必须有实际数据（不能ffill跨天）
                if not near_ok or not far_ok or near_p <= 0 or far_p <= 0:
                    continue
                
                spread = far_p - near_p
                exp_near = expiry_map[sym_near]
                exp_far = expiry_map[sym_far]
                
                d_near = max((exp_near - as_of_date).days, 1)
                d_far = max((exp_far - as_of_date).days, 1)
                t_span = max(d_far - d_near, 30) / 365.0
                
                div_pts = basis_calc.calc.get_dividend_between(product, exp_near, exp_far)
                
                raw_rate = spread / near_p / t_span * 100.0
                adj_rate = (spread + div_pts) / near_p / t_span * 100.0
                
                results.append({
                    'near_symbol': sym_near,
                    'far_symbol': sym_far,
                    'near_close': near_p,
                    'far_close': far_p,
                    'spread': spread,
                    'days_near': d_near,
                    'days_far': d_far,
                    't_years': round(t_span, 4),
                    'raw_annualized_rate': round(raw_rate, 4),
                    'dividend_between': round(div_pts, 2),
                    'adj_annualized_rate': round(adj_rate, 4),
                    'product': product,
                    'datetime': dt,
                })
            
            return results
        
        print(f"  价格查找表就绪: {len(symbols)} 合约, {len(pair_list)} 对")
        
        # ---- 回测主循环 ----
        equity = self.initial_capital
        state = {
            'holding': None,       # None 或 ('LONG'|'SHORT', near_sym, far_sym, entry_spread, entry_dt)
            'direction': 0,         # 当前方向信号 0/1/-1
            'entry_equity': equity,
            'entry_price_near': 0,
            'entry_price_far': 0,
            'position_volume': 0,
            'last_switch_bar': -9999,
        }
        
        mult = MULTIPLIER.get(product, 300)
        
        for bar_idx, dt in enumerate(common_times):
            # 获取方向信号
            sig = direction_signal_map.get(dt, 0)
            
            # 将当前bar索引注入state（供回调使用）
            state['_current_bar'] = bar_idx
            
            if sig == 0:
                if state['holding'] is not None:
                    pairs = compute_pairs_at(dt)
                    pnl = self._close_position(state, dt, pairs, mult, equity)
                    equity += pnl
                    state['holding'] = None
                    state['direction'] = 0
                self._record_equity(equity, dt)
                continue
            
            direction = 'LONG' if sig > 0 else 'SHORT'
            state['direction'] = sig
            
            in_cooldown = (bar_idx - state['last_switch_bar']) < self.cooldown_bars
            
            # 按需计算基差矩阵
            pairs = compute_pairs_at(dt)
            
            best_pair = self._select_best_pair(pairs, direction)
            
            if best_pair is None:
                self._record_equity(equity, dt)
                continue
            
            # ---- 决策状态机 ----
            
            if state['holding'] is None:
                # 空仓 -> 开仓
                action, new_state = self._decide_open(
                    state, direction, best_pair, dt, pairs, in_cooldown, bar_idx, equity)
                if action == 'OPEN':
                    state.update(new_state)
                    
            elif state['holding'][0] != direction:
                # 方向反转 -> 先平仓再开新仓
                cur_holding = state['holding']
                pnl = self._close_position(state, dt, pairs, mult, equity)
                equity += pnl
                state['holding'] = None
                
                action2, new_state2 = self._decide_open(
                    state, direction, best_pair, dt, pairs, False, bar_idx, equity)
                if action2 == 'OPEN':
                    state.update(new_state2)
                    
            else:
                # 同方向持仓中 -> 检查是否要切换合约对
                action, new_state = self._decide_switch(
                    state, direction, best_pair, dt, pairs, in_cooldown, bar_idx)
                if action == 'SWITCH':
                    # 平旧开新
                    pnl = self._close_position(state, dt, pairs, mult, equity, is_switch=True)
                    equity += pnl
                    state.update(new_state)
                elif action == 'CLOSE':
                    pnl = self._close_position(state, dt, pairs, mult, equity)
                    equity += pnl
                    state['holding'] = None
            
            # 记录权益曲线
            # 如果有仓位，用mark-to-market更新
            if state['holding'] is not None:
                _, hold_near_sym, hold_far_sym, _, _ = state['holding']
                # 找当前持仓对的最新价差
                current_spread = None
                for p in pairs:
                    if p['near_symbol'] == hold_near_sym and p['far_symbol'] == hold_far_sym:
                        current_spread = p['spread']
                        break
                
                if current_spread is not None:
                    entry_spread = state.get('entry_spread', 0)
                    spread_change = current_spread - entry_spread
                    dir_sign = 1 if direction == 'LONG' else -1
                    unrealized = dir_sign * spread_change * mult * self.volume
                    self.equity_curve.append(equity + unrealized)
                    self.equity_times.append(dt)
                else:
                    # 持仓合约对不在当前可用列表中了（可能交割了），强制平仓
                    pnl = self._close_position(state, dt, pairs, mult, equity)
                    equity += pnl
                    state['holding'] = None
                    self.equity_curve.append(equity)
                    self.equity_times.append(dt)
            else:
                self.equity_curve.append(equity)
                self.equity_times.append(dt)
        
        # 最终结果统计
        return self._build_results(product, equity, state)
    
    def _select_best_pair(self, pairs, direction):
        """选择最优合约对"""
        # 过滤1: 近月至少还有rollover_days天才到期
        # 过滤2: 年化跨度至少0.15年(~55天)，排除短跨度的伪信号
        # 过滤3: 绝对年化率 < 500%，排除极端值
        valid_pairs = [p for p in pairs 
                       if p['days_near'] > self.rollover_days
                       and p['t_years'] >= 0.15
                       and abs(p['adj_annualized_rate']) < 500]
        
        if not valid_pairs:
            # 放宽: T>=0.08(~30天)
            valid_pairs = [p for p in pairs 
                           if p['days_near'] > self.rollover_days
                           and p['t_years'] >= 0.08]
        
        if not valid_pairs:
            valid_pairs = [p for p in pairs if p['days_near'] > self.rollover_days]
        
        if not valid_pairs:
            return None
        
        if direction == 'LONG':
            # 做多：选年化基差最高（升水最大/贴水最浅）的对
            # 在贴水市场中，这意味着"亏损最少"的仓位
            best = max(valid_pairs, key=lambda p: p['adj_annualized_rate'])
        else:
            # 做空：选年化基差最低（贴水最深）的对
            best = min(valid_pairs, key=lambda p: p['adj_annualized_rate'])
        
        return best
    
    def _decide_open(self, state, direction, best_pair, dt, pairs, in_cooldown, bar_idx, equity):
        """决定是否开仓"""
        if in_cooldown:
            return None, state
        
        rate = best_pair['adj_annualized_rate']
        
        # 开仓条件：基差绝对值足够大（或至少有有效价差）
        threshold = self.min_spread_threshold
        if abs(rate) >= threshold:
            new_state = {
                'holding': (
                    direction,
                    best_pair['near_symbol'],
                    best_pair['far_symbol'],
                    best_pair['spread'],
                    dt
                ),
                'entry_price_near': best_pair['near_close'],
                'entry_price_far': best_pair['far_close'],
                'entry_spread': best_pair['spread'],
                'equity_before_entry': equity,
                'last_switch_bar': bar_idx,
                'entry_rate': rate,
            }
            self._log_trade('OPEN', best_pair, direction, rate, dt, reason=f"开{direction}, 基差={rate:+.2f}%/年")
            return 'OPEN', new_state
        
        return None, state
    
    def _decide_switch(self, state, direction, best_pair, dt, pairs, in_cooldown, bar_idx):
        """决定是否切换到更好的合约对"""
        if in_cooldown:
            return None, state
        
        _, cur_near, cur_far, cur_spread, cur_dt = state['holding']
        cur_rate = state.get('entry_rate', 0)
        best_rate = best_pair['adj_annualized_rate']
        
        is_same_pair = (best_pair['near_symbol'] == cur_near and 
                        best_pair['far_symbol'] == cur_far)
        
        if is_same_pair:
            return None, state
        
        # 切换条件：
        # 1. 新对的基差优势明显超过旧对
        # 2. 或者当前对即将交割需要展期
        cur_near_exp = get_expiry(cur_near)
        days_left = (cur_near_exp.date() if hasattr(cur_near_exp, 'date') else cur_near_exp - dt.date() if hasattr(dt, 'date') else 99).days if hasattr(cur_near_exp, 'date') else (cur_near_exp - dt.date()).days
        
        # 强制展期
        if days_left <= self.rollover_days:
            new_state = {
                'holding': (
                    direction,
                    best_pair['near_symbol'],
                    best_pair['far_symbol'],
                    best_pair['spread'],
                    dt
                ),
                'entry_price_near': best_pair['near_close'],
                'entry_price_far': best_pair['far_close'],
                'entry_spread': best_pair['spread'],
                'last_switch_bar': bar_idx,
                'entry_rate': best_rate,
            }
            self._log_trade('SWITCH', best_pair, direction, best_rate, dt, 
                          reason=f"展期({days_left}d剩余), {cur_near}/{cur_far}->{best_pair['near_symbol']}/{best_pair['far_symbol']}")
            return 'SWITCH', new_state
        
        # 主动切换：新对比当前入场时的基差更优
        if direction == 'LONG' and best_rate > cur_rate + self.sigma_exit:
            new_state = {
                'holding': (
                    direction,
                    best_pair['near_symbol'],
                    best_pair['far_symbol'],
                    best_pair['spread'],
                    dt
                ),
                'entry_price_near': best_pair['near_close'],
                'entry_price_far': best_pair['far_close'],
                'entry_spread': best_pair['spread'],
                'last_switch_bar': bar_idx,
                'entry_rate': best_rate,
            }
            self._log_trade('SWITCH', best_pair, direction, best_rate, dt,
                          reason=f"优化切换: {cur_rate:+.2f}%->{best_rate:+.2f}%/年")
            return 'SWITCH', new_state
        
        if direction == 'SHORT' and best_rate < cur_rate - self.sigma_exit:
            new_state = {
                'holding': (
                    direction,
                    best_pair['near_symbol'],
                    best_pair['far_symbol'],
                    best_pair['spread'],
                    dt
                ),
                'entry_price_near': best_pair['near_close'],
                'entry_price_far': best_pair['far_close'],
                'entry_spread': best_pair['spread'],
                'last_switch_bar': bar_idx,
                'entry_rate': best_rate,
            }
            self._log_trade('SWITCH', best_pair, direction, best_rate, dt,
                          reason=f"优化切换: {cur_rate:+.2f}%->{best_rate:+.2f}%/年")
            return 'SWITCH', new_state
        
        return None, state
    
    def _close_position(self, state, dt, pairs, mult, equity, is_switch=False):
        """平仓并返回PnL"""
        if state['holding'] is None:
            return 0
        
        _, near_sym, far_sym, entry_spread, entry_dt = state['holding']
        direction = state['holding'][0]
        
        # 找平仓时的价差
        exit_spread = None
        for p in pairs:
            if p['near_symbol'] == near_sym and p['far_symbol'] == far_sym:
                exit_spread = p['spread']
                break
        
        if exit_spread is None:
            # 用最后一笔已知价差（可能已经过期）
            exit_spread = state.get('entry_spread', 0)
        
        spread_change = exit_spread - entry_spread
        dir_sign = 1 if direction == 'LONG' else -1
        
        gross_pnl = dir_sign * spread_change * mult * self.volume
        commission = (state.get('entry_price_near', 0) + state.get('entry_price_far', 0)) * mult * self.volume * self.commission_rate
        if not is_switch:
            commission *= 2  # 开+平
        
        net_pnl = gross_pnl - commission
        
        trade_type = 'SWITCH' if is_switch else 'CLOSE'
        self.trades.append({
            'timestamp': dt,
            'action': trade_type,
            'near_sym': near_sym,
            'far_sym': far_sym,
            'direction': direction,
            'entry_spread': entry_spread,
            'exit_spread': exit_spread,
            'spread_change': spread_change,
            'pnl_point': dir_sign * spread_change,
            'pnl_rmb': net_pnl,
            'commission': commission,
            'reason': '',
        })
        
        return net_pnl
    
    def _log_trade(self, action, pair, direction, rate, dt, reason=''):
        self.trades.append({
            'timestamp': dt,
            'action': action,
            'near_sym': pair['near_symbol'],
            'far_sym': pair['far_symbol'],
            'direction': direction,
            'entry_spread': pair['spread'],
            'exit_spread': 0,
            'spread_change': 0,
            'pnl_point': 0,
            'pnl_rmb': 0,
            'commission': 0,
            'rate': rate,
            'reason': reason,
        })
    
    def _record_equity(self, equity, dt):
        self.equity_curve.append(equity)
        self.equity_times.append(dt)
    
    def _build_results(self, product, final_equity, state):
        """构建回测结果报告"""
        eq = np.array(self.equity_curve)
        
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        
        # 最大回撤
        peak = np.maximum.accumulate(eq)
        drawdowns = (eq - peak) / peak * 100
        max_dd = drawdowns.min()
        
        # 统计交易次数
        n_opens = sum(1 for t in self.trades if t['action'] == 'OPEN')
        n_switches = sum(1 for t in self.trades if t['action'] == 'SWITCH')
        n_closes = sum(1 for t in self.trades if t['action'] == 'CLOSE')
        
        # 盈利交易
        profitable = [t for t in self.trades if t.get('pnl_rmb', 0) > 0 and t['action'] != 'OPEN']
        win_rate = len(profitable) / max(len([t for t in self.trades if t.get('pnl_rmb', 0) != 0 and t['action'] != 'OPEN']), 1) * 100
        
        # 总PnL（不含未平仓）
        realized_pnl = sum(t.get('pnl_rmb', 0) for t in self.trades)
        
        # Profit Factor
        gains = sum(t.get('pnl_rmb', 0) for t in self.trades if t.get('pnl_rmb', 0) > 0)
        losses = abs(sum(t.get('pnl_rmb', 0) for t in self.trades if t.get('pnl_rmb', 0) < 0))
        pf = gains / losses if losses > 0 else float('inf')
        
        # 年化收益
        if len(self.equity_times) >= 2:
            days_elapsed = (self.equity_times[-1] - self.equity_times[0]).days
            annualized_return = total_return / max(days_elapsed, 1) * 365
        else:
            annualized_return = 0
        
        results = {
            'product': product,
            'total_return_pct': round(total_return, 2),
            'annualized_return_pct': round(annualized_return, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'n_trades': n_opens + n_switches + n_closes,
            'n_opens': n_opens,
            'n_switches': n_switches,
            'n_closes': n_closes,
            'win_rate_pct': round(win_rate, 1),
            'profit_factor': round(pf, 2),
            'final_equity': round(final_equity, 2),
            'realized_pnl': round(realized_pnl, 2),
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'equity_times': self.equity_times,
        }
        
        # 打印摘要
        print(f"\n  --- {product} 多合约回测结果 ---")
        print(f"  总收益率:     {total_return:+.2f}%")
        print(f"  年化收益率:   {annualized_return:+.2f}%")
        print(f"  最大回撤:     {max_dd:.2f}%")
        print(f"  交易数:       {n_opens}(开) + {n_switches}(切) + {n_closes}(平)")
        print(f"  胜率:         {win_rate:.1f}%")
        print(f"  盈亏比(PF):   {pf:.2f}")
        print(f"  已实现盈亏:   ¥{realized_pnl:,.0f}")
        print(f"  最终权益:     ¥{final_equity:,.0f}")
        
        # 打印交易明细
        print(f"\n  交易记录:")
        print(f"  {'时间':<18} {'操作':<8} {'合约对':<16} {'方向':<6} {'基差%':>8} {'盈亏¥':>10} {'原因'}")
        print(f"  {'-'*75}")
        for t in self.trades:
            ts = str(t['timestamp'])[:16] if isinstance(t['timestamp'], datetime) else str(t['timestamp'])[:16]
            pair_str = f"{t.get('near_sym','')}/{t.get('far_sym','')}"
            pnl_str = f"{t.get('pnl_rmb',0):+,.0f}" if t.get('pnl_rmb', 0) != 0 else "-"
            rate_str = f"{t.get('rate', 0):+.2f}" if 'rate' in t else "-"
            reason = t.get('reason', '')[:35]
            print(f"  {ts:<18} {t['action']:<8} {pair_str:<16} {t.get('direction','-'):<6} {rate_str:>8} {pnl_str:>10} {reason}")
        
        return results


# ============================================================
# Phase 4: 生成HTML报告
# ============================================================

def generate_html_report(all_results, output_file):
    """生成多场景对比HTML报告"""
    scenes = []
    for r in all_results:
        scenes.append({
            'name': r['product'],
            'return': r['total_return_pct'],
            'annualized': r['annualized_return_pct'],
            'dd': r['max_drawdown_pct'],
            'trades': r['n_trades'],
            'wr': r['win_rate_pct'],
            'pf': r['profit_factor'],
            'equity_curve': r['equity_curve'],
            'equity_times': r['equity_times'],
            'trades_detail': r['trades'],
        })
    
    # 构建图表数据
    charts_js = ""
    for s in scenes:
        name = s['name']
        times_json = "[" + ",".join([f"'{str(t)}'" for t in s['equity_times']]) + "]"
        eq_json = "[" + ",".join([f"{v:.0f}" for v in s['equity_curve']]) + "]"
        charts_js += f"""
        // Equity curve for {name}
        ctx_{name} = document.getElementById('chart_{name}').getContext('2d');
        new Chart(ctx_{name}, {{
            type: 'line',
            data: {{
                labels: {times_json},
                datasets: [{{
                    label: '{name} 权益',
                    data: {eq_json},
                    borderColor: '#0066cc',
                    backgroundColor: 'rgba(0,102,204,0.1)',
                    borderWidth: 1.5,
                    fill: true,
                    pointRadius: 0,
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ display: false }},
                    y: {{ title: {{ display: true, text: '权益 (¥)' }} }}
                }}
            }}
        }});
        """
    
    # 表格行
    table_rows = ""
    for s in scenes:
        color = '#cc0000' if s['return'] >= 0 else '#00aa00'
        dd_color = '#cc0000' if s['dd'] >= 0 else '#00aa00'
        table_rows += f"""
        <tr>
          <td><b>{s['name']}</b></td>
          <td style="color:{color};font-weight:bold">{s['return']:+.2f}%</td>
          <td>{s['annualized']:+.2f}%</td>
          <td style="color:{dd_color}">{s['dd']:.2f}%</td>
          <td>{s['trades']}</td>
          <td>{s['wr']:.1f}%</td>
          <td>{s['pf']:.2f}</td>
        </tr>"""
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>多合约跨期价差策略回测 - CCFX真实数据</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 20px; background: #f8f9fa; }}
h1 {{ color: #333; text-align: center; }}
table {{ border-collapse: collapse; width: 100%; max-width: 900px; margin: 20px auto; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th, td {{ padding: 10px 14px; text-align: center; border-bottom: 1px solid #eee; }}
th {{ background: #2c3e50; color: white; font-weight: 600; }}
tr:hover {{ background: #f5f5f5; }}
.chart-container {{ width: 45%; display: inline-block; padding: 10px; vertical-align: top; }}
.charts-wrapper {{ text-align: center; }}
.trade-table {{ font-size: 13px; max-width: 1100px; }}
.summary {{ text-align: center; color: #666; margin: 10px 0; }}
</style></head><body>
<h1>📊 多合约跨期价差策略 — CCFX真实Tick数据</h1>
<p class="summary">数据源: CCFX L2逐笔行情 | 4合约全矩阵 | R=0分红调整 | 自动最优合约对选择</p>

<h2 style="text-align:center">📋 各品种回测对比</h2>
<table>
<tr><th>品种</th><th>总收益率</th><th>年化</th><th>最大回撤</th><th>交易数</th><th>胜率</th><th>PF</th></tr>
{table_rows}
</table>

<div class="charts-wrapper">
"""

    for s in scenes:
        html += f'<div class="chart-container"><canvas id="chart_{s["name"]}" width="400" height="250"></canvas></div>\n'
    
    html += "</div>"
    
    # 交易明细表
    for s in scenes:
        html += f"<h3 style='margin-top:30px'>{s['name']} 交易明细</h3>\n"
        html += f"<table class='trade-table'><tr><th>时间</th><th>操作</th><th>合约对</th><th>方向</th><th>基差%/年</th><th>盈亏(¥)</th><th>原因</th></tr>"
        for t in s['trades_detail']:
            ts = str(t['timestamp'])[:16] if isinstance(t['timestamp'], (str, datetime)) else '-'
            ps = f"{t.get('near_sym','')}/{t.get('far_sym','')}"
            pnl = f"{t.get('pnl_rmb',0):+,.0f}" if t.get('pnl_rmb', 0) != 0 else '-'
            rate = f"{t.get('rate',0):+.2f}" if 'rate' in t else '-'
            reason = t.get('reason', '')[:50]
            html += f"<tr><td>{ts}</td><td>{t['action']}</td><td>{ps}</td><td>{t.get('direction','-')}</td><td>{rate}</td><td>{pnl}</td><td>{reason}</td></tr>"
        html += "</table>\n"
    
    html += """
<script>
""" + charts_js + """
</script></body></html>"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n  HTML报告: {output_file}")


# ============================================================
# 主程序
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Step 1: 加载CCFX数据
    contracts_data = process_all_days()
    
    # Step 2: 初始化基差计算器
    basis_calc = MultiContractBasisCalculator()
    
    # Step 3: 定义四场景的方向信号
    # 场景定义与之前一致：
    # S1: IF基准看多(+1)
    # S2: IM强看多(+2)
    # S3: 混合看多(IM+2, IC+1, IF+1) — 取最强信号品种
    # S4: 看空(IC-1, IH-1) — 取空头信号
    
    # 获取公共时间轴（用IF作为参考）
    ref_product = 'IF'
    if ref_product in contracts_data and contracts_data[ref_product]:
        ref_symbols = list(contracts_data[ref_product].keys())
        ref_df = contracts_data[ref_product][ref_symbols[0]] if ref_symbols else None
        if ref_df is not None:
            all_times = ref_df['datetime'].tolist()
        else:
            all_times = []
    else:
        all_times = []
    
    # 构建方向信号映射 {datetime: signal_value}
    # 使用简单的固定信号（与之前的quick_backtest保持一致）
    def build_signal_map(signal_config):
        """
        signal_config: dict {product: signal_value}
        signal_value: -2 ~ +2
        返回: {datetime: value_for_primary_product}
        """
        sig_map = {}
        primary_prod = None
        max_sig = 0
        for prod, val in signal_config.items():
            if abs(val) > abs(max_sig):
                max_sig = val
                primary_prod = prod
        for dt in all_times:
            sig_map[dt] = signal_config.get(primary_prod, 0)
        return sig_map, primary_prod
    
    # 四场景配置
    scenarios = {
        'S1_IF基准':   {'IF': 1},              # IF +1 看多
        'S2_IM强看多': {'IM': 2},             # IM +2 强看多
        'S3_混合看多': {'IM': 2, 'IC': 1, 'IF': 1},  # 混合
        'S4_看空IC':   {'IC': -1, 'IH': -1},  # IC/IH 看空
    }
    
    all_results = []
    
    for scene_name, sig_config in scenarios.items():
        print(f"\n\n{'#'*70}")
        print(f"  📊 场景: {scene_name}")
        print(f"  信号配置: {sig_config}")
        print(f"{'#'*70}")
        
        # 确定主要品种（信号最强的那个）
        primary_prod = max(sig_config.keys(), key=lambda k: abs(sig_config[k]))
        primary_val = sig_config[primary_prod]
        
        # 为该品种构建信号map（简化：整个时间段内信号恒定）
        sig_map = {dt: primary_val for dt in all_times}
        
        if primary_prod not in contracts_data or not contracts_data[primary_prod]:
            print(f"  ⚠️ {primary_prod} 无数据，跳过")
            continue
        
        # 创建回测引擎
        engine = MultiContractBacktestEngine(
            sigma_entry=1.0,
            sigma_exit=0.3,
            cooldown_bars=12,
            rollover_days=5,
            initial_capital=1_000_000,
            volume=1,
        )
        
        # 运行回测
        result = engine.run(
            contracts_data[primary_prod],
            sig_map,
            primary_prod,
            basis_calc
        )
        
        result['scene'] = scene_name
        all_results.append(result)
    
    # Step 4: 生成综合HTML报告
    report_path = os.path.join(REPORT_DIR, 'multi_contract_backtest.html')
    generate_html_report(all_results, report_path)
    
    # 打印最终汇总
    print(f"\n\n{'='*80}")
    print(f"  🏆 多合约策略汇总 (CCFX真实Tick, 全部4合约)")
    print(f"{'='*80}")
    print(f"\n  {'场景':<14} {'品种':<6} {'收益率':>8} {'年化':>8} {'回撤':>8} {'交易':>5} {'胜率':>6} {'PF':>6}")
    print(f"  {'-'*68}")
    for r in all_results:
        c = '#cc0000' if r['total_return_pct'] >= 0 else '#00aa00'
        print(f"  {r['scene']:<14} {r['product']:<6} {r['total_return_pct']:>+7.2f}% {r['annualized_return_pct']:>+7.2f}% "
              f"{r['max_drawdown_pct']:>7.2f}% {r['n_trades']:>5} {r['win_rate_pct']:>5.1f}% {r['profit_factor']:>6.2f}")
    
    print(f"\n  报告文件: {report_path}")
    
    return all_results


if __name__ == '__main__':
    main()
