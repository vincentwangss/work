"""
main.py
离线回测 / 独立运行入口

用法：
    # 离线测试基差计算
    python main.py --mode basis_test

    # 写入一条模拟外部信号（测试信号接口）
    python main.py --mode write_signal --direction 1

    # 打印当前基差统计（连接行情数据后）
    python main.py --mode status
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime

import yaml

# 添加当前目录到 path
sys.path.insert(0, os.path.dirname(__file__))

from basis_calculator import BasisCalculator, ContractInfo
from direction_signal import (
    Direction, DirectionSignal, ExternalFileProvider,
    build_provider_from_config
)
from spread_engine import SpreadEngine
from risk_manager import RiskManager


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# 模式 1：测试基差计算
# ------------------------------------------------------------------

def run_basis_test(config: dict):
    """用模拟价格测试基差计算逻辑"""
    print("\n=== 基差计算测试 ===\n")
    calc = BasisCalculator(risk_free_rate=0.020)

    instruments = config.get("instruments", {})
    for product, cfg in instruments.items():
        if not cfg.get("enabled", False):
            continue

        # 加载分红
        dividend_schedule = cfg.get("dividend_schedule", {})
        calc.load_dividend_schedule(product, dividend_schedule)

        # 模拟近远月价格（用近似实际值）
        # 实际运行时这里替换为从行情接口获取
        mock_prices = {
            "IF": (3700.0, 3750.0),
            "IH": (2300.0, 2330.0),
            "IC": (5500.0, 5580.0),
            "IM": (5200.0, 5270.0),
        }
        near_price, far_price = mock_prices.get(product, (4000.0, 4050.0))

        near_symbol = cfg["near_symbol"]
        far_symbol = cfg["far_symbol"]

        # 解析交割日
        def get_expiry(symbol: str) -> date:
            year = int("20" + symbol[2:4])
            month = int(symbol[4:6])
            d = date(year, month, 1)
            days_to_friday = (4 - d.weekday()) % 7
            first_friday = d.replace(day=1 + days_to_friday)
            return first_friday.replace(day=first_friday.day + 14)

        near = ContractInfo(
            symbol=near_symbol, product=product,
            expiry_date=get_expiry(near_symbol), last_price=near_price
        )
        far = ContractInfo(
            symbol=far_symbol, product=product,
            expiry_date=get_expiry(far_symbol), last_price=far_price
        )

        snap = calc.calc_snapshot(near, far, product)
        print(f"品种: {product} ({cfg['name']})")
        print(f"  近月: {near_symbol} @ {near_price:.1f}  交割日: {near.expiry_date}")
        print(f"  远月: {far_symbol} @ {far_price:.1f}  交割日: {far.expiry_date}")
        print(f"  剩余天数差: {snap.days_far - snap.days_near}天")
        print(f"  区间分红预估: {snap.dividend_between:.2f}点")
        print(f"  原始基差: {snap.raw_basis:+.2f}点  ({snap.raw_annualized_rate:+.2f}% 年化)")
        print(f"  分红调整后基差: {snap.dividend_adjusted_basis:+.2f}点  ({snap.adj_annualized_rate:+.2f}% 年化)")
        print()


# ------------------------------------------------------------------
# 模式 2：写入外部信号（测试用）
# ------------------------------------------------------------------

def write_test_signal(config: dict, direction: int):
    """向外部信号文件写入一条测试信号"""
    signal_file = config["direction"]["external_signal_file"]
    os.makedirs(os.path.dirname(signal_file), exist_ok=True)

    dir_name = {1: "多头", -1: "空头", 0: "中性"}[direction]
    signals = {}
    for product, cfg in config["instruments"].items():
        if cfg.get("enabled", False):
            signals[product] = {
                "direction": direction,
                "confidence": 0.80,
                "reason": f"测试信号({dir_name})",
            }

    data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "signals": signals,
    }
    with open(signal_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已写入信号文件: {signal_file}")
    print(json.dumps(data, ensure_ascii=False, indent=2))


# ------------------------------------------------------------------
# 模式 3：决策演示
# ------------------------------------------------------------------

def run_decision_demo(config: dict):
    """模拟一次完整的决策流程"""
    print("\n=== 换仓决策演示 ===\n")

    calc = BasisCalculator()
    provider = build_provider_from_config(config["direction"])
    engine = SpreadEngine(calc, provider, config)

    # 模拟喂入一些历史基差数据（供 Z-score 计算）
    for product in ["IF", "IH", "IC"]:
        import numpy as np
        for i in range(65):
            from datetime import timedelta
            ts = datetime.now() - timedelta(days=65 - i)
            near_p = 3700 + np.random.randn() * 30
            far_p = near_p + 40 + np.random.randn() * 15
            state = engine.get_state(product)
            if not state:
                continue
            cfg = config["instruments"].get(product, {})

            def get_expiry(symbol: str) -> date:
                year = int("20" + symbol[2:4])
                month = int(symbol[4:6])
                d = date(year, month, 1)
                days_to_friday = (4 - d.weekday()) % 7
                ff = d.replace(day=1 + days_to_friday)
                return ff.replace(day=ff.day + 14)

            near = ContractInfo(cfg["near_symbol"], product,
                                get_expiry(cfg["near_symbol"]), near_p)
            far = ContractInfo(cfg["far_symbol"], product,
                               get_expiry(cfg["far_symbol"]), far_p)
            calc.calc_snapshot(near, far, product, as_of=ts)

    # 当前行情（模拟远月超升水）
    contracts_now = {}
    for product in ["IF", "IH", "IC"]:
        cfg = config["instruments"].get(product, {})
        if not cfg.get("enabled", False):
            continue

        def get_expiry(symbol: str) -> date:
            year = int("20" + symbol[2:4])
            month = int(symbol[4:6])
            d = date(year, month, 1)
            days_to_friday = (4 - d.weekday()) % 7
            ff = d.replace(day=1 + days_to_friday)
            return ff.replace(day=ff.day + 14)

        near_price = {"IF": 3700, "IH": 2300, "IC": 5500}[product]
        # 让远月超升水（触发换仓信号）
        far_price = near_price + 80

        contracts_now[product] = {
            "near": ContractInfo(cfg["near_symbol"], product,
                                 get_expiry(cfg["near_symbol"]), near_price),
            "far": ContractInfo(cfg["far_symbol"], product,
                                get_expiry(cfg["far_symbol"]), far_price),
        }

    decisions = engine.on_tick(contracts_now)
    if not decisions:
        print("当前无换仓信号（维持现状）")
    for d in decisions:
        print(str(d))

    print("\n" + engine.get_decision_summary())


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="股指期货跨期套利系统 - 工具入口")
    parser.add_argument("--config",
                        default="C:/Users/wang/WorkBuddy/20260425111208/directional_calendar/config.yaml",
                        help="配置文件路径")
    parser.add_argument("--mode",
                        choices=["basis_test", "write_signal", "demo"],
                        default="basis_test",
                        help="运行模式")
    parser.add_argument("--direction", type=int, default=1,
                        help="write_signal 模式的方向: 1=多 -1=空 0=中性")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "basis_test":
        run_basis_test(config)
    elif args.mode == "write_signal":
        write_test_signal(config, args.direction)
    elif args.mode == "demo":
        run_decision_demo(config)


if __name__ == "__main__":
    main()
