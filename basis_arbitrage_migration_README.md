# 股指期货基差套利系统 - 迁移包说明

> 打包时间: 2026-04-26 22:45
> 源目录: `directional_calendar/`
> 压缩包: `basis_arbitrage_migration_20260426.zip` (约21MB)

---

## 📦 包含内容

### 核心代码 (29个 .py 文件)

| 文件 | 说明 |
|------|------|
| `basis_arbitrage_backtest.py` | **v2 基差回归套利回测引擎**（多合约对扫描，z-score触发） |
| `quick_backtest.py` | 快速回测入口 |
| `backtest.py` | 带方向信号的回测 |
| `basis_calculator.py` | 基差计算器（分红调整、年化率） |
| `data_loader.py` / `minute_data_loader.py` | 数据加载器 |
| `spread_engine.py` | 跨期价差引擎 |
| `execution.py` | 执行模块（双腿下单逻辑） |
| `direction_signal.py` | 方向信号接口 |
| `risk_manager.py` | 风控管理 |
| `param_optimizer.py` | 参数优化扫描 |
| `long_history_builder.py` / `loader.py` | 长历史数据构建与加载 |
| `main.py` | 离线运行入口（测试/信号写入） |
| `install.py` | 依赖安装脚本 |
| `spread_matrix_viewer.py` | 价差矩阵可视化 |

### 辅助/调试脚本
- `_multi_contract_bt.py` - 多合约回测
- `_directional_bt.py` - 带方向回测
- `_long_bt.py` - 长周期回测
- `_check_range.py`, `_check_sources.py`, `_deep_dive.py`, `_explore_sources.py` - 数据检查工具
- `_test_akshare.py`, `_test_api.py` - API测试

### 配置文件
- `config.yaml` - 主配置（品种参数、分红预测、风控阈值等）
- `signals/direction.json` - 外部方向信号

### 数据文件 (~20MB)
```
data/
├── 5min_basis_{IF,IH,IC,IM}_20260425.csv        # 各品种5分钟基差(akshare)
├── 5min_basis_{IF,IH,IC,IM}_ccfx_20260425.csv    # ccfx源数据
├── 5min_basis_{IF,IH,IC,IM}_long_20260425.csv     # 长历史合成数据
└── minute_basis_IF_20260425.csv                    # 分钟级IF基差
```

### 回测报告输出
- `basis_arb_output.txt` - 基差套利回测结果
- `v2_output.txt` - v2引擎输出
- `sm_output.txt` - 价差矩阵输出
- `param_opt_output.txt` - 参数优化完整日志(~1MB)
- `reports/` - 参数优化CSV + OOS日志（多轮）

---

## 🚀 在新电脑上使用

### 1. 解压
```powershell
# 解压到目标位置（如 D:\projects\ 或任意目录）
Expand-Archive basis_arbitrage_migration_20260426.zip -DestinationPath .\basis_arbitrage
```

### 2. 安装依赖
```bash
cd basis_arbitrage\directional_calendar
python install.py
# 或者手动：
pip install numpy pandas pyyaml tushare akshare scipy
```

### 3. 运行回测
```bash
# 快速回测
python quick_backtest.py --config config.yaml

# v2 套利回测（主力）
python basis_arbitrage_backtest.py --config config.yaml

# 测试基差计算
python main.py --mode basis_test --config config.yaml

# 参数优化
python param_optimizer.py --config config.yaml
```

### 4. 注意事项
- `config.yaml` 中路径需要更新为新电脑的实际路径（搜索 `C:/Users/wang` 替换即可）
- 合约代码（如 IF2606）随月份变化需手动更新或接入自动识别
- 分红预测数据来自券商研报，每季度需更新 `dividend_schedule`

---

## 🔧 系统架构概览

```
外部择时信号 → direction_signal.py
                        ↓
行情数据 ← minute_data_loader.py / data_loader.py
                        ↓
              basis_calculator.py（基差+分红调整）
                        ↓
              spread_engine.py（z-score扫描+选合约对）
                        ↓
              execution.py（双腿执行）/ backtest.py（回测）
                        ↓
              risk_manager.py（风控）
```

**核心策略**: 扫描所有可用跨期合约对的 spread z-score，选 |z-score| 最大对交易，纯基差回归，不判断方向。
