"""
directional_calendar - 带方向的股指期货跨期套利策略

VeighNa 策略插件。

安装方式：
    1. 复制本目录到 VeighNa 的 strategies 目录
       (通常在 D:/veighna_studio/strategies/directional_calendar/)

    2. 或使用 install.py 自动安装：
       python install.py

    3. 在 VeighNa CtaStrategy 管理界面添加 "DirectionalCalendarStrategy"
"""

from execution import DirectionalCalendarStrategy

__all__ = ["DirectionalCalendarStrategy"]
__version__ = "1.0.0"
__author__ = "DirectionalCalendar Team"
