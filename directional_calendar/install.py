"""
install.py
VeighNa 策略安装/卸载脚本

用法：
    python install.py install      # 安装到 VeighNa
    python install.py uninstall    # 卸载（删除链接）
    python install.py status       # 查看当前状态
"""

from __future__ import annotations

import os
import shutil
import sys
import argparse
from pathlib import Path


# VeighNa Studio 常见路径候选（按优先级排序）
VEIGHNA_STRATEGY_DIRS = [
    r"D:\veighna_studio\strategies",
    r"C:\veighna_studio\strategies", 
    r"D:\VeighNa\strategies",
]

# 当前策略目录
STRATEGY_SRC = Path(__file__).parent.resolve()


def find_veighna_strategies_dir() -> Path:
    """自动定位 VeighNa strategies 目录"""
    for candidate in VEIGHNA_STRATEGY_DIRS:
        p = Path(candidate)
        if p.exists():
            return p
    
    # 尝试从 Python 路径推断
    try:
        import vnpy_ctastrategy
        module_dir = Path(vnpy_ctastrategy.__file__).parent
        # 通常在 veighna_xxx/strategies/
        parent = module_dir.parent
        strategy_dir = parent / "strategies"
        if strategy_dir.exists():
            return strategy_dir
        return parent  # 回退到模块父目录
    except ImportError:
        pass
    
    print("⚠️ 无法自动找到 VeighNa strategies 目录，请手动指定")
    sys.exit(1)


def install():
    """创建符号链接到 VeighNa strategies 目录"""
    target = find_veighna_strategies_dir() / STRATEGY_SRC.name
    
    if target.exists():
        if target.is_symlink():
            actual = target.resolve()
            if actual == STRATEGY_SRC:
                print(f"✅ 已安装: {target}")
                return True
            else:
                print(f"⚠️ 已存在不同版本的 {target}，先删除...")
                target.unlink()
        else:
            print(f"⚠️ {target} 是普通目录（非符号链接），请手动处理")
            return False
    
    try:
        # Windows 上用 mklink 创建符号链接（需要管理员权限或开发者模式）
        # 如果失败则回退为目录复制
        os.symlink(str(STRATEGY_SRC), str(target))
        print(f"✅ 符号链接已创建:")
        print(f"   {target} → {STRATEGY_SRC}")
        
        # 验证导入
        verify_import(target)
        return True
    except OSError as e:
        print(f"⚠️ 符号链接创建失败 ({e})，尝试复制目录...")
        shutil.copytree(str(STRATEGY_SRC), str(target))
        print(f"✅ 已复制到: {target}")
        verify_import(target)
        return True


def uninstall():
    """移除安装"""
    target = find_veighna_strategies_dir() / STRATEGY_SRC.name
    
    if not target.exists():
        print(f"未安装: {target}")
        return
    
    if target.is_symlink():
        target.unlink()
        print(f"🗑️ 已移除符号链接: {target}")
    elif target.is_dir():
        ans = input(f"⚠️ {target} 是真实目录（非符号链接），确定删除吗？[y/N] ")
        if ans.lower() == 'y':
            shutil.rmtree(target)
            print(f"🗑️ 已删除目录: {target}")


def status():
    """查看状态"""
    target = find_veighna_strategies_dir() / STRATEGY_SRC.name
    if target.exists():
        print(f"状态: 已安装")
        print(f"  目标: {target}")
        if target.is_symlink():
            print(f"  类型: 符号链接 → {target.resolve()}")
        else:
            print(f"  类型: 目录复制")
        
        # 列出文件
        files = [f.name for f in sorted(target.glob("*.py"))]
        print(f"  文件: {', '.join(files)}")
    else:
        print("状态: 未安装")
        print(f"  源代码位置: {STRATEGY_SRC}")


def verify_installation():
    """验证策略能否被 VeighNa 正常加载"""
    print("\n--- 验证安装 ---")
    try:
        sys.path.insert(0, str(find_veighna_strategies_dir()))
        from directional_calendar import DirectionalCalendarStrategy
        
        # 检查关键属性
        assert hasattr(DirectionalCalendarStrategy, 'author'), "缺少 author 属性"
        assert hasattr(DirectionalCalendarStrategy, 'parameters'), "缺少 parameters"
        assert hasattr(DirectionalCalendarStrategy, 'on_init'), "缺少 on_init 方法"
        
        print(f"✅ DirectionalCalendarStrategy 加载成功")
        print(f"   作者: {DirectionalCalendarStrategy.author}")
        print(f"   参数: {DirectionalCalendarStrategy.parameters}")
        print(f"   变量: {DirectionalCalendarStrategy.variables}")
        return True
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False


def verify_import(target_dir: Path):
    """快速验证导入"""
    try:
        sys.path.insert(0, str(target_dir.parent))
        from directional_calendar import DirectionalCalendarStrategy
        print(f"   ✅ 导入成功: DirectionalCalendarStrategy (author={DirectionalCalendarStrategy.author})")
    except ImportError as e:
        print(f"   ⚠️ 导入测试跳过（依赖可能不在运行时路径）: {e}")


def main():
    parser = argparse.ArgumentParser(description="VeighNa 策略安装工具")
    parser.add_argument("action", choices=["install", "uninstall", "status"],
                        default="status", nargs="?")
    args = parser.parse_args()

    print("=" * 50)
    print("  带方向跨期套利策略 — VeighNa 安装器")
    print("=" * 50)

    if args.action == "install":
        success = install()
        if success:
            verify_installation()
    elif args.action == "uninstall":
        uninstall()
    elif args.action == "status":
        status()


if __name__ == "__main__":
    main()
