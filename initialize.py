#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
初始化脚本 - 首次使用系统前运行
"""

import os
import sys
import logging
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent

# 必要的目录列表
REQUIRED_DIRS = [
    "data/logs",
    "data/models",
    "data/reports",
    "data/test_audio"
]

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_directories():
    """创建必要的目录结构"""
    print("正在创建必要的目录...")
    
    for dir_path in REQUIRED_DIRS:
        full_path = os.path.join(ROOT_DIR, dir_path)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            print(f"  创建目录: {dir_path}")
        else:
            print(f"  目录已存在: {dir_path}")

def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    
    major, minor, _ = sys.version_info
    min_version = (3, 11)
    
    if (major, minor) >= min_version:
        print(f"  Python版本 {major}.{minor} 符合要求")
        return True
    else:
        print(f"  警告: 当前Python版本 {major}.{minor}，建议使用 3.11 或更高版本")
        return False

def main():
    """主函数"""
    print("\n====== 课堂语义行为实时分析系统 - 初始化 ======\n")
    
    # 检查Python版本
    check_python_version()
    
    # 创建目录
    create_directories()
    
    print("\n初始化完成!")
    print("您现在可以通过以下方式启动系统:")
    print("  - Windows: 双击 start.bat")
    print("  - PowerShell: 右键 start.ps1 选择'使用PowerShell运行'")
    print("  - 命令行: python run.py")
    print("\n===============================================\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
