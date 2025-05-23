#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
课堂语义行为实时分析系统 - 主启动脚本
此脚本是系统的主要入口点，负责启动Web服务和分析引擎。
"""

import os
import sys
import argparse
import logging
import threading
import webbrowser
import time
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# 打印启动标志
print("""
┌─────────────────────────────────────────────────┐
│   课堂语义行为实时分析系统 v0.1.0               │
│                                                 │
│   自动监测和分析课堂语音内容，识别潜在风险      │
└─────────────────────────────────────────────────┘
""")

# 尝试导入必要的模块
try:
    from src.web.app import app, socketio
    from src.utils.config_loader import ConfigLoader
except ImportError as e:
    print(f"错误: 导入模块失败 - {e}")
    print("请确保已安装所有依赖: pip install -r requirements.txt")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(ROOT_DIR, 'data', 'logs', 'system.log'), encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='课堂语义行为实时分析系统')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')
    return parser.parse_args()

def create_directories():
    """创建必要的目录结构"""
    dirs = [
        os.path.join(ROOT_DIR, 'data', 'logs'),
        os.path.join(ROOT_DIR, 'data', 'models'),
        os.path.join(ROOT_DIR, 'data', 'reports'),
        os.path.join(ROOT_DIR, 'data', 'test_audio')
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"创建目录: {dir_path}")

def open_browser(host, port, delay=1.5):
    """延迟一段时间后打开浏览器"""
    def _open_browser():
        time.sleep(delay)  # 等待服务器启动
        url = f"http://{host}:{port}"
        webbrowser.open(url)
        logger.info(f"已在浏览器中打开: {url}")
    
    browser_thread = threading.Thread(target=_open_browser, daemon=True)
    browser_thread.start()

def main():
    """主函数"""
    args = parse_args()
    
    # 创建必要的目录
    create_directories()
    
    # 加载配置
    config = ConfigLoader().load_config()
    
    # 打印启动信息
    logger.info("="*50)
    logger.info("课堂语义行为实时分析系统启动")
    logger.info(f"Web界面: http://{args.host}:{args.port}")
    logger.info("管理员账号: admin")
    logger.info("="*50)
    
    # 如果未指定--no-browser，则自动打开浏览器
    if not args.no_browser:
        open_browser(args.host, args.port)
    
    # 启动Flask应用
    try:
        socketio.run(app, host=args.host, port=args.port, debug=args.debug, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        logger.info("用户中断，系统关闭中...")
    except Exception as e:
        logger.error(f"启动失败: {str(e)}")
        return 1
    
    logger.info("系统已正常关闭")
    return 0

if __name__ == "__main__":
    sys.exit(main())
