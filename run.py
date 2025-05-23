
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
│   课堂语义行为实时分析系统 v0.2.1               │
│                                                 │
│   自动监测和分析课堂语音内容，识别潜在风险      │
│   Python 3.13 兼容版本                         │
└─────────────────────────────────────────────────┘
""")

# 显示Python版本信息
python_version = sys.version_info
print(f"Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")

if python_version >= (3, 13):
    print("⚠️  检测到Python 3.13+，某些功能使用兼容性实现")

print("正在检查系统模块...")

# 尝试导入必要的模块
module_status = {}
try:
    from src.web.app import app, socketio
    module_status['web'] = True
    print("✓ Web应用模块导入成功")
except ImportError as e:
    module_status['web'] = False
    print(f"✗ Web应用模块导入失败: {e}")

try:
    from src.utils.config_loader import ConfigLoader
    module_status['config'] = True
    print("✓ 配置加载器导入成功")
except ImportError as e:
    module_status['config'] = False
    print(f"✗ 配置加载器导入失败: {e}")

# 检查其他关键模块
try:
    from src.audio.speech_to_text import SpeechToText
    module_status['speech'] = True
    print("✓ 语音转文字模块导入成功")
except ImportError as e:
    module_status['speech'] = False
    print(f"✗ 语音转文字模块导入失败: {e}")

try:
    from src.nlp.analyzer import SemanticAnalyzer
    module_status['nlp'] = True
    print("✓ 语义分析模块导入成功")
except ImportError as e:
    module_status['nlp'] = False
    print(f"✗ 语义分析模块导入失败: {e}")

# 如果关键模块导入失败，退出
if not module_status.get('web', False) or not module_status.get('config', False):
    print("\n错误: 关键模块导入失败")
    print("请确保已安装所有依赖: pip install -r requirements.txt")
    if python_version >= (3, 13):
        print("或运行兼容性检查: python python313_compatibility_check.py")
    sys.exit(1)

# 显示模块状态摘要
success_count = sum(module_status.values())
total_count = len(module_status)
print(f"\n模块检查完成: {success_count}/{total_count} 个模块正常")

if success_count < total_count:
    print("注意: 某些模块有问题，但系统会使用后备实现继续运行")

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
