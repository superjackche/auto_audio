#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
课堂语义行为实时分析系统 - 核心分析引擎
此脚本是系统的核心分析引擎，负责音频捕获、语音识别和语义分析，不包含Web界面。
如果您只需要核心分析功能而不需要Web界面，请使用此脚本。
如果您需要完整的功能包括Web界面，请使用 run.py。
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# 打印启动标志
print("""
┌─────────────────────────────────────────────────┐
│   课堂语义行为分析引擎 v0.1.0                   │
│                                                 │
│   核心分析引擎 - 命令行模式                     │
└─────────────────────────────────────────────────┘
""")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(ROOT_DIR, 'data', 'logs', 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='课堂语义行为实时分析系统')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='配置文件路径')
    parser.add_argument('--no-web', action='store_true',
                      help='不启动Web界面')
    parser.add_argument('--debug', action='store_true',
                      help='启用调试模式')
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    config_path = os.path.join(ROOT_DIR, args.config)
    config = load_config(config_path)
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    
    try:        # 初始化组件
        logger.info("初始化系统组件...")
        audio_recorder = AudioRecorder(config['audio'])
        stt_engine = SpeechToTextEngine(config['speech_to_text'])
        text_analyzer = BilingualTextAnalyzer(config['nlp'])
        risk_assessor = RiskAssessor(config['risk_assessment'])
        
        # 启动Web服务器（如果需要）
        web_server = None
        if not args.no_web:
            logger.info("启动Web界面...")
            web_server = start_web_server(config['web'])
        
        # 启动处理管道
        logger.info("系统启动完成，开始处理...")
        
        # 主循环
        try:
            while True:
                # 捕获音频
                audio_data = audio_recorder.record()
                  # 语音转文本
                text = stt_engine.transcribe(audio_data)
                if text:
                    logger.debug(f"识别文本: {text}")
                    
                    # 语义分析
                    analysis_result = text_analyzer.analyze(text)
                    
                    # 风险评估
                    risk_result = risk_assessor.assess(analysis_result)
                    
                    # 处理风险结果
                    if risk_result['risk_level'] > config['risk_assessment']['alert_threshold']:
                        logger.warning(f"发现高风险内容: {risk_result['risk_factors']}")
                        # 这里可以添加预警通知的代码
                
                time.sleep(0.1)  # 避免CPU过度使用
                
        except KeyboardInterrupt:
            logger.info("接收到停止信号，正在关闭系统...")
        
        # 关闭资源
        if web_server:
            web_server.stop()
        audio_recorder.stop()
        
        logger.info("系统已正常关闭")
        
    except Exception as e:
        logger.exception(f"系统运行出错: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
