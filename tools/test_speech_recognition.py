#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音识别测试工具 - 用于测试系统的语音识别功能
"""

import os
import sys
import argparse
import logging
import wave
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from src.utils.config_loader import ConfigLoader
    from src.audio.speech_to_text import SpeechToText
    from src.nlp.analyzer import TextAnalyzer
    from src.analysis.risk_assessment import RiskAssessor
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有依赖: pip install -r requirements.txt")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def read_wav_file(file_path):
    """读取WAV文件"""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            # 读取所有帧
            frames = wav_file.readframes(n_frames)
            
            # 转换为NumPy数组
            if sample_width == 2:  # 16-bit音频
                data = np.frombuffer(frames, dtype=np.int16)
            elif sample_width == 4:  # 32-bit音频
                data = np.frombuffer(frames, dtype=np.int32)
            else:
                data = np.frombuffer(frames, dtype=np.uint8)  # 8-bit音频
            
            logger.info(f"已读取WAV文件: {file_path}")
            logger.info(f"  - 通道数: {n_channels}")
            logger.info(f"  - 采样率: {frame_rate} Hz")
            logger.info(f"  - 采样位深: {sample_width * 8} bit")
            logger.info(f"  - 帧数: {n_frames}")
            
            return {
                "data": data,
                "sample_rate": frame_rate,
                "channels": n_channels,
                "sample_width": sample_width
            }
    except Exception as e:
        logger.error(f"读取WAV文件失败: {str(e)}")
        return None

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试语音识别功能')
    parser.add_argument('--file', type=str, help='测试用的WAV文件路径')
    parser.add_argument('--dir', type=str, help='包含WAV文件的目录路径')
    args = parser.parse_args()
    
    # 加载配置
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    
    # 初始化组件
    stt = SpeechToText()
    analyzer = TextAnalyzer()
    risk_assessor = RiskAssessor()
    
    # 确定要处理的文件列表
    files_to_process = []
    
    if args.file:
        if os.path.exists(args.file) and args.file.lower().endswith('.wav'):
            files_to_process.append(args.file)
        else:
            logger.error(f"指定的文件不存在或不是WAV文件: {args.file}")
    
    if args.dir:
        if os.path.isdir(args.dir):
            for filename in os.listdir(args.dir):
                if filename.lower().endswith('.wav'):
                    files_to_process.append(os.path.join(args.dir, filename))
        else:
            logger.error(f"指定的目录不存在: {args.dir}")
    
    # 如果没有指定文件或目录，使用默认测试目录
    if not files_to_process:
        test_dir = os.path.join(ROOT_DIR, 'data', 'test_audio')
        if os.path.isdir(test_dir):
            for filename in os.listdir(test_dir):
                if filename.lower().endswith('.wav'):
                    files_to_process.append(os.path.join(test_dir, filename))
    
    # 处理每个文件
    for file_path in files_to_process:
        logger.info(f"处理文件: {file_path}")
        
        # 读取WAV文件
        audio_data = read_wav_file(file_path)
        if not audio_data:
            continue
        
        # 语音识别
        logger.info("正在进行语音识别...")
        text = stt.transcribe(audio_data["data"])
        logger.info(f"识别结果: {text}")
        
        if text:
            # 语义分析
            logger.info("正在进行语义分析...")
            analysis_result = analyzer.analyze(text)
            
            # 风险评估
            logger.info("正在进行风险评估...")
            risk_score, risk_factors = risk_assessor.assess(
                text=text, 
                analysis_result=analysis_result
            )
            
            logger.info(f"风险评分: {risk_score}")
            logger.info(f"风险因素: {risk_factors}")
        
        logger.info("-" * 50)

if __name__ == "__main__":
    main()
