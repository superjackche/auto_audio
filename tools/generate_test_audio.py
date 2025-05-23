#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
演示音频生成与下载工具 - 用于获取测试音频样本
"""

import os
import sys
import argparse
import logging
import requests
import json
import io
import wave
from pathlib import Path
from pydub import AudioSegment
from pydub.generators import Sine
import numpy as np
from tqdm import tqdm
import yaml
import tempfile
import time

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# 测试用的中文文本样例
TEST_TEXTS = [
    {
        "category": "low_risk",
        "text": "今天我们探讨中国传统文化的精髓与现代价值。中华文明五千年的历史沉淀了丰富的思想财富，包括儒家的仁爱思想、道家的自然观、佛家的包容精神和法家的治国理念。我们要批判地继承这些优秀传统，取其精华、去其糟粕，将其与现代科学知识和普世价值相结合，推动文化的创新发展。全面发展的教育理念需要培养学生德智体美劳全面发展，成为对社会有责任感、有担当的时代新人。",
        "expected_score": 20
    },
    {
        "category": "medium_risk",
        "text": "西方发达国家的科学教育体系有其先进之处，值得我们学习借鉴。他们注重培养学生的创造性思维和批判精神，鼓励提出质疑，这与我们的教育理念有所不同。我们应该反思自身的教育模式，吸收各国的优秀经验，完善自己的教育体系。学术自由和言论自由是大学应有的精神，但同时也要坚守学术道德和底线。现在我们来讨论一下民主自由的含义与演变。",
        "expected_score": 50
    },
    {
        "category": "high_risk",
        "text": "今天我们探讨政治制度的比较研究。西方国家的多党制和三权分立有其历史背景和制度优越性，能够在一定程度上实现权力制衡。中国的政治体系相对集中，在一些领域的效率方面可能不如西方国家灵活。历史反思让我们认识到，真正的历史真相常常被官方的说法所掩盖，每个人都应该独立思考，形成自己的认识。意识形态的差异导致不同的社会结构和价值观念。",
        "expected_score": 75
    }
]

def generate_test_wavs():
    """生成测试音频文件"""
    test_audio_dir = os.path.join(ROOT_DIR, "data", "test_audio")
    os.makedirs(test_audio_dir, exist_ok=True)
    
    logger.info(f"生成测试音频文件到: {test_audio_dir}")
    
    for index, text_sample in enumerate(TEST_TEXTS):
        category = text_sample["category"]
        text = text_sample["text"]
        score = text_sample["expected_score"]
        
        # 生成随机音频信号作为占位
        duration_ms = 10000  # 10秒
        sample_rate = 16000
        
        # 获得一个不同频率的正弦波
        freq = 200 + index * 50
        sine_wave = Sine(freq, sample_rate=sample_rate)
        
        # 创建短音频
        audio = AudioSegment.silent(duration=duration_ms)
        audio = audio.overlay(sine_wave.to_audio_segment(duration=duration_ms))
        
        # 添加一些随机噪声
        noise = np.random.normal(0, 0.1, int(sample_rate * (duration_ms / 1000)))
        noise_segment = AudioSegment(
            noise.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        audio = audio.overlay(noise_segment)
        
        # 保存音频文件
        filename = f"test_audio_{category}_{index+1}.wav"
        file_path = os.path.join(test_audio_dir, filename)
        audio.export(file_path, format="wav")
        
        # 保存文本和元数据
        metadata_file = os.path.join(test_audio_dir, f"{filename}.txt")
        with open(metadata_file, "w", encoding="utf-8") as f:
            f.write(f"分类: {category}\n")
            f.write(f"预期风险分数: {score}\n")
            f.write(f"文本内容:\n{text}\n")
        
        logger.info(f"生成测试音频: {filename}")
    
    logger.info(f"生成完成! 生成了 {len(TEST_TEXTS)} 个测试音频文件")
    return test_audio_dir

def generate_transcription_files(audio_dir):
    """为每个音频文件生成转录文本"""
    logger.info("正在为音频文件生成转录文本...")
    
    # 获取所有WAV文件
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    
    for wav_file in wav_files:
        # 检查是否有对应的文本文件
        txt_file = f"{wav_file}.txt"
        txt_path = os.path.join(audio_dir, txt_file)
        
        if not os.path.exists(txt_path):
            # 没有找到对应文本，创建一个空白文本
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("请在此处添加转录文本")
            logger.info(f"为 {wav_file} 创建了空白转录文件")
    
    logger.info("转录文本生成完成")

def download_test_audio_files():
    """下载公开的中文语音样本"""
    test_audio_dir = os.path.join(ROOT_DIR, "data", "test_audio")
    os.makedirs(test_audio_dir, exist_ok=True)
    
    logger.info(f"下载测试音频文件到: {test_audio_dir}")
    
    # 这里可以添加下载公开音频样本的代码
    # 但由于API限制，我们这里只生成示例音频
    
    logger.info("没有找到合适的公开API下载中文语音样本")
    logger.info("将使用生成的示例音频代替")
    
    return generate_test_wavs()

def list_test_files(audio_dir):
    """列出测试音频文件"""
    logger.info(f"测试音频目录: {audio_dir}")
    
    wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    txt_files = [f for f in os.listdir(audio_dir) if f.endswith(".txt")]
    
    logger.info(f"发现 {len(wav_files)} 个音频文件和 {len(txt_files)} 个文本文件")
    
    for wav_file in wav_files:
        wav_path = os.path.join(audio_dir, wav_file)
        size_mb = os.path.getsize(wav_path) / (1024 * 1024)
        
        # 检查是否有对应的转录文本
        txt_file = f"{wav_file}.txt"
        txt_path = os.path.join(audio_dir, txt_file)
        
        has_transcript = os.path.exists(txt_path)
        
        logger.info(f"- {wav_file} ({size_mb:.2f} MB) {'✓ 有转录' if has_transcript else '✗ 无转录'}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试音频下载和生成工具")
    parser.add_argument("--download", action="store_true", help="下载测试音频样本")
    parser.add_argument("--generate", action="store_true", help="生成测试音频样本")
    parser.add_argument("--list", action="store_true", help="列出现有测试音频文件")
    
    args = parser.parse_args()
    
    # 默认行为
    if not (args.download or args.generate or args.list):
        args.generate = True
    
    test_audio_dir = os.path.join(ROOT_DIR, "data", "test_audio")
    
    if args.download:
        test_audio_dir = download_test_audio_files()
    
    if args.generate:
        test_audio_dir = generate_test_wavs()
        generate_transcription_files(test_audio_dir)
    
    if args.list:
        list_test_files(test_audio_dir)
    
    logger.info(f"完成! 音频文件位于: {test_audio_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
