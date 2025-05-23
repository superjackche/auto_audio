#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英文演示音频生成工具 - 用于获取英文测试音频样本
"""

import os
import sys
import argparse
import logging
import io
import wave
from pathlib import Path
from pydub import AudioSegment
from pydub.generators import Sine
import numpy as np
from tqdm import tqdm
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

# 测试用的英文文本样例
ENGLISH_TEST_TEXTS = [
    {
        "category": "low_risk",
        "text": "Today we will explore the academic structure and educational philosophy in different countries. Educational systems around the world have unique strengths. The Chinese education system emphasizes discipline and foundational knowledge, while Western education often focuses on creativity and critical thinking. Both approaches have merits and can complement each other. In our international cooperation programs, we should combine these strengths to develop well-rounded students who can excel in global contexts while maintaining their cultural identity and values.",
        "expected_score": 25
    },
    {
        "category": "medium_risk",
        "text": "In our comparative politics class today, we'll examine governance systems across different countries. Democratic systems typically feature separation of powers and regular elections, while other systems might prioritize centralized decision-making for efficiency. Each system has evolved in response to specific historical and cultural contexts. It's important to understand that no single system is universally superior, though transparency and accountability should be valued in any governance structure. Academic freedom allows us to discuss these topics openly, but always with nuance and respect for different perspectives.",
        "expected_score": 45
    },
    {
        "category": "high_risk",
        "text": "Today's discussion focuses on human rights and freedom of speech. Western democracies often criticize other countries for human rights issues while overlooking their own problems. The concept of universal human rights can sometimes conflict with national sovereignty and cultural contexts. Some argue that social stability should take precedence over unlimited individual freedoms. Others believe that political pluralism and opposition parties are essential for genuine democratic governance. These tensions between collective security and individual liberties create ongoing debates about the ideal balance in governance systems.",
        "expected_score": 70
    },
    {
        "category": "critical_risk",
        "text": "Our analysis today examines authoritarian systems and their failures. Government corruption and elite capture have undermined public institutions in many countries. Mass surveillance and privacy invasion have become tools of digital authoritarianism. The repression of minorities and ethnic discrimination represent serious human rights abuses that demand international attention. True democracy requires free elections without voter suppression. We must critically examine cases of arbitrary detention, re-education camps, and forced labor that continue to occur in various regions worldwide.",
        "expected_score": 90
    }
]

def generate_english_test_wavs():
    """生成英文测试音频文件"""
    test_audio_dir = os.path.join(ROOT_DIR, "data", "test_audio", "english")
    os.makedirs(test_audio_dir, exist_ok=True)
    
    logger.info(f"生成英文测试音频文件到: {test_audio_dir}")
    
    for index, text_sample in enumerate(ENGLISH_TEST_TEXTS):
        category = text_sample["category"]
        text = text_sample["text"]
        score = text_sample["expected_score"]
        
        # 生成随机音频信号作为占位
        duration_ms = 12000  # 12秒
        sample_rate = 16000
        
        # 获得一个不同频率的正弦波
        freq = 220 + index * 55
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
        filename = f"english_test_{category}_{index+1}.wav"
        file_path = os.path.join(test_audio_dir, filename)
        audio.export(file_path, format="wav")
        
        # 保存文本和元数据
        metadata_file = os.path.join(test_audio_dir, f"{filename}.txt")
        with open(metadata_file, "w", encoding="utf-8") as f:
            f.write(f"Classification: {category}\n")
            f.write(f"Expected Risk Score: {score}\n")
            f.write(f"Content:\n{text}\n")
        
        logger.info(f"生成英文测试音频: {filename}")
    
    logger.info(f"生成完成! 生成了 {len(ENGLISH_TEST_TEXTS)} 个英文测试音频文件")
    return test_audio_dir

def list_english_test_files(audio_dir):
    """列出英文测试音频文件"""
    logger.info(f"英文测试音频目录: {audio_dir}")
    
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
    parser = argparse.ArgumentParser(description="英文测试音频生成工具")
    parser.add_argument("--generate", action="store_true", help="生成英文测试音频样本")
    parser.add_argument("--list", action="store_true", help="列出现有英文测试音频文件")
    
    args = parser.parse_args()
    
    # 默认行为
    if not (args.generate or args.list):
        args.generate = True
    
    test_audio_dir = os.path.join(ROOT_DIR, "data", "test_audio", "english")
    os.makedirs(test_audio_dir, exist_ok=True)
    
    if args.generate:
        test_audio_dir = generate_english_test_wavs()
    
    if args.list:
        list_english_test_files(test_audio_dir)
    
    logger.info(f"完成! 英文音频文件位于: {test_audio_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
