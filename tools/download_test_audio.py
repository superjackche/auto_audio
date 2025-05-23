#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试音频下载工具 - 用于下载一些用于测试的中文语音样本
"""

import os
import sys
import argparse
import logging
import requests
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# 测试音频资源列表
TEST_AUDIO_RESOURCES = [
    {
        "name": "chinese_speech_sample_1.wav",
        "url": "https://github.com/speechio/BigCiDian/raw/master/samples/BAIDU/001.wav"
    },
    {
        "name": "chinese_speech_sample_2.wav",
        "url": "https://github.com/chenchao700/THCHS30-data/raw/master/data_thchs30/train/A11_0.wav"
    },
    {
        "name": "chinese_speech_sample_3.wav",
        "url": "https://github.com/chenchao700/THCHS30-data/raw/master/data_thchs30/train/A2_0.wav"
    }
]

def download_file(url, output_path):
    """下载文件到指定路径"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))
                
        logger.info(f"下载完成: {output_path}")
        return True
    except Exception as e:
        logger.error(f"下载失败: {url} - {str(e)}")
        return False

def main():
    """主函数"""
    # 创建音频目录
    audio_dir = os.path.join(ROOT_DIR, 'data', 'test_audio')
    os.makedirs(audio_dir, exist_ok=True)
    
    # 下载测试音频
    success_count = 0
    for resource in TEST_AUDIO_RESOURCES:
        output_path = os.path.join(audio_dir, resource['name'])
        
        # 如果文件已存在，跳过
        if os.path.exists(output_path):
            logger.info(f"文件已存在，跳过: {output_path}")
            success_count += 1
            continue
            
        logger.info(f"下载: {resource['name']} - {resource['url']}")
        if download_file(resource['url'], output_path):
            success_count += 1
    
    logger.info(f"下载完成: {success_count}/{len(TEST_AUDIO_RESOURCES)} 个文件")

if __name__ == "__main__":
    main()
