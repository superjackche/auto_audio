#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音频文件分析工具
用于分析本地音频文件（mp3等格式）并进行风险评估
支持中英文自动检测和混合分析
"""

import os
import sys
import numpy as np
import librosa
import logging
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.audio.speech_to_text import SpeechToText
from src.nlp.unified_language_analyzer import UnifiedLanguageAnalyzer
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class AudioFileAnalyzer:
    """音频文件分析器 - 支持中英文自动检测"""
    
    def __init__(self):
        """初始化分析器"""
        self.stt = SpeechToText()
        
        # 使用统一语言分析器替代原来的语义分析器
        self.language_analyzer = UnifiedLanguageAnalyzer()
        
        # 加载配置
        config_loader = ConfigLoader()
        config = config_loader.load_config()
        
        self.sample_rate = 16000  # Whisper需要16kHz采样率
        
    def load_audio_file(self, file_path):
        """
        加载音频文件
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            tuple: (音频数据, 采样率)
        """
        try:
            # librosa可以处理多种格式的音频文件，包括mp3
            audio_data, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return audio_data, sr
        except Exception as e:
            logger.error(f"加载音频文件失败: {str(e)}")
            return None, None
    
    def analyze(self, file_path):
        """
        分析音频文件
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            dict: 分析结果
        """
        logger.info(f"开始分析音频文件: {file_path}")
        
        audio_data, sr = self.load_audio_file(file_path)
        if audio_data is None:
            return {'error': '无法加载音频文件'}
        
        # 音频转文本 (语言自动检测)
        text = self.stt.transcribe(audio_data)
        if not text:
            logger.warning(f"语音识别失败或未检测到文本: {file_path}")
            return {'error': '语音识别失败或未检测到文本'}
        
        logger.info(f"语音识别结果 (前100字符): {text[:100]}...")

        # 统一语言分析器进行分析
        unified_analysis_result = self.language_analyzer.analyze_text(text)

        # 定义语言和风险等级的映射关系
        risk_level_map = {
            'low': "低风险",
            'medium': "中风险",
            'high': "高风险",
            'critical': "极高风险",
        }
        language_map = {
            'zh': "中文",
            'en': "英文",
            'mixed': "混合语言",
            'unknown': "未知语言"
        }
        segment_language_map = {
            'zh': "中文",
            'en': "英文",
            'unknown': "未知"
        }

        risk_level_code = unified_analysis_result.get('risk_level', 'unknown_level')
        detected_lang_code = unified_analysis_result.get('detected_language', 'unknown')

        # 从关键词提取风险因素
        risk_factors_list = []
        if unified_analysis_result.get('weighted_keywords'):
            for kw_info in unified_analysis_result['weighted_keywords']:
                risk_factors_list.append(
                    f"{kw_info.get('term', 'N/A')} (分数: {kw_info.get('score', 0):.2f}, 分类: {kw_info.get('category', 'N/A')})"
                )

        # 处理分段信息以供显示
        processed_segments = []
        if unified_analysis_result.get('segments'):
            for seg in unified_analysis_result['segments']:
                processed_seg = seg.copy()
                processed_seg['language_name'] = segment_language_map.get(seg.get('language'), seg.get('language'))
                processed_segments.append(processed_seg)
        
        detailed_analysis_output = unified_analysis_result.copy()
        if 'segments' in detailed_analysis_output:
            detailed_analysis_output['segments'] = processed_segments

        result = {
            'text': text,
            'risk_score': unified_analysis_result.get('overall_risk_score', 0.0),
            'risk_level': risk_level_map.get(risk_level_code, f"未知级别 ({risk_level_code})"),
            'detected_language': language_map.get(detected_lang_code, f"未知 ({detected_lang_code})"),
            'risk_factors': risk_factors_list,
            'detailed_analysis': detailed_analysis_output
        }
        
        logger.info(f"分析完成: 风险等级 {result['risk_level']}, 检测到语言: {result['detected_language']}")
        return result

def main():
    """命令行入口"""
    if len(sys.argv) < 2:
        print("用法: python analyze_audio_file.py <音频文件路径>")
        return 1
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        return 1
    
    # 配置日志
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    analyzer = AudioFileAnalyzer()
    result = analyzer.analyze(file_path) # 移除了 language 参数
    
    print("\n===== 音频分析结果 =====\n")
    print(f"识别文本: {result.get('text', '无法识别')}")
    print(f"风险等级: {result.get('risk_level', '未知')}")
    print(f"风险分数: {result.get('risk_score', 0):.2f}")
    print(f"检测到语言: {result.get('detected_language', '未知')}") # 新增
    
    if result.get('risk_factors'):
        print("\n风险因素:")
        for factor in result['risk_factors']:
            print(f"- {factor}")
    
    # 打印详细分析（可选）
    detailed_analysis = result.get('detailed_analysis')
    if detailed_analysis:
        print("\n详细分析:")
        if detailed_analysis.get('segments'):
            print("  片段分析:")
            for i, segment in enumerate(detailed_analysis['segments']):
                print(f"    片段 {i+1} ({segment.get('language_name', '未知语言')}): "{segment.get('text', '')[:50]}..." - 风险分数: {segment.get('risk_score', 0):.2f}")
                if segment.get('keywords_found'):
                    print("      关键词:")
                    for kw in segment['keywords_found']:
                        print(f"        - {kw.get('term', 'N/A')} (分数: {kw.get('score',0):.2f}, 分类: {kw.get('category', 'N/A')})")
        
        if detailed_analysis.get('explanation'):
            print(f"  说明: {detailed_analysis.get('explanation')}")
            
    print("\n=========================\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
