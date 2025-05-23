#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统一语言分析器 - 同时支持中英文分析，用于课堂语义行为实时分析
"""

import os
import re
import json
import logging
import langdetect
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

# 导入现有的分析器以继承其功能
from src.nlp.educational_analyzer import EducationalSemanticAnalyzer
from src.nlp.english_educational_analyzer import EnglishEducationalAnalyzer

logger = logging.getLogger(__name__)

class UnifiedLanguageAnalyzer:
    """统一语言分析器 - 同时支持中英文"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化统一语言分析器"""
        self.config = config or {}
        
        # 初始化中文分析器
        self.chinese_analyzer = EducationalSemanticAnalyzer(config)
        
        # 初始化英文分析器
        self.english_analyzer = EnglishEducationalAnalyzer(config)
        
        # 历史分析记录
        self.analysis_history = []
        self.max_history = 50
        
        logger.info("统一语言分析器初始化完成")
    
    def detect_language(self, text: str) -> str:
        """
        检测文本语言
        
        Args:
            text: 要检测的文本
            
        Returns:
            str: 检测到的语言代码 ('zh', 'en', 'mixed')
        """
        if not text or len(text.strip()) < 10:
            return 'unknown'
            
        try:
            # 检测中文字符比例
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            text_length = len(text.strip())
            chinese_ratio = chinese_chars / text_length if text_length > 0 else 0
            
            # 使用langdetect进行语言检测
            detected = langdetect.detect(text)
            
            # 综合判断
            if chinese_ratio > 0.3:
                if chinese_ratio > 0.7:
                    return 'zh'  # 主要是中文
                else:
                    return 'mixed'  # 中英混合，中文占比较大
            elif detected == 'zh-cn' or detected == 'zh-tw':
                return 'zh'
            elif detected == 'en':
                return 'en'
            else:
                # 其他情况，用简单规则判断
                if chinese_chars > 10:
                    return 'mixed'  # 包含足够的中文字符，视为混合
                else:
                    return 'en'  # 默认英文
        except Exception as e:
            logger.warning(f"语言检测失败: {e}, 默认使用中文")
            return 'zh'  # 出错时默认中文
    
    def split_mixed_text(self, text: str) -> Tuple[str, str]:
        """
        拆分混合语言文本为中文和英文部分
        
        Args:
            text: 混合语言文本
            
        Returns:
            Tuple[str, str]: (中文部分, 英文部分)
        """
        # 简单分隔句子
        sentences = re.split(r'([.!?。！？\n])', text)
        
        chinese_parts = []
        english_parts = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # 检测单个句子的语言
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', sentence))
            sentence_length = len(sentence.strip())
            
            if sentence_length > 0:
                chinese_ratio = chinese_chars / sentence_length
                
                if chinese_ratio > 0.5:
                    chinese_parts.append(sentence)
                else:
                    english_parts.append(sentence)
        
        chinese_text = ''.join(chinese_parts)
        english_text = ''.join(english_parts)
        
        return chinese_text, english_text
    
    def analyze_text(self, text: str, force_language: Optional[str] = None) -> Dict[str, Any]:
        """
        分析文本内容，自动检测语言并使用相应的分析器
        
        Args:
            text: 要分析的文本
            force_language: 强制使用指定的语言分析器 ('zh', 'en', None)
            
        Returns:
            Dict: 分析结果字典
        """
        if not text or len(text.strip()) == 0:
            return {
                'timestamp': datetime.now().isoformat(),
                'text': text,
                'risk_score': 0,
                'risk_level': 'safe',
                'detected_keywords': [],
                'key_segments': [],
                'language': 'unknown',
                'error': '文本内容为空'
            }
        
        # 检测语言
        if force_language:
            language = force_language
        else:
            language = self.detect_language(text)
        
        logger.info(f"检测到语言: {language}")
        
        # 根据语言选择分析器
        if language == 'zh':
            # 纯中文
            result = self.chinese_analyzer.analyze_text(text)
            result['language'] = 'zh'
            
        elif language == 'en':
            # 纯英文
            result = self.english_analyzer.analyze_text(text)
            result['language'] = 'en'
            
        elif language == 'mixed':
            # 混合语言，分别分析后合并结果
            chinese_text, english_text = self.split_mixed_text(text)
            
            # 分别分析
            if chinese_text:
                chinese_result = self.chinese_analyzer.analyze_text(chinese_text)
            else:
                chinese_result = {'risk_score': 0, 'detected_keywords': [], 'key_segments': []}
                
            if english_text:
                english_result = self.english_analyzer.analyze_text(english_text)
            else:
                english_result = {'risk_score': 0, 'detected_keywords': [], 'key_segments': []}
            
            # 合并结果
            # 风险分数取两者中较高值，并按各自文本长度比例加权
            chinese_length = len(chinese_text.strip())
            english_length = len(english_text.strip())
            total_length = chinese_length + english_length
            
            if total_length > 0:
                chinese_weight = chinese_length / total_length
                english_weight = english_length / total_length
                
                risk_score = (chinese_result['risk_score'] * chinese_weight + 
                             english_result['risk_score'] * english_weight)
            else:
                risk_score = 0
            
            # 确定风险级别
            if risk_score >= 80:
                risk_level = 'critical'
            elif risk_score >= 60:
                risk_level = 'high'
            elif risk_score >= 40:
                risk_level = 'medium'
            elif risk_score >= 20:
                risk_level = 'low'
            else:
                risk_level = 'safe'
            
            # 合并关键词和关键片段
            detected_keywords = chinese_result.get('detected_keywords', []) + english_result.get('detected_keywords', [])
            key_segments = chinese_result.get('key_segments', []) + english_result.get('key_segments', [])
            
            # 构造混合结果
            result = {
                'timestamp': datetime.now().isoformat(),
                'text': text,
                'risk_score': round(risk_score, 1),
                'risk_level': risk_level,
                'detected_keywords': detected_keywords,
                'key_segments': key_segments,
                'language': 'mixed',
                'components': {
                    'chinese': {
                        'text': chinese_text,
                        'risk_score': chinese_result['risk_score'],
                        'weight': round(chinese_weight, 2) if total_length > 0 else 0
                    },
                    'english': {
                        'text': english_text,
                        'risk_score': english_result['risk_score'],
                        'weight': round(english_weight, 2) if total_length > 0 else 0
                    }
                }
            }
        
        else:
            # 未知语言，默认使用中文分析器
            result = self.chinese_analyzer.analyze_text(text)
            result['language'] = 'unknown'
        
        # 保存到历史记录
        self._update_history(result)
        
        return result
    
    def _update_history(self, analysis_result: Dict[str, Any]):
        """
        更新历史分析记录
        
        Args:
            analysis_result: 当前分析结果
        """
        self.analysis_history.append(analysis_result)
        
        # 限制历史记录数量
        if len(self.analysis_history) > self.max_history:
            self.analysis_history = self.analysis_history[-self.max_history:]
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        获取分析结果摘要
        
        Returns:
            Dict: 分析摘要
        """
        if not self.analysis_history:
            return {
                'total_analyses': 0,
                'average_risk_score': 0,
                'risk_level_counts': {},
                'language_counts': {},
                'most_frequent_keywords': []
            }
        
        # 计算平均风险分数
        risk_scores = [result['risk_score'] for result in self.analysis_history]
        avg_risk_score = sum(risk_scores) / len(risk_scores)
        
        # 统计风险级别
        risk_levels = [result['risk_level'] for result in self.analysis_history]
        risk_level_counts = {}
        for level in ['critical', 'high', 'medium', 'low', 'safe']:
            risk_level_counts[level] = risk_levels.count(level)
        
        # 统计语言分布
        languages = [result.get('language', 'unknown') for result in self.analysis_history]
        language_counts = {}
        for lang in ['zh', 'en', 'mixed', 'unknown']:
            language_counts[lang] = languages.count(lang)
        
        # 统计最常见关键词
        keyword_counter = defaultdict(int)
        for result in self.analysis_history:
            for keyword_info in result.get('detected_keywords', []):
                keyword_counter[keyword_info['keyword']] += 1
        
        # 获取前10个最常见关键词
        most_frequent_keywords = sorted(
            [{'keyword': k, 'count': v} for k, v in keyword_counter.items()],
            key=lambda x: x['count'],
            reverse=True
        )[:10]
        
        return {
            'total_analyses': len(self.analysis_history),
            'average_risk_score': round(avg_risk_score, 2),
            'risk_level_counts': risk_level_counts,
            'language_counts': language_counts,
            'most_frequent_keywords': most_frequent_keywords
        }
