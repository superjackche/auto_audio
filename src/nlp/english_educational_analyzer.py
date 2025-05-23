#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英文教育内容分析器 - 专门用于分析英文教育内容的意识形态风险
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# 导入现有的分析器以继承其功能
from src.nlp.educational_analyzer import EducationalSemanticAnalyzer

logger = logging.getLogger(__name__)

class EnglishEducationalAnalyzer:
    """英文教育内容分析器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化英文教育内容分析器"""
        self.config = config or {}
        self._load_english_keywords()
        self._load_english_risk_patterns()
        self._initialize_scoring_system()
        
        # 历史分析记录
        self.analysis_history = []
        self.max_history = 50
        
    def _load_english_keywords(self):
        """加载英文教育相关关键词库"""
        # 正面教育关键词
        self.positive_educational_keywords = {
            # 学术相关
            'academic freedom', 'critical thinking', 'scholarly research', 'scientific method',
            'innovation', 'academic integrity', 'evidence-based', 'peer review',
            
            # 教育理念
            'holistic education', 'student-centered', 'lifelong learning', 'inclusive education',
            'quality education', 'educational reform', 'talent development', 'cultural heritage',
            
            # 国际合作
            'international cooperation', 'cultural exchange', 'mutual learning', 'shared future',
            'global perspective', 'cross-cultural', 'collaborative research', 'educational partnership'
        }
        
        # 风险关键词
        self.risk_keywords = {
            # 政治制度相关
            'political system': {'level': 'medium', 'weight': 7},
            'system superiority': {'level': 'medium', 'weight': 6},
            'democracy freedom': {'level': 'medium', 'weight': 5},
            'human rights': {'level': 'medium', 'weight': 5},
            
            # 历史相关
            'historical nihilism': {'level': 'high', 'weight': 9},
            'cultural revolution': {'level': 'high', 'weight': 8},
            'historical reflection': {'level': 'medium', 'weight': 6},
            
            # 西方价值观相关
            'western values': {'level': 'high', 'weight': 8},
            'universal values': {'level': 'high', 'weight': 8},
            'individualism': {'level': 'medium', 'weight': 5},
            'liberalism': {'level': 'medium', 'weight': 6},
            
            # 宗教相关
            'religious belief': {'level': 'medium', 'weight': 6},
            'christianity': {'level': 'medium', 'weight': 5},
            'religious advocacy': {'level': 'high', 'weight': 8},
            
            # 特殊敏感词
            'taiwan independence': {'level': 'critical', 'weight': 15},
            'hong kong independence': {'level': 'critical', 'weight': 15},
            'tibet independence': {'level': 'critical', 'weight': 15},
            'xinjiang issues': {'level': 'high', 'weight': 10},
            'one country two systems': {'level': 'medium', 'weight': 6},
            
            # 更多英文特有关键词
            'authoritarian': {'level': 'high', 'weight': 7},
            'oppression': {'level': 'high', 'weight': 8},
            'censorship': {'level': 'high', 'weight': 7},
            'propaganda': {'level': 'medium', 'weight': 6},
            'surveillance': {'level': 'medium', 'weight': 6},
        }
        
        # 加载外部英文关键词文件
        self._load_external_keywords()
        
        # 学术争议词汇（需要上下文分析）
        self.controversial_academic_terms = {
            'academic freedom', 'freedom of speech', 'free press', 'critical spirit', 
            'independent thinking', 'questioning authority', 'academic independence', 
            'value neutrality', 'objectivity', 'diversity'
        }
    
    def _load_external_keywords(self):
        """加载外部英文关键词文件"""
        english_keywords_file = os.path.join("config", "english_keywords.txt")
        if os.path.exists(english_keywords_file):
            try:
                with open(english_keywords_file, 'r', encoding='utf-8') as f:
                    current_category = None
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            # 空行或注释行用于分类标记
                            if line.startswith('# '):
                                current_category = line[2:].strip().lower()
                            continue
                        
                        parts = line.split(',')
                        if len(parts) >= 2:
                            keyword = parts[0].strip()
                            weight = float(parts[1].strip())
                            
                            # 根据权重和类别确定风险级别
                            level = 'medium'
                            if weight >= 0.9:
                                level = 'critical'
                            elif weight >= 0.8:
                                level = 'high'
                            elif weight >= 0.5:
                                level = 'medium'
                            else:
                                level = 'low'
                            
                            self.risk_keywords[keyword] = {'level': level, 'weight': weight * 10}
                
                logger.info(f"成功加载外部英文关键词")
            except Exception as e:
                logger.error(f"加载外部英文关键词失败: {e}")
    
    def _load_english_risk_patterns(self):
        """加载英文风险模式"""
        self.risk_patterns = {
            # 比较模式（暗示制度对比）
            'comparison_patterns': [
                r'(china|chinese).*?(not as|inferior to|worse than).*(western|europe|america|developed countries)',
                r'(western|europe|america|developed countries).*?(advanced|superior|better).*(china|chinese)',
                r'why.*(china|we).*?cannot.*?(like|learn from).*?',
            ],
            
            # 暗示模式
            'implication_patterns': [
                r'(true|real|genuine).*(democracy|freedom)',
                r'(lacks|absence of|without|no).*(freedom|democracy|rights)',
                r'(human rights|democracy|freedom).*?(issues|problems|concerns)',
            ],
            
            # 质疑与批判模式
            'questioning_patterns': [
                r'(question|challenge|doubt).*?(official|authorities|government|leadership)',
                r'(truth|facts).*?(covered up|hidden|concealed)',
                r'(history|historical).*?(rewritten|falsified|distorted)',
            ],
            
            # 分裂话题相关模式
            'separation_patterns': [
                r'(taiwan|hong kong|tibet|xinjiang).*?(independence|separate|autonomous)',
                r'(one country two systems).*?(failure|failed)',
                r'(self-determination|self-governance).*?(right|entitled)',
            ]
        }
    
    def _initialize_scoring_system(self):
        """初始化风险评分系统"""
        # 风险权重配置
        self.scoring_weights = {
            'keyword_weight': 1.0,         # 关键词权重
            'pattern_weight': 1.5,         # 风险模式权重
            'context_weight': 0.8,         # 上下文权重
            'segment_weight': 1.2,         # 文本片段权重
            'controversial_term_weight': 0.6  # 争议性术语权重
        }
        
        # 风险级别阈值
        self.risk_thresholds = {
            'critical': 80,   # 极高风险
            'high': 60,       # 高风险
            'medium': 40,     # 中等风险
            'low': 20,        # 低风险
            'safe': 0         # 安全
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        分析英文文本内容的风险
        
        Args:
            text: 要分析的英文文本
            
        Returns:
            Dict: 分析结果字典
        """
        # 进行文本预处理
        processed_text = self._preprocess_text(text)
        
        # 分词
        words = self._tokenize(processed_text)
        
        # 关键词分析
        keyword_results = self._analyze_keywords(processed_text, words)
        
        # 模式分析
        pattern_results = self._analyze_patterns(processed_text)
        
        # 上下文分析
        context_results = self._analyze_context(processed_text)
        
        # 计算总体风险评分
        risk_score = self._calculate_risk_score(keyword_results, pattern_results, context_results)
        
        # 确定风险级别
        risk_level = self._determine_risk_level(risk_score)
        
        # 获取关键片段
        key_segments = self._extract_key_segments(processed_text, keyword_results, pattern_results)
        
        # 提取出现的敏感关键词
        detected_keywords = keyword_results.get('detected_keywords', [])
        
        # 构造分析结果
        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'text': text,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'detected_keywords': detected_keywords,
            'key_segments': key_segments,
            'language': 'english'
        }
        
        # 保存到历史记录
        self._update_history(analysis_result)
        
        return analysis_result
    
    def _preprocess_text(self, text: str) -> str:
        """
        预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            str: 预处理后的文本
        """
        # 转为小写
        text = text.lower()
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 清理标点符号
        text = re.sub(r'[^\w\s]', ' ', text)
        
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        文本分词
        
        Args:
            text: 预处理后的文本
            
        Returns:
            List[str]: 分词结果列表
        """
        # 简单空格分词
        return text.split()
    
    def _analyze_keywords(self, text: str, words: List[str]) -> Dict[str, Any]:
        """
        分析关键词
        
        Args:
            text: 预处理后的文本
            words: 分词结果
            
        Returns:
            Dict: 关键词分析结果
        """
        detected_keywords = []
        total_weight = 0
        
        # 扫描单词
        for word in words:
            if word in self.risk_keywords:
                keyword_info = self.risk_keywords[word]
                detected_keywords.append({
                    'keyword': word,
                    'level': keyword_info['level'],
                    'weight': keyword_info['weight']
                })
                total_weight += keyword_info['weight']
        
        # 扫描词组（N-gram）
        for n in range(2, 5):  # 2~4词组合
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                if phrase in self.risk_keywords:
                    keyword_info = self.risk_keywords[phrase]
                    detected_keywords.append({
                        'keyword': phrase,
                        'level': keyword_info['level'],
                        'weight': keyword_info['weight']
                    })
                    # 词组权重更高
                    total_weight += keyword_info['weight'] * 1.2
        
        # 根据上下文分析争议性术语
        for term in self.controversial_academic_terms:
            if term in text:
                # 这里可以添加更复杂的上下文分析逻辑
                # 简单起见，这里只做基本检测
                context_risk = self._analyze_controversial_term_context(text, term)
                if context_risk > 0:
                    detected_keywords.append({
                        'keyword': term,
                        'level': 'context_dependent',
                        'weight': context_risk
                    })
                    total_weight += context_risk
        
        return {
            'detected_keywords': detected_keywords,
            'keyword_weight_sum': total_weight
        }
    
    def _analyze_controversial_term_context(self, text: str, term: str) -> float:
        """
        分析争议性术语的上下文风险
        
        Args:
            text: 文本内容
            term: 争议性术语
            
        Returns:
            float: 风险权重
        """
        # 发现术语所在位置
        term_index = text.find(term)
        if term_index == -1:
            return 0
            
        # 获取术语前后的上下文（前后各50个字符）
        start = max(0, term_index - 50)
        end = min(len(text), term_index + len(term) + 50)
        context = text[start:end]
        
        # 判断上下文是否包含风险词
        risk_score = 0
        for risk_word, info in self.risk_keywords.items():
            if risk_word in context:
                risk_score += info['weight'] * 0.3
        
        return risk_score
    
    def _analyze_patterns(self, text: str) -> Dict[str, Any]:
        """
        分析风险模式
        
        Args:
            text: 预处理后的文本
            
        Returns:
            Dict: 模式分析结果
        """
        detected_patterns = []
        total_pattern_weight = 0
        
        # 检查各类风险模式
        for pattern_type, patterns in self.risk_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    matched_text = match.group(0)
                    weight = 0
                    
                    # 根据模式类型分配权重
                    if pattern_type == 'comparison_patterns':
                        weight = 8
                    elif pattern_type == 'implication_patterns':
                        weight = 7
                    elif pattern_type == 'questioning_patterns':
                        weight = 6
                    elif pattern_type == 'separation_patterns':
                        weight = 10
                    
                    detected_patterns.append({
                        'pattern_type': pattern_type,
                        'matched_text': matched_text,
                        'weight': weight
                    })
                    
                    total_pattern_weight += weight
        
        return {
            'detected_patterns': detected_patterns,
            'pattern_weight_sum': total_pattern_weight
        }
    
    def _analyze_context(self, text: str) -> Dict[str, Any]:
        """
        分析文本上下文
        
        Args:
            text: 预处理后的文本
            
        Returns:
            Dict: 上下文分析结果
        """
        # 这里可以实现更复杂的上下文分析
        # 例如分析句子间的逻辑关系、语义连贯性等
        
        # 简单的实现是检查特定的上下文标记词
        context_markers = {
            'compare': ['compared to', 'unlike', 'in contrast', 'whereas', 'while'],
            'negative': ['problem', 'issue', 'concern', 'unfortunately', 'however'],
            'critical': ['must', 'should', 'have to', 'need to', 'required']
        }
        
        context_scores = defaultdict(float)
        
        for category, markers in context_markers.items():
            for marker in markers:
                count = text.count(marker)
                if count > 0:
                    if category == 'compare':
                        context_scores['compare'] += count * 2.0
                    elif category == 'negative':
                        context_scores['negative'] += count * 1.5
                    elif category == 'critical':
                        context_scores['critical'] += count * 1.0
        
        # 计算总上下文风险分数
        total_context_weight = sum(context_scores.values())
        
        return {
            'context_scores': dict(context_scores),
            'context_weight_sum': total_context_weight
        }
    
    def _calculate_risk_score(self, keyword_results, pattern_results, context_results) -> float:
        """
        计算总体风险评分
        
        Args:
            keyword_results: 关键词分析结果
            pattern_results: 模式分析结果
            context_results: 上下文分析结果
            
        Returns:
            float: 风险评分
        """
        # 获取各部分权重和
        keyword_weight = keyword_results.get('keyword_weight_sum', 0)
        pattern_weight = pattern_results.get('pattern_weight_sum', 0)
        context_weight = context_results.get('context_weight_sum', 0)
        
        # 应用权重系数
        weighted_keyword = keyword_weight * self.scoring_weights['keyword_weight']
        weighted_pattern = pattern_weight * self.scoring_weights['pattern_weight']
        weighted_context = context_weight * self.scoring_weights['context_weight']
        
        # 计算总评分
        total_score = weighted_keyword + weighted_pattern + weighted_context
        
        # 标准化分数到0-100范围
        normalized_score = min(100, total_score)
        
        return round(normalized_score, 1)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """
        根据风险评分确定风险级别
        
        Args:
            risk_score: 风险评分
            
        Returns:
            str: 风险级别
        """
        if risk_score >= self.risk_thresholds['critical']:
            return 'critical'
        elif risk_score >= self.risk_thresholds['high']:
            return 'high'
        elif risk_score >= self.risk_thresholds['medium']:
            return 'medium'
        elif risk_score >= self.risk_thresholds['low']:
            return 'low'
        else:
            return 'safe'
    
    def _extract_key_segments(self, text: str, keyword_results, pattern_results) -> List[Dict[str, Any]]:
        """
        提取关键文本片段
        
        Args:
            text: 处理后的文本
            keyword_results: 关键词分析结果
            pattern_results: 模式分析结果
            
        Returns:
            List: 关键片段列表
        """
        key_segments = []
        
        # 简单起见，这里只基于关键词和模式提取
        # 1. 提取包含关键词的句子
        sentences = re.split(r'[.!?]', text)
        
        for keyword_info in keyword_results.get('detected_keywords', []):
            keyword = keyword_info['keyword']
            for sentence in sentences:
                if keyword in sentence:
                    key_segments.append({
                        'text': sentence.strip(),
                        'type': 'keyword',
                        'keyword': keyword,
                        'risk_level': keyword_info['level']
                    })
                    break  # 每个关键词只取一个句子
        
        # 2. 提取包含风险模式的句子
        for pattern_info in pattern_results.get('detected_patterns', []):
            matched_text = pattern_info['matched_text']
            for sentence in sentences:
                if matched_text in sentence:
                    key_segments.append({
                        'text': sentence.strip(),
                        'type': 'pattern',
                        'pattern_type': pattern_info['pattern_type'],
                        'matched_text': matched_text
                    })
                    break  # 每个模式只取一个句子
        
        return key_segments
    
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
            'most_frequent_keywords': most_frequent_keywords
        }
