#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
双语文本分析模块 - 支持中英文
"""

import os
import re
import json
import logging
import numpy as np
from datetime import datetime
from collections import deque

# 适配可能使用的NLP库
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    
try:
    import torch
    from transformers import BertTokenizer, BertModel
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

logger = logging.getLogger(__name__)

class BilingualTextAnalyzer:
    """双语文本分析器，支持中英文"""
    
    def __init__(self, config=None):
        """
        初始化双语文本分析器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.embedding_model_name = self.config.get('embedding_model', 'bert-base-multilingual-cased')
        self.tokenizer_name = self.config.get('tokenizer', 'jieba')
        
        # 存储历史文本用于上下文分析
        self.history = deque(maxlen=self.config.get('logic_analysis', {}).get('history_window', 10))
        
        # 加载关键词列表
        self.keywords = self._load_keywords()
        self.tokenizer = self._initialize_tokenizer()
        self.embedding_model = self._initialize_embedding_model()
        
        logger.info("双语文本分析器初始化完成")
    
    def _load_keywords(self):
        """
        加载关键词列表
        
        Returns:
            dict: 关键词及其权重字典
        """
        keywords = {}
        
        # 加载中文关键词
        chinese_keyword_file = self.config.get('keyword_detection', {}).get('keyword_file', 'config/keywords.txt')
        if chinese_keyword_file and os.path.exists(chinese_keyword_file):
            try:
                with open(chinese_keyword_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        # 跳过注释和空行
                        if not line or line.startswith('#'):
                            continue
                            
                        parts = line.split(',')
                        if len(parts) >= 2:
                            word = parts[0].strip()
                            weight = float(parts[1].strip())
                            keywords[word] = weight
                
                logger.info(f"成功加载 {len(keywords)} 个中文关键词")
            except Exception as e:
                logger.exception(f"加载中文关键词文件失败: {str(e)}")
        else:
            logger.warning(f"中文关键词文件不存在: {chinese_keyword_file}")
            
        # 加载英文关键词
        english_keyword_file = self.config.get('keyword_detection', {}).get('english_keyword_file', 'config/english_keywords.txt')
        if english_keyword_file and os.path.exists(english_keyword_file):
            try:
                english_keywords_count = 0
                with open(english_keyword_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        # 跳过注释和空行
                        if not line or line.startswith('#'):
                            continue
                            
                        parts = line.split(',')
                        if len(parts) >= 2:
                            word = parts[0].strip()
                            weight = float(parts[1].strip())
                            keywords[word] = weight
                            english_keywords_count += 1
                
                logger.info(f"成功加载 {english_keywords_count} 个英文关键词")
            except Exception as e:
                logger.exception(f"加载英文关键词文件失败: {str(e)}")
        else:
            logger.warning(f"英文关键词文件不存在: {english_keyword_file}")
        
        logger.info(f"总计加载 {len(keywords)} 个关键词")
        return keywords
    
    def _initialize_tokenizer(self):
        """
        初始化分词器
        
        Returns:
            object: 分词器对象
        """
        if self.tokenizer_name == 'jieba':
            if not JIEBA_AVAILABLE:
                logger.error("未安装jieba模块，请使用 'pip install jieba'")
                return None
                
            # 加载用户词典（如果有）
            user_dict = self.config.get('user_dict')
            if user_dict and os.path.exists(user_dict):
                jieba.load_userdict(user_dict)
                
            logger.info("jieba分词器初始化完成")
            return jieba
        elif self.tokenizer_name == 'bert':
            if not BERT_AVAILABLE:
                logger.error("未安装transformers模块，请使用 'pip install transformers'")
                return None
                
            try:
                tokenizer = BertTokenizer.from_pretrained(self.embedding_model_name)
                logger.info(f"BERT分词器初始化完成: {self.embedding_model_name}")
                return tokenizer
            except Exception as e:
                logger.exception(f"初始化BERT分词器失败: {str(e)}")
                return None
        else:
            logger.error(f"不支持的分词器: {self.tokenizer_name}")
            return None
    
    def _initialize_embedding_model(self):
        """
        初始化嵌入模型
        
        Returns:
            object: 嵌入模型对象
        """
        if not BERT_AVAILABLE:
            logger.error("未安装transformers或torch模块")
            return None
            
        try:
            model = BertModel.from_pretrained(self.embedding_model_name)
            logger.info(f"嵌入模型初始化完成: {self.embedding_model_name}")
            return model
        except Exception as e:
            logger.exception(f"初始化嵌入模型失败: {str(e)}")
            return None
    
    def analyze(self, text):
        """
        分析文本
        
        Args:
            text: 待分析的文本
            
        Returns:
            dict: 分析结果
        """
        if not text or not isinstance(text, str):
            logger.warning(f"无效的输入文本: {text}")
            return {
                'text': text,
                'timestamp': datetime.now().isoformat(),
                'risk_score': 0.0,
                'risk_factors': [],
                'keyword_matches': [],
                'segments': [],
                'consistency_score': 1.0,
                'embeddings': None
            }
        
        # 添加到历史记录
        self.history.append(text)
        
        # 检测语言
        is_chinese = self._is_chinese_text(text)
        is_english = self._is_english_text(text)
        language = "chinese" if is_chinese else "english" if is_english else "mixed"
        
        # 执行各种分析
        keyword_results = self._analyze_keywords(text)
        semantic_results = self._analyze_semantics(text, language)
        logic_results = self._analyze_logic_consistency()
        
        # 合并结果
        results = {
            'text': text,
            'language': language,
            'timestamp': datetime.now().isoformat(),
            'segments': self._segment_text(text, language),
            'keyword_matches': keyword_results['matches'],
            'keyword_score': keyword_results['score'],
            'semantic_score': semantic_results['score'],
            'consistency_score': logic_results['score'],
            'risk_factors': keyword_results['risk_factors'] + semantic_results['risk_factors'] + logic_results['risk_factors'],
            'embeddings': semantic_results.get('embeddings')
        }
        
        # 计算综合风险分数
        kw_weight = self.config.get('keyword_detection', {}).get('weight', 0.4)
        sem_weight = self.config.get('semantic_analysis', {}).get('weight', 0.3)
        log_weight = self.config.get('logic_analysis', {}).get('weight', 0.3)
        
        risk_score = (
            kw_weight * keyword_results['score'] +
            sem_weight * semantic_results['score'] +
            log_weight * logic_results['score']
        )
        
        results['risk_score'] = min(1.0, max(0.0, risk_score)) * 100.0  # 转换为百分比
        
        return results
    def _is_english_text(self, text):
        """
        判断文本是否主要是英文
        
        Args:
            text: 待检测的文本
            
        Returns:
            bool: 是否主要是英文
        """
        if not text:
            return False
            
        # 英文字符比例
        english_chars = sum(1 for c in text if ord('a') <= ord(c.lower()) <= ord('z'))
        total_chars = len(text.strip())
        
        return total_chars > 0 and english_chars / total_chars > 0.5
    
    def _is_chinese_text(self, text):
        """
        判断文本是否主要是中文
        
        Args:
            text: 待检测的文本
            
        Returns:
            bool: 是否主要是中文
        """
        if not text:
            return False
            
        # 中文字符比例
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text.strip())
        
        return total_chars > 0 and chinese_chars / total_chars > 0.5
    
    def _segment_text(self, text, language=None):
        """
        对文本进行分词
        
        Args:
            text: 待分词的文本
            language: 文本语言
            
        Returns:
            list: 分词结果
        """
        if language is None:
            language = "chinese" if self._is_chinese_text(text) else "english" if self._is_english_text(text) else "mixed"
            
        if language == "chinese" and self.tokenizer_name == 'jieba' and self.tokenizer:
            return list(self.tokenizer.cut(text))
        elif self.tokenizer_name == 'bert' and self.tokenizer:
            return self.tokenizer.tokenize(text)
        else:
            # 对英文文本使用简单的空格分词
            return text.split()
    
    def _analyze_keywords(self, text):
        """
        关键词分析
        
        Args:
            text: 待分析的文本
            
        Returns:
            dict: 关键词分析结果
        """
        if not self.config.get('keyword_detection', {}).get('enabled', True):
            return {'score': 0.0, 'matches': [], 'risk_factors': []}
        
        matches = []
        risk_factors = []
        score = 0.0
        
        # 预处理文本 - 转为小写，方便检测
        text_lower = text.lower()
        
        # 检测语言类型
        is_english = self._is_english_text(text)
        is_chinese = self._is_chinese_text(text)
        
        # 遍历关键词进行匹配
        for keyword, weight in self.keywords.items():
            # 对英文关键词使用单词边界匹配，对中文关键词直接匹配
            if self._is_english_text(keyword) and is_english:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches_count = len(re.findall(pattern, text_lower, re.IGNORECASE))
            elif self._is_chinese_text(keyword) and is_chinese:
                # 中文关键词直接匹配
                matches_count = text_lower.count(keyword.lower())
            else:
                # 对于混合文本，使用简单的字符串搜索
                matches_count = text_lower.count(keyword.lower())
            
            if matches_count > 0:
                match_info = {
                    'keyword': keyword,
                    'count': matches_count,
                    'weight': weight
                }
                matches.append(match_info)
                
                # 将高风险关键词添加到风险因素
                if weight >= 0.7:
                    risk_factor = {
                        'type': 'keyword',
                        'description': f"检测到高风险关键词: {keyword}",
                        'score': weight,
                        'evidence': keyword
                    }
                    risk_factors.append(risk_factor)
                
                # 累加风险分数
                score += weight * matches_count
        
        # 归一化分数
        max_score = self.config.get('keyword_detection', {}).get('max_score', 3.0)
        normalized_score = min(1.0, score / max_score)
        
        return {
            'score': normalized_score,
            'matches': matches,
            'risk_factors': risk_factors
        }
    
    def _analyze_semantics(self, text, language=None):
        """
        语义分析
        
        Args:
            text: 待分析的文本
            language: 文本语言
            
        Returns:
            dict: 语义分析结果
        """
        if not self.config.get('semantic_analysis', {}).get('enabled', True):
            return {'score': 0.0, 'risk_factors': [], 'embeddings': None}
        
        if language is None:
            language = "chinese" if self._is_chinese_text(text) else "english" if self._is_english_text(text) else "mixed"
            
        # 这里可以添加更复杂的语义分析逻辑，例如使用BERT模型
        # 简化版本，基于正则表达式模式匹配进行评估
        
        # 根据语言选择分析方法
        if language == "english":
            # 英文文本分析
            sensitive_patterns = [
                (r'\b(overthrow|topple|against)\s+.{0,20}\s+(government|communist|party)\b', 0.8, '反政府言论'),
                (r'\b(freedom|democracy|liberty)\s+.{0,20}\s+(suppressed|denied|restricted)\b', 0.7, '政治批评'),
                (r'\b(human rights|abuses|violations)\s+.{0,20}\s+(china|chinese)\b', 0.7, '人权批评'),
                (r'\b(censorship|surveillance|monitoring)\s+.{0,20}\s+(internet|media|press)\b', 0.6, '关于言论自由限制的言论'),
                (r'\b(taiwan|hong kong|tibet|xinjiang)\s+.{0,20}\s+(independence|free|freedom|autonomous)\b', 0.9, '分裂言论'),
                (r'\b(government|party|communist)\s+.{0,20}\s+(corrupt|corruption|authoritarian|dictatorship)\b', 0.7, '政府批评'),
                (r'\b(protest|uprising|revolution|demonstration)\s+.{0,20}\s+(support|should|must|need)\b', 0.8, '煽动抗议'),
                (r'\b(history|historical|massacre|tiananmen)\s+.{0,20}\s+(hidden|covered up|suppressed|truth)\b', 0.8, '历史敏感事件')
            ]
        else:
            # 中文文本分析
            sensitive_patterns = [
                (r'(推翻|反对|打倒).*?(政府|共产党|党)', 0.8, '反政府言论'),
                (r'(自由|民主|民权).*?(被限制|受限|压制)', 0.7, '政治批评'),
                (r'(人权|侵害|迫害).*?(中国|国内)', 0.7, '人权批评'),
                (r'(审查|监控|监听).*?(互联网|媒体|新闻)', 0.6, '关于言论自由限制的言论'),
                (r'(台湾|香港|西藏|新疆).*?(独立|自治|自由)', 0.9, '分裂言论')
            ]
        
        risk_factors = []
        score = 0.0
        
        for pattern, weight, description in sensitive_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                risk_factor = {
                    'type': 'semantic',
                    'description': description,
                    'score': weight,
                    'evidence': matches[0] if isinstance(matches[0], str) else ' '.join(matches[0])
                }
                risk_factors.append(risk_factor)
                score += weight
        
        # 归一化分数
        max_score = self.config.get('semantic_analysis', {}).get('max_score', 2.0)
        normalized_score = min(1.0, score / max_score)
        
        return {
            'score': normalized_score,
            'risk_factors': risk_factors,
            'embeddings': None  # 简化版中不返回嵌入
        }
    
    def _analyze_logic_consistency(self):
        """
        逻辑一致性分析
        
        Returns:
            dict: 逻辑一致性分析结果
        """
        if len(self.history) < 2 or not self.config.get('logic_analysis', {}).get('enabled', True):
            return {'score': 0.0, 'risk_factors': []}
            
        # 简单版本，检测最近陈述中的矛盾
        risk_factors = []
        score = 0.0
        
        # 这里简化为如果最近的两次陈述风险评分都很高，认为可能存在刻意宣传
        recent_texts = list(self.history)[-2:]
        if len(recent_texts) >= 2:
            kw_scores = [self._analyze_keywords(text)['score'] for text in recent_texts]
            if all(s > 0.5 for s in kw_scores):
                risk_factor = {
                    'type': 'logic',
                    'description': '检测到持续高风险言论，可能存在刻意宣传',
                    'score': 0.7,
                    'evidence': recent_texts[-1]
                }
                risk_factors.append(risk_factor)
                score = 0.7
        
        return {
            'score': score,
            'risk_factors': risk_factors
        }
