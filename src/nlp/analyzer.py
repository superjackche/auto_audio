#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语义分析模块 - 优化后的版本，支持中英文双语分析
"""

import os
import re
import json
import logging
import numpy as np
from datetime import datetime
from collections import deque

# 适配可能使用的NLP库
import sys
JIEBA_AVAILABLE = False
BERT_AVAILABLE = False

# jieba中文分词库
try:
    import jieba
    import jieba.posseg as pseg
    JIEBA_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✓ jieba 中文分词库已加载")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("⚠ jieba 库不可用，将使用简化分词方案")

# BERT模型相关
try:
    import torch
    from transformers import BertTokenizer, BertModel
    BERT_AVAILABLE = True
    logger.info("✓ BERT 模型库已加载")
except ImportError:
    logger.warning("⚠ transformers 库不可用，将使用简化语义分析")

# 其他NLP库
try:
    import spacy
    SPACY_AVAILABLE = True
    logger.info("✓ spaCy 库已加载")
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("⚠ spaCy 库不可用")

logger = logging.getLogger(__name__)

class TextAnalyzer:
    """文本分析器，支持中英文关键词检测"""
    
    def __init__(self, config):
        """
        初始化语义分析器
        
        Args:
            config: 自然语言处理配置字典
        """
        self.config = config
        self.embedding_model_name = config.get('embedding_model', 'bert-base-chinese')
        self.tokenizer_name = config.get('tokenizer', 'jieba')
        
        # 存储历史文本用于上下文分析
        self.history = deque(maxlen=config.get('logic_analysis', {}).get('history_window', 10))
        
        # 加载关键词列表
        self.keywords = self._load_keywords()
        
        # 初始化分词器
        self.tokenizer = self._initialize_tokenizer()
        
        # 初始化嵌入模型
        self.embedding_model = self._initialize_embedding_model()
        
        logger.info("语义分析器初始化完成")
        
    def _load_keywords(self):
        """
        加载中英文关键词列表
        
        Returns:
            dict: 关键词及其权重字典
        """
        keywords = {}
        
        # 中文关键词
        chinese_keyword_file = self.config.get('keyword_detection', {}).get('keyword_file')
        if chinese_keyword_file and os.path.exists(chinese_keyword_file):
            keywords.update(self._load_keyword_file(chinese_keyword_file))
            
        # 英文关键词
        english_keyword_file = self.config.get('keyword_detection', {}).get('english_keyword_file',
                                              os.path.join(os.path.dirname(chinese_keyword_file or ''), 'english_keywords.txt'))
        if english_keyword_file and os.path.exists(english_keyword_file):
            keywords.update(self._load_keyword_file(english_keyword_file))
            
        logger.info(f"已加载 {len(keywords)} 个敏感关键词")
        return keywords

    def _load_keyword_file(self, file_path):
        """
        从文件加载关键词
        Returns:
            dict: 关键词及其权重字典
        """
        keywords = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # 跳过注释和空行
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(',')
                    if len(parts) >= 2:
                        word = parts[0].strip()
                        try:
                            weight = float(parts[1].strip())
                        except Exception:                        weight = 0.5
                        keywords[word] = weight
            logger.info(f"成功加载关键词文件: {file_path}")
        except Exception as e:
            logger.error(f"加载关键词文件失败: {file_path} - {str(e)}")
        return keywords
    
    def _initialize_tokenizer(self):
        """
        初始化分词器
        
        Returns:
            object: 分词器对象
        """
        if self.tokenizer_name == 'jieba':
            if not JIEBA_AVAILABLE:
                logger.warning("jieba不可用，将使用简单分词替代方案")
                return self._create_simple_tokenizer()
                
            # 加载用户词典（如果有）
            user_dict = self.config.get('user_dict')
            if user_dict and os.path.exists(user_dict):
                jieba.load_userdict(user_dict)
                
            logger.info("jieba分词器初始化完成")
            return jieba
        elif self.tokenizer_name == 'bert':
            if not BERT_AVAILABLE:
                logger.error("未安装transformers模块，请使用 'pip install transformers'")
                raise ImportError("未安装transformers模块")
            tokenizer = BertTokenizer.from_pretrained(self.embedding_model_name)
            logger.info(f"BERT分词器初始化完成: {self.embedding_model_name}")
            return tokenizer
        else:
            logger.warning(f"不支持的分词器: {self.tokenizer_name}，将使用简单分词替代方案")
            return self._create_simple_tokenizer()
    
    def _create_simple_tokenizer(self):
        """
        创建增强的分词器替代方案
        """
        class EnhancedTokenizer:
            def __init__(self):
                # 常见停用词
                self.stopwords = set([
                    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
                    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
                    'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
                    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
                ])
            
            def lcut(self, text):
                """增强的中英文分词"""
                import re
                if not text:
                    return []
                
                # 预处理：清理文本
                text = re.sub(r'\s+', ' ', text.strip())
                
                # 分离中文字符、英文单词、数字等
                # 改进的正则表达式，更好地处理混合文本
                patterns = [
                    r'[\u4e00-\u9fff]+',  # 中文字符
                    r'[a-zA-Z]+(?:\'[a-zA-Z]+)*',  # 英文单词（包括缩写）
                    r'\d+(?:\.\d+)?',  # 数字（包括小数）
                    r'[!@#$%^&*(),.?":{}|<>]'  # 特殊符号
                ]
                
                tokens = []
                for pattern in patterns:
                    matches = re.findall(pattern, text)
                    tokens.extend(matches)
                    # 从原文本中移除已匹配的部分
                    text = re.sub(pattern, ' ', text)
                
                # 进一步处理中文文本：简单的字符级分词
                chinese_tokens = []
                for token in tokens:
                    if re.match(r'[\u4e00-\u9fff]+', token):
                        # 对于中文，尝试按语义单元分割
                        chinese_tokens.extend(self._segment_chinese(token))
                    else:
                        chinese_tokens.append(token)
                
                # 过滤停用词和短词
                filtered_tokens = []
                for token in chinese_tokens:
                    token = token.strip().lower()
                    if len(token) >= 1 and token not in self.stopwords:
                        filtered_tokens.append(token)
                
                return filtered_tokens
            
            def _segment_chinese(self, text):
                """简单的中文分词"""
                # 基于词典的简单中文分词
                common_words = {
                    '语音': 2, '识别': 2, '文本': 2, '分析': 2, '系统': 2, '课堂': 2,
                    '学生': 2, '老师': 2, '教学': 2, '学习': 2, '问题': 2, '内容': 2,
                    '时间': 2, '方法': 2, '知识': 2, '技术': 2, '数据': 2, '信息': 2,
                    '网络': 2, '计算机': 3, '人工智能': 4, '机器学习': 4, '深度学习': 4
                }
                
                result = []
                i = 0
                while i < len(text):
                    matched = False
                    # 尝试最长匹配
                    for length in range(min(6, len(text) - i), 0, -1):
                        word = text[i:i+length]
                        if word in common_words:
                            result.append(word)
                            i += length
                            matched = True
                            break
                    
                    if not matched:
                        # 单字符处理
                        if len(text[i]) > 0:
                            result.append(text[i])
                        i += 1
                
                return result
            
            def cut(self, text):
                """兼容jieba的cut方法"""
                return self.lcut(text)
            
            def posseg(self, text):
                """简单的词性标注"""
                tokens = self.lcut(text)
                # 简单的词性推断
                tagged = []
                for token in tokens:
                    if re.match(r'\d+', token):
                        pos = 'm'  # 数词
                    elif re.match(r'[a-zA-Z]+', token):
                        pos = 'eng'  # 英文
                    elif len(token) == 1 and re.match(r'[\u4e00-\u9fff]', token):
                        pos = 'n'  # 默认名词
                    else:
                        pos = 'n'  # 默认名词
                    
                    # 创建一个简单的对象来模拟jieba.posseg的返回值
                    class Word:
                        def __init__(self, word, flag):
                            self.word = word
                            self.flag = flag
                    
                    tagged.append(Word(token, pos))
                
                return tagged
                
        logger.info("增强分词器初始化完成")
        return EnhancedTokenizer()
    
    def _initialize_embedding_model(self):
        """
        初始化嵌入模型
        
        Returns:
            object: 嵌入模型对象
        """
        try:
            if not BERT_AVAILABLE:
                logger.warning("未安装transformers或torch模块，文本向量化功能将受限")
                return None
                
            model = BertModel.from_pretrained(self.embedding_model_name)
            
            # 移至GPU（如果可用）
            if torch.cuda.is_available():
                model = model.cuda()
                logger.info("模型已移至GPU")
                
            model.eval()  # 设置为评估模式
            logger.info(f"嵌入模型初始化完成: {self.embedding_model_name}")
            return model
            
        except Exception as e:
            logger.exception(f"初始化嵌入模型失败: {str(e)}")
            return None
    
    def analyze(self, text):
        """
        分析文本内容
        
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
        
        # 执行各种分析
        keyword_results = self._analyze_keywords(text)
        semantic_results = self._analyze_semantics(text)
        logic_results = self._analyze_logic_consistency()
        
        # 合并结果
        results = {
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'segments': self._segment_text(text),
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
        
        results['risk_score'] = min(1.0, max(0.0, risk_score))
        
        return results
    
    def _segment_text(self, text):
        """
        对文本进行分词
        
        Args:
            text: 待分词的文本
            
        Returns:
            list: 分词结果
        """
        if self.tokenizer_name == 'jieba':
            return list(self.tokenizer.cut(text))
        elif self.tokenizer_name == 'bert':
            return self.tokenizer.tokenize(text)
        else:
            # 简单的按空格分词
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
                if weight > 0.6:  # 只记录高风险关键词
                    risk_factors.append(f"敏感关键词: {keyword} (风险值: {weight:.2f})")
                
                # 累积分数，多次出现同一关键词风险增加但不是线性增加
                score += weight * (1 - 0.5 ** matches_count)
        
        # 归一化分数，确保不超过1.0
        score = min(1.0, score)
        
        return {
            'score': score,
            'matches': matches,
            'risk_factors': risk_factors
        }
    
    def _analyze_semantics(self, text):
        """
        语义分析
        
        Args:
            text: 待分析的文本
            
        Returns:
            dict: 语义分析结果
        """
        if not self.config.get('semantic_analysis', {}).get('enabled', True):
            return {'score': 0.0, 'risk_factors': [], 'embeddings': None}
        
        model_type = self.config.get('semantic_analysis', {}).get('model', 'local')
        
        if model_type == 'local' and self.embedding_model is not None:
            return self._analyze_semantics_local(text)
        elif model_type == 'api':
            return self._analyze_semantics_api(text)
        else:
            logger.warning(f"无法执行语义分析: 模型类型 '{model_type}' 不可用或未正确配置")
            return {'score': 0.0, 'risk_factors': [], 'embeddings': None}
    
    def _analyze_semantics_local(self, text):
        """使用本地模型进行语义分析"""
        try:
            # 简单规则检测
            risk_factors = []
            score = 0.0
            
            # 检测否定或批判性表述
            negative_patterns = [
                r'\b(批判|否定|反对|摒弃|拒绝|错误)\b.{0,20}\b(马克思|社会主义|共产主义|党的领导)\b',
                r'\b(马克思|社会主义|共产主义|党的领导).{0,20}\b(批判|否定|反对|摒弃|拒绝|错误)\b',
                r'\b(宣传|推广|提倡|鼓吹)\b.{0,20}\b(西方|外国|境外|资本主义).{0,20}(价值观|制度|理念)\b',
                r'\b(专制|独裁|集权|不民主)\b.{0,20}(国家|政府|制度|领导)',
                r'\b(政治)\b.{0,20}(转型|改革|变革|蜕变|转向)\b'
            ]
            
            for pattern in negative_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    risk_factors.append(f"可能的意识形态风险表述: {matches[0]}")
                    score += 0.3
            
            # 对比词分析
            positive_terms = ['社会主义', '马克思主义', '中国特色', '党的领导', '集体主义']
            negative_terms = ['资本主义', '西方民主', '个人主义', '自由主义', '新自由主义']
            
            # 检查是否有对比并偏向负面描述
            for pos_term in positive_terms:
                for neg_term in negative_terms:
                    compare_pattern = fr'{pos_term}.{{0,30}}不如.{{0,30}}{neg_term}'
                    if re.search(compare_pattern, text):
                        risk_factors.append(f"不当对比: 将{pos_term}与{neg_term}进行负面对比")
                        score += 0.4
            
            # 计算文本嵌入（如果可用）
            embeddings = None
            if self.embedding_model is not None:
                embeddings = self._get_text_embedding(text)
            
            # 限制分数在[0,1]范围
            score = min(1.0, score)
            
            return {
                'score': score,
                'risk_factors': risk_factors,
                'embeddings': embeddings
            }
            
        except Exception as e:
            logger.exception(f"本地语义分析失败: {str(e)}")
            return {'score': 0.0, 'risk_factors': [], 'embeddings': None}
    
    def _analyze_semantics_api(self, text):
        """使用API进行语义分析"""
        try:
            api_url = self.config.get('semantic_analysis', {}).get('api_url')
            api_key = self.config.get('semantic_analysis', {}).get('api_key')
            
            if not api_url:
                logger.error("未配置语义分析API URL")
                return {'score': 0.0, 'risk_factors': [], 'embeddings': None}
            
            # 这里应该实现API调用代码
            # 例如使用requests库发送请求
            
            # 为了示例，这里返回一个假数据
            return {
                'score': 0.0,
                'risk_factors': [],
                'embeddings': None
            }
            
        except Exception as e:
            logger.exception(f"API语义分析失败: {str(e)}")
            return {'score': 0.0, 'risk_factors': [], 'embeddings': None}
    
    def _get_text_embedding(self, text):
        """获取文本的嵌入向量"""
        try:
            # 对文本进行编码
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            
            # 移至GPU（如果可用）
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 获取模型输出
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
            
            # 使用最后一层的[CLS]向量作为文本表示
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings.tolist()[0]
            
        except Exception as e:
            logger.exception(f"获取文本嵌入失败: {str(e)}")
            return None
    
    def _analyze_logic_consistency(self):
        """
        分析逻辑一致性
        
        Returns:
            dict: 逻辑一致性分析结果
        """
        if not self.config.get('logic_analysis', {}).get('enabled', True) or len(self.history) < 2:
            return {'score': 0.0, 'risk_factors': []}
        
        risk_factors = []
        score = 0.0
        
        # 获取最新的记录
        current_text = self.history[-1]
        
        # 获取之前的记录并连接成一个大文本
        previous_text = ' '.join(list(self.history)[:-1])
        
        # 检测自相矛盾的表述
        pairs = [
            ('肯定', '否定'),
            ('好', '坏'),
            ('优点', '缺点'),
            ('支持', '反对'),
            ('先进', '落后'),
            ('民主', '专制'),
            ('集中', '分散'),
            ('中国特色', '西方模式')
        ]
        
        for pos, neg in pairs:
            # 检查前后表述不一致
            if pos in previous_text and neg in current_text and pos in current_text:
                context = re.findall(r'.{0,15}' + re.escape(pos) + r'.{0,15}', current_text)
                if context:
                    risk_factors.append(f"前后表述可能不一致: 之前提到'{pos}'，现在提到'{neg}'，上下文: {context[0]}")
                    score += 0.25
            
            if neg in previous_text and pos in current_text and neg in current_text:
                context = re.findall(r'.{0,15}' + re.escape(neg) + r'.{0,15}', current_text)
                if context:
                    risk_factors.append(f"前后表述可能不一致: 之前提到'{neg}'，现在提到'{pos}'，上下文: {context[0]}")
                    score += 0.25
        
        # 限制分数在[0,1]范围
        score = min(1.0, score)
        
        return {
            'score': score,
            'risk_factors': risk_factors
        }
    
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

# 为了向后兼容，创建SemanticAnalyzer别名
SemanticAnalyzer = TextAnalyzer
