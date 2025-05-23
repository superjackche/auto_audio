"""
简化的文本分析模块 - 为Python 3.13.2优化
提供完整的中英文文本分析功能
"""

import re
import os
import logging
from typing import List, Dict, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)

class TextAnalyzer:
    """简化的文本分析器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化文本分析器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.tokenizer = None
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        """初始化分词器"""
        
        # 尝试使用jieba分词
        try:
            import jieba
            import jieba.posseg as pseg
            
            # 加载用户词典（如果有）
            user_dict = self.config.get('user_dict')
            if user_dict and os.path.exists(user_dict):
                jieba.load_userdict(user_dict)
            
            self.tokenizer = jieba
            self.pseg = pseg
            self.tokenizer_type = 'jieba'
            logger.info("使用jieba分词器")
            return
            
        except ImportError:
            logger.warning("jieba不可用，使用简单分词器")
        
        # 使用简单分词器
        self.tokenizer = self._create_simple_tokenizer()
        self.tokenizer_type = 'simple'
        logger.info("使用简单分词器")
    
    def _create_simple_tokenizer(self):
        """创建简单分词器"""
        
        class SimpleTokenizer:
            def __init__(self):
                # 常见停用词
                self.stopwords = set([
                    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
                    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
                    'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
                    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
                ])
                
                # 常见词汇词典（用于改善分词效果）
                self.common_words = {
                    '机器学习', '人工智能', '深度学习', '神经网络', '自然语言', '计算机', '互联网',
                    '数据分析', '算法', '编程', '软件', '硬件', '系统', '网络', '安全', '数据库',
                    'machine learning', 'artificial intelligence', 'deep learning', 'neural network',
                    'natural language', 'computer science', 'data analysis', 'algorithm', 'programming'
                }
            
            def lcut(self, text):
                """分词方法"""
                if not text:
                    return []
                
                # 预处理
                text = re.sub(r'\s+', ' ', text.strip())
                
                # 中英文混合分词
                words = []
                
                # 使用正则表达式分离中文、英文、数字
                pattern = r'[\u4e00-\u9fff]+|[a-zA-Z]+|[0-9]+|[^\u4e00-\u9fffa-zA-Z0-9\s]+'
                tokens = re.findall(pattern, text)
                
                for token in tokens:
                    if re.match(r'[\u4e00-\u9fff]+', token):
                        # 中文文本：进行简单的基于词典的分词
                        words.extend(self._segment_chinese(token))
                    elif re.match(r'[a-zA-Z]+', token):
                        # 英文单词
                        if len(token) > 1:  # 过滤单字母
                            words.append(token.lower())
                    elif re.match(r'[0-9]+', token):
                        # 数字
                        words.append(token)
                
                return words
            
            def _segment_chinese(self, text):
                """简单的中文分词"""
                words = []
                i = 0
                
                while i < len(text):
                    # 尝试匹配常见词汇
                    matched = False
                    for length in range(min(8, len(text) - i), 0, -1):
                        candidate = text[i:i+length]
                        if candidate in self.common_words:
                            words.append(candidate)
                            i += length
                            matched = True
                            break
                    
                    if not matched:
                        # 单字符
                        words.append(text[i])
                        i += 1
                
                return words
        
        return SimpleTokenizer()
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        分析文本
        
        Args:
            text: 要分析的文本
            
        Returns:
            分析结果字典
        """
        if not text or not text.strip():
            return {
                'word_count': 0,
                'char_count': 0,
                'keywords': [],
                'sentiment': 'neutral',
                'language': 'unknown',
                'tokens': []
            }
        
        # 基本统计
        char_count = len(text)
        
        # 分词
        tokens = self.tokenizer.lcut(text)
        word_count = len(tokens)
        
        # 过滤停用词（如果使用简单分词器）
        if hasattr(self.tokenizer, 'stopwords'):
            filtered_tokens = [token for token in tokens if token not in self.tokenizer.stopwords]
        else:
            # jieba分词器，需要自己定义停用词
            stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一'}
            filtered_tokens = [token for token in tokens if token not in stopwords]
        
        # 提取关键词（词频统计）
        word_freq = Counter(filtered_tokens)
        keywords = [word for word, freq in word_freq.most_common(10)]
        
        # 语言检测
        language = self._detect_language(text)
        
        # 简单情感分析
        sentiment = self._analyze_sentiment(text)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'keywords': keywords,
            'sentiment': sentiment,
            'language': language,
            'tokens': tokens[:50],  # 只返回前50个token
            'word_frequency': dict(word_freq.most_common(20))
        }
    
    def _detect_language(self, text: str) -> str:
        """简单的语言检测"""
        # 统计中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # 统计英文字符
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = chinese_chars + english_chars
        if total_chars == 0:
            return 'unknown'
        
        chinese_ratio = chinese_chars / total_chars
        
        if chinese_ratio > 0.5:
            return 'chinese'
        elif chinese_ratio > 0.1:
            return 'mixed'
        else:
            return 'english'
    
    def _analyze_sentiment(self, text: str) -> str:
        """简单的情感分析"""
        # 定义情感词典
        positive_words = {
            '好', '棒', '优秀', '喜欢', '开心', '满意', '赞', '不错', '很好', '完美',
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'perfect', 'love',
            'like', 'happy', 'satisfied', 'awesome', 'fantastic'
        }
        
        negative_words = {
            '坏', '差', '糟糕', '讨厌', '不好', '失望', '愤怒', '难过', '痛苦', '烦人',
            'bad', 'terrible', 'awful', 'hate', 'angry', 'sad', 'disappointed',
            'frustrated', 'annoying', 'horrible', 'worst'
        }
        
        # 分词
        tokens = self.tokenizer.lcut(text.lower())
        
        # 计算情感分数
        positive_score = sum(1 for token in tokens if token in positive_words)
        negative_score = sum(1 for token in tokens if token in negative_words)
        
        if positive_score > negative_score:
            return 'positive'
        elif negative_score > positive_score:
            return 'negative'
        else:
            return 'neutral'
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """提取关键词"""
        result = self.analyze(text)
        return result['keywords'][:top_k]
    
    def get_word_frequency(self, text: str, top_k: int = 20) -> Dict[str, int]:
        """获取词频统计"""
        result = self.analyze(text)
        return dict(list(result['word_frequency'].items())[:top_k])
    
    def tokenize(self, text: str) -> List[str]:
        """分词"""
        return self.tokenizer.lcut(text)
    
    def is_available(self) -> bool:
        """检查分析器是否可用"""
        return self.tokenizer is not None
    
    def get_analyzer_info(self) -> Dict[str, str]:
        """获取分析器信息"""
        return {
            'tokenizer_type': self.tokenizer_type,
            'available': self.is_available()
        }
