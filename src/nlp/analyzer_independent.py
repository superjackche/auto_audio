"""
完全独立的文本分析器 - 不依赖外部NLP库
适用于Python 3.13.2
"""

import re
import os
import logging
from typing import List, Dict, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)

class IndependentTextAnalyzer:
    """完全独立的文本分析器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化分析器"""
        self.config = config or {}
        self.stopwords = self._load_stopwords()
        self.common_words = self._load_common_words()
    
    def _load_stopwords(self) -> set:
        """加载停用词"""
        return {
            # 中文停用词
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '这', '那', '什么', '怎么', '为什么', '但是', '可以', '应该', '能够', '已经',
            '还是', '或者', '因为', '所以', '如果', '虽然', '然后', '现在', '时候', '地方',
            
            # 英文停用词
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its',
            'our', 'their', 'what', 'who', 'how', 'when', 'where', 'why', 'which'
        }
    
    def _load_common_words(self) -> set:
        """加载常见词汇（用于改善分词）"""
        return {
            # 技术词汇
            '人工智能', '机器学习', '深度学习', '神经网络', '自然语言', '计算机科学',
            '数据分析', '大数据', '云计算', '物联网', '区块链', '虚拟现实',
            '增强现实', '算法', '编程', '软件工程', '系统架构', '数据库',
            '网络安全', '信息安全', '移动应用', '用户体验', '产品设计',
            
            # 英文技术词汇
            'artificial intelligence', 'machine learning', 'deep learning',
            'neural network', 'natural language', 'computer science',
            'data analysis', 'big data', 'cloud computing', 'internet of things',
            'blockchain', 'virtual reality', 'augmented reality', 'algorithm',
            'programming', 'software engineering', 'system architecture',
            'database', 'cybersecurity', 'information security', 'mobile app',
            'user experience', 'product design'
        }
    
    def tokenize(self, text: str) -> List[str]:
        """分词"""
        if not text:
            return []
          # 预处理
        text = re.sub(r'\s+', ' ', text.strip())
        words = []
          # 使用正则表达式进行基础分词
        # 匹配中文词汇、英文单词、数字
        pattern = r'[\u4e00-\u9fff]+|[a-zA-Z]+|[0-9]+|[^\u4e00-\u9fffa-zA-Z0-9\s]+'
        tokens = re.findall(pattern, text)
        
        for token in tokens:
            if re.match(r'[\u4e00-\u9fff]+', token):
                # 中文文本：基于词典的简单分词
                words.extend(self._segment_chinese(token))
            elif re.match(r'[a-zA-Z]+', token):
                # 英文单词
                if len(token) > 1:  # 过滤单字母
                    words.append(token.lower())
            elif re.match(r'[0-9]+', token):
                # 数字
                words.append(token)
        
        return words
    
    def _segment_chinese(self, text: str) -> List[str]:
        """简单的中文分词"""
        words = []
        i = 0
        
        while i < len(text):
            # 尝试匹配常见词汇（最长匹配）
            matched = False
            for length in range(min(8, len(text) - i), 0, -1):
                candidate = text[i:i+length]
                if candidate in self.common_words:
                    words.append(candidate)
                    i += length
                    matched = True
                    break
            
            if not matched:
                # 双字符词汇优先
                if i + 1 < len(text):
                    words.append(text[i:i+2])
                    i += 2
                else:
                    words.append(text[i])
                    i += 1
        
        return words
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """分析文本"""
        if not text or not text.strip():
            return {
                'word_count': 0,
                'char_count': 0,
                'keywords': [],
                'sentiment': 'neutral',
                'language': 'unknown',
                'tokens': [],
                'word_frequency': {}
            }
        
        # 基本统计
        char_count = len(text)
        
        # 分词
        tokens = self.tokenize(text)
        word_count = len(tokens)
        
        # 过滤停用词
        filtered_tokens = [token for token in tokens if token not in self.stopwords and len(token) > 1]
        
        # 词频统计
        word_freq = Counter(filtered_tokens)
        
        # 提取关键词
        keywords = [word for word, freq in word_freq.most_common(10)]
        
        # 语言检测
        language = self._detect_language(text)
        
        # 情感分析
        sentiment = self._analyze_sentiment(text, tokens)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'keywords': keywords,
            'sentiment': sentiment,
            'language': language,
            'tokens': tokens[:20],  # 只返回前20个token
            'word_frequency': dict(word_freq.most_common(15))
        }
    
    def _detect_language(self, text: str) -> str:
        """语言检测"""        # 统计不同类型字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = chinese_chars + english_chars
        if total_chars == 0:
            return 'unknown'
        
        chinese_ratio = chinese_chars / total_chars
        
        if chinese_ratio > 0.6:
            return 'chinese'
        elif chinese_ratio > 0.2:
            return 'mixed'
        else:
            return 'english'
    
    def _analyze_sentiment(self, text: str, tokens: List[str]) -> str:
        """情感分析"""
        # 情感词典
        positive_words = {
            # 中文正面词汇
            '好', '棒', '优秀', '喜欢', '开心', '满意', '赞', '不错', '很好', '完美',
            '成功', '高兴', '快乐', '兴奋', '惊喜', '感谢', '赞美', '欣赏', '支持',
            '同意', '认可', '肯定', '积极', '正面', '有用', '有效', '值得', '推荐',
            
            # 英文正面词汇
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'perfect', 'love',
            'like', 'happy', 'satisfied', 'awesome', 'fantastic', 'brilliant',
            'outstanding', 'impressive', 'remarkable', 'superb', 'magnificent',
            'beautiful', 'success', 'successful', 'positive', 'effective'
        }
        
        negative_words = {
            # 中文负面词汇
            '坏', '差', '糟糕', '讨厌', '不好', '失望', '愤怒', '难过', '痛苦', '烦人',
            '错误', '失败', '问题', '困难', '麻烦', '担心', '害怕', '紧张', '压力',
            '不满', '抱怨', '批评', '反对', '拒绝', '否定', '消极', '负面', '无用',
            
            # 英文负面词汇
            'bad', 'terrible', 'awful', 'hate', 'angry', 'sad', 'disappointed',
            'frustrated', 'annoying', 'horrible', 'worst', 'failed', 'failure',
            'problem', 'difficult', 'trouble', 'worried', 'afraid', 'negative',
            'useless', 'ineffective', 'wrong', 'error', 'mistake'
        }
        
        # 计算情感分数
        positive_score = sum(1 for token in tokens if token.lower() in positive_words)
        negative_score = sum(1 for token in tokens if token.lower() in negative_words)
        
        # 考虑否定词
        negation_words = {'不', '没', '无', '非', 'not', 'no', 'never', 'none'}
        has_negation = any(token.lower() in negation_words for token in tokens)
        
        if has_negation:
            # 如果有否定词，调整情感分数
            positive_score, negative_score = negative_score, positive_score
        
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
    
    def get_word_frequency(self, text: str, top_k: int = 15) -> Dict[str, int]:
        """获取词频统计"""
        result = self.analyze(text)
        return dict(list(result['word_frequency'].items())[:top_k])
    
    def is_available(self) -> bool:
        """检查分析器是否可用"""
        return True  # 独立分析器总是可用
    
    def get_analyzer_info(self) -> Dict[str, str]:
        """获取分析器信息"""
        return {
            'tokenizer_type': 'independent',
            'available': True,
            'description': '独立文本分析器，不依赖外部NLP库'
        }
