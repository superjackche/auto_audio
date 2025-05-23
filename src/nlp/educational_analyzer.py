"""
教育语义监控分析器 - 专门用于课堂教学内容分析
针对中外合作办学意识形态风险防控设计
"""

import re
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class EducationalSemanticAnalyzer:
    """教育语义监控分析器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化分析器"""
        self.config = config or {}
        self._load_educational_keywords()
        self._load_risk_patterns()
        self._initialize_scoring_system()
        
        # 历史分析记录（用于前后逻辑比较）
        self.analysis_history = []
        self.max_history = 50  # 保留最近50次分析
        
    def _load_educational_keywords(self):
        """加载教育相关关键词库"""
        # 正面教育关键词
        self.positive_educational_keywords = {
            # 爱国主义教育
            '爱国', '民族', '国家', '祖国', '中华民族', '民族团结', '国家统一',
            '社会主义核心价值观', '中国特色社会主义', '新时代',
            
            # 学术正面词汇
            '学术研究', '科学精神', '创新', '实践', '理论联系实际', '批判性思维',
            '学术诚信', '学术规范', '科研伦理', '学术交流',
            
            # 教育正面词汇
            '教育公平', '因材施教', '立德树人', '全面发展', '素质教育',
            '师德师风', '教书育人', '为人师表', '传道授业解惑',
            
            # 国际交流正面词汇
            '文化交流', '互学互鉴', '和而不同', '包容开放', '合作共赢',
            '人类命运共同体', '文明对话', '文化自信'
        }
        
        # 风险关键词
        self.risk_keywords = {
            # 意识形态风险词汇
            '意识形态': {'level': 'high', 'weight': 10},
            '政治制度': {'level': 'medium', 'weight': 7},
            '制度优越性': {'level': 'medium', 'weight': 6},
            '民主自由': {'level': 'medium', 'weight': 5},
            '人权': {'level': 'medium', 'weight': 5},
            
            # 历史虚无主义风险
            '历史虚无': {'level': 'high', 'weight': 9},
            '文革': {'level': 'high', 'weight': 8},
            '历史反思': {'level': 'medium', 'weight': 6},
            
            # 西方价值观风险
            '西方价值观': {'level': 'high', 'weight': 8},
            '普世价值': {'level': 'high', 'weight': 8},
            '个人主义': {'level': 'medium', 'weight': 5},
            '自由主义': {'level': 'medium', 'weight': 6},
            
            # 宗教渗透风险
            '宗教信仰': {'level': 'medium', 'weight': 6},
            '基督教': {'level': 'medium', 'weight': 5},
            '传教': {'level': 'high', 'weight': 8},
            
            # 分裂风险词汇
            '台独': {'level': 'critical', 'weight': 15},
            '港独': {'level': 'critical', 'weight': 15},
            '藏独': {'level': 'critical', 'weight': 15},
            '新疆问题': {'level': 'high', 'weight': 10},
            '一国两制': {'level': 'medium', 'weight': 6}
        }
        
        # 学术争议词汇（需要上下文分析）
        self.controversial_academic_terms = {
            '学术自由', '言论自由', '新闻自由', '批判精神', '独立思考',
            '质疑权威', '学术独立', '价值中立', '客观性', '多元化'
        }
        
    def _load_risk_patterns(self):
        """加载风险模式"""
        self.risk_patterns = {
            # 比较模式（暗示制度对比）
            'comparison_patterns': [
                r'(中国|国内).*?(不如|落后于|比不上).*(西方|欧美|发达国家)',
                r'(西方|欧美|发达国家).*?(先进|优越|领先).*(中国|国内)',
                r'为什么.*?(中国|我们).*?不能.*?(像.*?一样|学习.*?)',
            ],
            
            # 暗示模式
            'implication_patterns': [
                r'你们(应该|需要|必须).*?(思考|反思|质疑)',
                r'(真正的|真实的).*?(历史|事实|真相)',
                r'(官方|政府).*?(说法|版本|解释).*?(但是|然而|实际上)',
            ],
            
            # 价值观引导模式
            'value_guidance_patterns': [
                r'(自由|民主|人权).*?(重要|宝贵|珍贵)',
                r'每个人.*?(都有权|应该).*?(选择|决定|表达)',
                r'(多元化|包容性).*?(社会|文化|观念)',
            ],
            
            # 历史否定模式
            'history_denial_patterns': [
                r'(历史|过去).*?(错误|问题|反思)',
                r'(那个时代|当时).*?(不了解|不知道|被误导)',
                r'现在.*?(重新|客观|理性).*?(看待|评价|认识)',
            ]
        }
    
    def _initialize_scoring_system(self):
        """初始化评分系统"""
        self.scoring_weights = {
            'keyword_risk': 0.4,      # 关键词风险权重
            'pattern_risk': 0.3,      # 模式风险权重
            'logic_consistency': 0.2,  # 逻辑一致性权重
            'sentiment_bias': 0.1     # 情感倾向权重
        }
        
        # 风险等级阈值
        self.risk_thresholds = {
            'low': 30,
            'medium': 50,
            'high': 70,
            'critical': 85
        }
    
    def analyze_educational_content(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        分析教育内容
        
        Args:
            text: 要分析的文本（课堂讲话内容）
            context: 上下文信息（课程、教师、时间等）
            
        Returns:
            详细的分析结果
        """
        if not text or not text.strip():
            return self._empty_analysis_result()
        
        # 基础文本分析
        basic_analysis = self._basic_text_analysis(text)
        
        # 关键词风险分析
        keyword_analysis = self._analyze_keywords(text)
        
        # 模式风险分析
        pattern_analysis = self._analyze_patterns(text)
        
        # 逻辑一致性分析
        logic_analysis = self._analyze_logic_consistency(text, basic_analysis)
        
        # 情感倾向分析
        sentiment_analysis = self._analyze_educational_sentiment(text)
        
        # 综合风险评分
        risk_score = self._calculate_risk_score(
            keyword_analysis, pattern_analysis, logic_analysis, sentiment_analysis
        )
        
        # 生成分析结果
        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'text_length': len(text),
            'basic_analysis': basic_analysis,
            'keyword_analysis': keyword_analysis,
            'pattern_analysis': pattern_analysis,
            'logic_analysis': logic_analysis,
            'sentiment_analysis': sentiment_analysis,
            'risk_score': risk_score,
            'risk_level': self._determine_risk_level(risk_score['total_score']),
            'recommendations': self._generate_recommendations(risk_score, keyword_analysis, pattern_analysis),
            'context': context or {}
        }
        
        # 保存到历史记录
        self._save_to_history(analysis_result)
        
        return analysis_result
    
    def _basic_text_analysis(self, text: str) -> Dict[str, Any]:
        """基础文本分析"""
        # 简单分词
        words = self._simple_tokenize(text)
        
        # 语言检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = chinese_chars + english_chars
        
        language = 'unknown'
        if total_chars > 0:
            chinese_ratio = chinese_chars / total_chars
            if chinese_ratio > 0.7:
                language = 'chinese'
            elif chinese_ratio < 0.3:
                language = 'english'
            else:
                language = 'mixed'
        
        return {
            'word_count': len(words),
            'char_count': len(text),
            'language': language,
            'tokens': words[:20]  # 只保留前20个词汇
        }
    
    def _analyze_keywords(self, text: str) -> Dict[str, Any]:
        """关键词风险分析"""
        found_risk_keywords = []
        found_positive_keywords = []
        found_controversial_keywords = []
        
        text_lower = text.lower()
        
        # 检查风险关键词
        for keyword, info in self.risk_keywords.items():
            if keyword in text:
                found_risk_keywords.append({
                    'keyword': keyword,
                    'level': info['level'],
                    'weight': info['weight'],
                    'context': self._extract_context(text, keyword)
                })
        
        # 检查正面教育关键词
        for keyword in self.positive_educational_keywords:
            if keyword in text:
                found_positive_keywords.append({
                    'keyword': keyword,
                    'context': self._extract_context(text, keyword)
                })
        
        # 检查争议性学术词汇
        for keyword in self.controversial_academic_terms:
            if keyword in text:
                found_controversial_keywords.append({
                    'keyword': keyword,
                    'context': self._extract_context(text, keyword)
                })
        
        # 计算关键词风险分数
        risk_score = sum(kw['weight'] for kw in found_risk_keywords)
        positive_score = len(found_positive_keywords) * 2  # 正面词汇降低风险
        
        return {
            'risk_keywords': found_risk_keywords,
            'positive_keywords': found_positive_keywords,
            'controversial_keywords': found_controversial_keywords,
            'risk_score': max(0, risk_score - positive_score),
            'keyword_density': len(found_risk_keywords) / max(1, len(text.split()))
        }
    
    def _analyze_patterns(self, text: str) -> Dict[str, Any]:
        """模式风险分析"""
        found_patterns = []
        
        for pattern_type, patterns in self.risk_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    found_patterns.append({
                        'type': pattern_type,
                        'pattern': pattern,
                        'matched_text': match.group(),
                        'position': match.span(),
                        'context': self._extract_context(text, match.group())
                    })
        
        # 计算模式风险分数
        pattern_weights = {
            'comparison_patterns': 8,
            'implication_patterns': 6,
            'value_guidance_patterns': 7,
            'history_denial_patterns': 9
        }
        
        pattern_score = sum(pattern_weights.get(p['type'], 5) for p in found_patterns)
        
        return {
            'found_patterns': found_patterns,
            'pattern_count': len(found_patterns),
            'pattern_score': pattern_score
        }
    
    def _analyze_logic_consistency(self, text: str, basic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """逻辑一致性分析"""
        consistency_score = 100  # 基础分数
        inconsistencies = []
        
        # 检查与历史内容的一致性
        if len(self.analysis_history) > 0:
            recent_analyses = self.analysis_history[-5:]  # 最近5次分析
            
            # 检查主题一致性
            current_keywords = set()
            if basic_analysis.get('tokens'):
                current_keywords = set(basic_analysis['tokens'][:10])
            
            for historical in recent_analyses:
                if 'basic_analysis' in historical and 'tokens' in historical['basic_analysis']:
                    historical_keywords = set(historical['basic_analysis']['tokens'][:10])
                    similarity = len(current_keywords & historical_keywords) / max(1, len(current_keywords | historical_keywords))
                    
                    if similarity < 0.3:  # 主题变化较大
                        inconsistencies.append({
                            'type': 'topic_shift',
                            'description': '主题变化较大，可能存在话题跳跃',
                            'severity': 'medium'
                        })
                        consistency_score -= 10
        
        # 检查文本内部逻辑
        sentences = re.split(r'[。！？；]', text)
        if len(sentences) > 2:
            # 检查转折词使用
            transition_words = ['但是', '然而', '不过', '实际上', '事实上', '其实']
            transition_count = sum(1 for word in transition_words if word in text)
            
            if transition_count > len(sentences) * 0.3:  # 转折词过多
                inconsistencies.append({
                    'type': 'excessive_transitions',
                    'description': '转折词使用过多，可能存在观点摇摆',
                    'severity': 'low'
                })
                consistency_score -= 5
        
        return {
            'consistency_score': max(0, consistency_score),
            'inconsistencies': inconsistencies,
            'analysis_count': len(self.analysis_history)
        }
    
    def _analyze_educational_sentiment(self, text: str) -> Dict[str, Any]:
        """教育情感分析"""
        # 教育正面情感词汇
        positive_educational_words = {
            '优秀', '卓越', '进步', '发展', '成就', '荣誉', '自豪', '骄傲',
            '希望', '未来', '梦想', '理想', '奋斗', '努力', '坚持', '成功',
            '学习', '成长', '提高', '改善', '创新', '突破', '贡献', '奉献'
        }
        
        # 负面或争议情感词汇
        negative_educational_words = {
            '落后', '失败', '错误', '问题', '困难', '挑战', '批评', '质疑',
            '反对', '抗议', '不满', '愤怒', '失望', '担心', '忧虑', '恐惧',
            '混乱', '危机', '冲突', '对立', '分歧', '争议', '怀疑', '否定'
        }
        
        words = self._simple_tokenize(text)
        
        positive_count = sum(1 for word in words if word in positive_educational_words)
        negative_count = sum(1 for word in words if word in negative_educational_words)
        
        total_emotional_words = positive_count + negative_count
        
        if total_emotional_words == 0:
            sentiment = 'neutral'
            bias_score = 0
        else:
            positive_ratio = positive_count / total_emotional_words
            if positive_ratio > 0.6:
                sentiment = 'positive'
                bias_score = 0
            elif positive_ratio < 0.4:
                sentiment = 'negative'
                bias_score = (0.4 - positive_ratio) * 50  # 负面情感的风险分数
            else:
                sentiment = 'neutral'
                bias_score = 0
        
        return {
            'sentiment': sentiment,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'bias_score': bias_score,
            'emotional_intensity': total_emotional_words / max(1, len(words))
        }
    
    def _calculate_risk_score(self, keyword_analysis: Dict, pattern_analysis: Dict, 
                            logic_analysis: Dict, sentiment_analysis: Dict) -> Dict[str, Any]:
        """计算综合风险评分"""
        
        # 各项分数
        keyword_score = keyword_analysis.get('risk_score', 0)
        pattern_score = pattern_analysis.get('pattern_score', 0)
        logic_score = max(0, 100 - logic_analysis.get('consistency_score', 100))
        sentiment_score = sentiment_analysis.get('bias_score', 0)
        
        # 加权计算
        weights = self.scoring_weights
        total_score = (
            keyword_score * weights['keyword_risk'] +
            pattern_score * weights['pattern_risk'] +
            logic_score * weights['logic_consistency'] +
            sentiment_score * weights['sentiment_bias']
        )
        
        return {
            'keyword_score': keyword_score,
            'pattern_score': pattern_score,
            'logic_score': logic_score,
            'sentiment_score': sentiment_score,
            'total_score': min(100, total_score),  # 最高100分
            'score_breakdown': {
                'keyword_risk': keyword_score * weights['keyword_risk'],
                'pattern_risk': pattern_score * weights['pattern_risk'],
                'logic_consistency': logic_score * weights['logic_consistency'],
                'sentiment_bias': sentiment_score * weights['sentiment_bias']
            }
        }
    
    def _determine_risk_level(self, score: float) -> str:
        """确定风险等级"""
        if score >= self.risk_thresholds['critical']:
            return 'critical'
        elif score >= self.risk_thresholds['high']:
            return 'high'
        elif score >= self.risk_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, risk_score: Dict, keyword_analysis: Dict, 
                                pattern_analysis: Dict) -> List[str]:
        """生成建议"""
        recommendations = []
        
        total_score = risk_score['total_score']
        
        if total_score >= self.risk_thresholds['high']:
            recommendations.append("🚨 建议立即关注：检测到高风险内容，需要及时干预")
        elif total_score >= self.risk_thresholds['medium']:
            recommendations.append("⚠️ 建议关注：检测到中等风险内容，建议进一步观察")
        
        # 具体建议
        if keyword_analysis.get('risk_keywords'):
            recommendations.append("• 注意敏感关键词的使用，建议谨慎表达相关内容")
        
        if pattern_analysis.get('found_patterns'):
            recommendations.append("• 检测到潜在风险表达模式，建议调整表达方式")
        
        if len(keyword_analysis.get('positive_keywords', [])) < 2:
            recommendations.append("• 建议增加正面教育引导内容")
        
        if not recommendations:
            recommendations.append("✅ 内容表达规范，继续保持")
        
        return recommendations
    
    def _extract_context(self, text: str, keyword: str, context_length: int = 30) -> str:
        """提取关键词上下文"""
        index = text.find(keyword)
        if index == -1:
            return ""
        
        start = max(0, index - context_length)
        end = min(len(text), index + len(keyword) + context_length)
        
        context = text[start:end]
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."
            
        return context
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """简单分词"""
        # 中英文混合分词
        pattern = r'[\u4e00-\u9fff]+|[a-zA-Z]+|[0-9]+'
        tokens = re.findall(pattern, text)
        return [token.lower() for token in tokens if len(token) > 1]
    
    def _save_to_history(self, analysis_result: Dict[str, Any]):
        """保存分析结果到历史记录"""
        self.analysis_history.append(analysis_result)
        
        # 保持历史记录在限制范围内
        if len(self.analysis_history) > self.max_history:
            self.analysis_history = self.analysis_history[-self.max_history:]
    
    def _empty_analysis_result(self) -> Dict[str, Any]:
        """空分析结果"""
        return {
            'timestamp': datetime.now().isoformat(),
            'text_length': 0,
            'basic_analysis': {'word_count': 0, 'char_count': 0, 'language': 'unknown', 'tokens': []},
            'keyword_analysis': {'risk_keywords': [], 'positive_keywords': [], 'risk_score': 0},
            'pattern_analysis': {'found_patterns': [], 'pattern_score': 0},
            'logic_analysis': {'consistency_score': 100, 'inconsistencies': []},
            'sentiment_analysis': {'sentiment': 'neutral', 'bias_score': 0},
            'risk_score': {'total_score': 0},
            'risk_level': 'low',
            'recommendations': [],
            'context': {}
        }
    
    def get_analysis_summary(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """获取指定时间范围内的分析摘要"""
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        
        recent_analyses = [
            analysis for analysis in self.analysis_history
            if datetime.fromisoformat(analysis['timestamp']) > cutoff_time
        ]
        
        if not recent_analyses:
            return {'message': '指定时间范围内无分析记录'}
        
        # 统计分析
        risk_levels = [analysis['risk_level'] for analysis in recent_analyses]
        risk_level_counts = Counter(risk_levels)
        
        avg_risk_score = sum(analysis['risk_score']['total_score'] for analysis in recent_analyses) / len(recent_analyses)
        
        # 频繁出现的风险关键词
        all_risk_keywords = []
        for analysis in recent_analyses:
            all_risk_keywords.extend([kw['keyword'] for kw in analysis['keyword_analysis']['risk_keywords']])
        
        frequent_risk_keywords = Counter(all_risk_keywords).most_common(5)
        
        return {
            'time_range_hours': time_range_hours,
            'total_analyses': len(recent_analyses),
            'risk_level_distribution': dict(risk_level_counts),
            'average_risk_score': round(avg_risk_score, 2),
            'frequent_risk_keywords': frequent_risk_keywords,
            'high_risk_count': risk_level_counts.get('high', 0) + risk_level_counts.get('critical', 0)
        }
