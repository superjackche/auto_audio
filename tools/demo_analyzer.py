#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
演示音频分析工具 - 专门用于参赛作品展示
"""

import os
import sys
import logging
import tempfile
import json
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.audio.speech_to_text import SpeechToText
from src.nlp.educational_analyzer import EducationalSemanticAnalyzer
from src.nlp.english_educational_analyzer import EnglishEducationalAnalyzer
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class DemoAudioAnalyzer:
    """演示音频分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.stt = SpeechToText()
        self.educational_analyzer = EducationalSemanticAnalyzer()
        self.english_educational_analyzer = EnglishEducationalAnalyzer()
        self.config_loader = ConfigLoader()
        
    def analyze_audio_file(self, audio_file_path: str, context: dict = None) -> dict:
        """
        分析音频文件并生成详细报告
        
        Args:
            audio_file_path: 音频文件路径
            context: 上下文信息（课程、教师等）
            
        Returns:
            分析结果字典
        """
        try:
            logger.info(f"开始分析音频文件: {audio_file_path}")
            
            # 检查文件是否存在
            if not os.path.exists(audio_file_path):
                return {
                    'error': f'音频文件不存在: {audio_file_path}',
                    'timestamp': datetime.now().isoformat()
                }
              # 语音转文本
            logger.info("进行语音识别...")
            transcript = self.stt.transcribe_file(audio_file_path)
            
            if not transcript:
                return {
                    'error': '语音识别失败，无法获取文本内容',
                    'timestamp': datetime.now().isoformat()
                }
            
            logger.info(f"语音识别成功，文本长度: {len(transcript)} 字符")
            
            # 确定使用哪种分析器
            language = context.get('language', 'chinese') if context else 'chinese'
            
            # 教育语义分析
            logger.info(f"进行{language}教育语义分析...")
            if language == 'english':
                # 使用英文分析器
                analysis_result = self.english_educational_analyzer.analyze_text(
                    text=transcript
                )
            else:
                # 使用中文分析器
                analysis_result = self.educational_analyzer.analyze_educational_content(
                    text=transcript,
                    context=context
                )
            
            # 生成完整的演示报告
            demo_report = self._generate_demo_report(
                audio_file=audio_file_path,
                transcript=transcript,
                analysis_result=analysis_result,
                context=context,
                language=language
            )
            
            logger.info(f"分析完成，风险等级: {demo_report['risk_level']}")
            return demo_report
            
        except Exception as e:
            logger.exception(f"音频分析失败: {str(e)}")
            return {
                'error': f'音频分析失败: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_demo_report(self, audio_file: str, transcript: str, 
                            analysis_result: dict, context: dict = None) -> dict:
        """生成演示报告"""
        
        # 基础信息
        file_info = {
            'filename': os.path.basename(audio_file),
            'file_size': os.path.getsize(audio_file),
            'duration': self._get_audio_duration(audio_file)
        }
        
        # 风险等级颜色映射
        risk_level_colors = {
            'low': '#28a745',      # 绿色
            'medium': '#ffc107',   # 黄色
            'high': '#fd7e14',     # 橙色
            'critical': '#dc3545'  # 红色
        }
        
        risk_level = analysis_result.get('risk_level', 'low')
        risk_score = analysis_result.get('risk_score', {}).get('total_score', 0)
        
        # 生成可视化数据
        visualization_data = self._generate_visualization_data(analysis_result)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'file_info': file_info,
            'context': context or {},
            'transcript': {
                'text': transcript,
                'length': len(transcript),
                'word_count': len(transcript.split())
            },
            'analysis_result': analysis_result,
            'risk_assessment': {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'risk_color': risk_level_colors.get(risk_level, '#6c757d'),
                'risk_percentage': min(100, risk_score),
                'risk_description': self._get_risk_description(risk_level, risk_score)
            },
            'visualization': visualization_data,
            'recommendations': analysis_result.get('recommendations', []),
            'demo_mode': True
        }
    
    def _get_audio_duration(self, audio_file: str) -> float:
        """获取音频时长"""
        try:
            import wave
            with wave.open(audio_file, 'r') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                return frames / float(sample_rate)
        except:
            return 0.0
    
    def _get_risk_description(self, risk_level: str, risk_score: float) -> str:
        """获取风险描述"""
        descriptions = {
            'low': f"风险评分 {risk_score:.1f}/100，内容表达规范，符合教育标准",
            'medium': f"风险评分 {risk_score:.1f}/100，存在需要关注的表达，建议进一步观察",
            'high': f"风险评分 {risk_score:.1f}/100，检测到较高风险内容，建议及时干预",
            'critical': f"风险评分 {risk_score:.1f}/100，检测到严重风险内容，需要立即处理"
        }
        return descriptions.get(risk_level, f"风险评分 {risk_score:.1f}/100")
    
    def _generate_visualization_data(self, analysis_result: dict) -> dict:
        """生成可视化数据"""
        
        # 风险分数分解
        risk_breakdown = analysis_result.get('risk_score', {}).get('score_breakdown', {})
        
        # 关键词分析
        keyword_analysis = analysis_result.get('keyword_analysis', {})
        risk_keywords = keyword_analysis.get('risk_keywords', [])
        positive_keywords = keyword_analysis.get('positive_keywords', [])
        
        # 模式分析
        pattern_analysis = analysis_result.get('pattern_analysis', {})
        found_patterns = pattern_analysis.get('found_patterns', [])
        
        return {
            'risk_breakdown': {
                'keyword_risk': risk_breakdown.get('keyword_risk', 0),
                'pattern_risk': risk_breakdown.get('pattern_risk', 0),
                'logic_consistency': risk_breakdown.get('logic_consistency', 0),
                'sentiment_bias': risk_breakdown.get('sentiment_bias', 0)
            },
            'keyword_stats': {
                'risk_count': len(risk_keywords),
                'positive_count': len(positive_keywords),
                'risk_keywords': [kw['keyword'] for kw in risk_keywords[:5]],  # 显示前5个
                'positive_keywords': [kw['keyword'] for kw in positive_keywords[:5]]
            },
            'pattern_stats': {
                'pattern_count': len(found_patterns),
                'pattern_types': list(set(p['type'] for p in found_patterns))
            }
        }

def main():
    """主函数 - 命令行演示"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI智能反馈系统 - 音频分析演示工具')
    parser.add_argument('audio_file', help='音频文件路径')
    parser.add_argument('--teacher', default='演示教师', help='教师姓名')
    parser.add_argument('--course', default='演示课程', help='课程名称')
    parser.add_argument('--output', help='输出结果到JSON文件')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建分析器
    analyzer = DemoAudioAnalyzer()
    
    # 设置上下文
    context = {
        'teacher': args.teacher,
        'course': args.course,
        'demo_mode': True
    }
    
    # 分析音频
    print(f"🎯 开始分析音频文件: {args.audio_file}")
    result = analyzer.analyze_audio_file(args.audio_file, context)
    
    if 'error' in result:
        print(f"❌ 分析失败: {result['error']}")
        return 1
    
    # 显示结果
    print("\n" + "="*60)
    print("🎓 AI智能反馈系统 - 教育语义监控分析报告")
    print("="*60)
    
    # 基础信息
    file_info = result['file_info']
    print(f"📁 文件名: {file_info['filename']}")
    print(f"⏱️  时长: {file_info['duration']:.1f} 秒")
    print(f"📊 文本长度: {result['transcript']['length']} 字符")
    
    # 风险评估
    risk_assessment = result['risk_assessment']
    risk_level = risk_assessment['risk_level']
    risk_score = risk_assessment['risk_score']
    
    risk_emojis = {
        'low': '✅',
        'medium': '⚠️',
        'high': '🚨',
        'critical': '🔴'
    }
    
    print(f"\n{risk_emojis.get(risk_level, '❓')} 风险等级: {risk_level.upper()}")
    print(f"📈 风险评分: {risk_score:.1f}/100")
    print(f"📝 评估说明: {risk_assessment['risk_description']}")
    
    # 关键词统计
    keyword_stats = result['visualization']['keyword_stats']
    print(f"\n🔍 关键词分析:")
    print(f"   风险关键词: {keyword_stats['risk_count']} 个")
    print(f"   正面关键词: {keyword_stats['positive_count']} 个")
    
    if keyword_stats['risk_keywords']:
        print(f"   检测到的风险词汇: {', '.join(keyword_stats['risk_keywords'])}")
    
    # 建议
    recommendations = result['recommendations']
    if recommendations:
        print(f"\n💡 分析建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    # 保存结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n💾 分析结果已保存到: {args.output}")
    
    print("\n" + "="*60)
    print("✨ 分析完成！适合用于参赛作品演示展示")
    print("="*60)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
