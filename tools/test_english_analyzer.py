#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英文教育内容分析器测试工具
用于测试英文教育内容分析功能
"""

import os
import sys
import json
import logging
from pathlib import Path
import argparse
import time

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def analyze_english_text(text, analyzer=None):
    """分析英文文本内容"""
    from src.nlp.english_educational_analyzer import EnglishEducationalAnalyzer
    
    if analyzer is None:
        analyzer = EnglishEducationalAnalyzer()
    
    # 执行分析
    start_time = time.time()
    result = analyzer.analyze_text(text)
    end_time = time.time()
    
    # 添加性能数据
    result['analysis_time'] = round(end_time - start_time, 3)
    
    return result

def analyze_english_audio_file(file_path):
    """分析英文音频文件"""
    # 检查是否存在对应的转录文本
    txt_path = f"{file_path}.txt"
    
    if not os.path.exists(txt_path):
        logger.error(f"找不到音频对应的转录文本: {txt_path}")
        return None
    
    # 读取转录文本内容
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 解析文本内容
    expected_score = None
    content = ""
    for i, line in enumerate(lines):
        if line.startswith("Expected Risk Score:"):
            try:
                expected_score = float(line.split(":")[1].strip())
            except ValueError:
                logger.warning(f"无法解析预期风险分数: {line}")
        elif line.startswith("Content:"):
            # 内容在Content:标记之后的所有行
            content = ''.join(lines[i+1:]).strip()
            break
    
    if not content:
        logger.error(f"找不到有效的文本内容: {txt_path}")
        return None
    
    # 分析文本内容
    result = analyze_english_text(content)
    
    # 添加文件信息
    result['file_path'] = file_path
    result['expected_score'] = expected_score
    
    return result

def analyze_english_test_directory(directory):
    """分析目录中的所有英文测试音频文件"""
    if not os.path.exists(directory):
        logger.error(f"测试目录不存在: {directory}")
        return []
    
    logger.info(f"分析测试目录: {directory}")
    
    # 创建分析器实例（避免重复创建）
    from src.nlp.english_educational_analyzer import EnglishEducationalAnalyzer
    analyzer = EnglishEducationalAnalyzer()
    
    results = []
    wav_files = [f for f in os.listdir(directory) if f.endswith(".wav")]
    
    for wav_file in wav_files:
        wav_path = os.path.join(directory, wav_file)
        logger.info(f"分析文件: {wav_file}")
        
        result = analyze_english_audio_file(wav_path)
        if result:
            results.append(result)
    
    return results

def print_analysis_results(results):
    """打印分析结果"""
    if not results:
        logger.info("没有分析结果可显示")
        return
    
    print("\n" + "="*80)
    print(" 英文教育内容风险分析结果 ".center(80, "="))
    print("="*80)
    
    for i, result in enumerate(results):
        print(f"\n[{i+1}] 文件: {os.path.basename(result.get('file_path', 'N/A'))}")
        print(f"风险级别: {result['risk_level'].upper()} (得分: {result['risk_score']})")
        
        expected = result.get('expected_score')
        if expected is not None:
            diff = abs(result['risk_score'] - expected)
            print(f"预期风险分数: {expected} (差异: {diff:.1f})")
        
        print(f"分析耗时: {result.get('analysis_time', 'N/A')} 秒")
        
        if result.get('detected_keywords'):
            print("\n检测到的关键词:")
            for kw in result['detected_keywords'][:5]:  # 仅显示前5个
                print(f"- {kw['keyword']} (级别: {kw['level']}, 权重: {kw['weight']})")
            
            if len(result['detected_keywords']) > 5:
                print(f"... 以及 {len(result['detected_keywords']) - 5} 个更多关键词")
        
        if result.get('key_segments'):
            print("\n关键文本片段:")
            for i, segment in enumerate(result['key_segments'][:3]):  # 仅显示前3个
                print(f"- [{i+1}] {segment['text'][:100]}...")
            
            if len(result['key_segments']) > 3:
                print(f"... 以及 {len(result['key_segments']) - 3} 个更多片段")
        
        print("-"*80)
    
    # 打印总结
    print("\n总结:")
    print(f"分析了 {len(results)} 个文件")
    
    # 风险级别统计
    risk_counts = {}
    for level in ['critical', 'high', 'medium', 'low', 'safe']:
        count = len([r for r in results if r['risk_level'] == level])
        if count > 0:
            risk_counts[level] = count
    
    print(f"风险级别分布: {risk_counts}")
    
    # 平均分数
    avg_score = sum(r['risk_score'] for r in results) / len(results)
    print(f"平均风险分数: {avg_score:.2f}")
    
    print("="*80 + "\n")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="英文教育内容分析器测试工具")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="直接分析给定的英文文本")
    group.add_argument("--file", help="分析单个英文音频文件")
    group.add_argument("--dir", help="分析目录中的所有英文音频文件")
    
    parser.add_argument("--json", action="store_true", help="以JSON格式输出详细结果")
    parser.add_argument("--output", help="将JSON结果保存到指定文件")
    
    args = parser.parse_args()
    
    results = []
    
    if args.text:
        logger.info("分析文本内容")
        result = analyze_english_text(args.text)
        results.append(result)
    
    elif args.file:
        logger.info(f"分析音频文件: {args.file}")
        result = analyze_english_audio_file(args.file)
        if result:
            results.append(result)
    
    elif args.dir:
        logger.info(f"分析目录: {args.dir}")
        results = analyze_english_test_directory(args.dir)
    
    # 打印结果
    if not args.json:
        print_analysis_results(results)
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    
    # 保存JSON结果
    if args.output and results:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"已将结果保存到: {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
