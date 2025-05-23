#!/usr/bin/env python3
"""
最简化的测试脚本 - 测试核心功能
"""

import sys
import os

print("=" * 50)
print("Python 版本检查")
print("=" * 50)
print(f"Python 版本: {sys.version}")
print(f"Python 可执行文件: {sys.executable}")

print("\n" + "=" * 50)
print("基础模块测试")
print("=" * 50)

# 测试基础包
try:
    import numpy as np
    print("✓ numpy 可用")
except ImportError as e:
    print(f"✗ numpy 不可用: {e}")

try:
    import jieba
    print("✓ jieba 可用")
    # 测试分词
    words = jieba.lcut("这是一个测试")
    print(f"  分词测试: {words}")
except ImportError as e:
    print(f"✗ jieba 不可用: {e}")

try:
    from faster_whisper import WhisperModel
    print("✓ faster-whisper 可用")
except ImportError as e:
    print(f"✗ faster-whisper 不可用: {e}")

try:
    import speech_recognition as sr
    print("✓ SpeechRecognition 可用")
except ImportError as e:
    print(f"✗ SpeechRecognition 不可用: {e}")

print("\n" + "=" * 50)
print("自定义模块测试")
print("=" * 50)

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from nlp.analyzer_simple import TextAnalyzer
    analyzer = TextAnalyzer()
    print("✓ 简化文本分析器加载成功")
    
    # 测试分析
    test_text = "这是一个测试。This is a test."
    result = analyzer.analyze(test_text)
    print(f"  测试结果: {result['word_count']} 词, 语言: {result['language']}")
    
except Exception as e:
    print(f"✗ 文本分析器失败: {e}")

try:
    from audio.speech_to_text_simple import SpeechToText
    stt = SpeechToText()
    print("✓ 简化语音识别器加载成功")
    
    engine_info = stt.get_engine_info()
    print(f"  引擎类型: {engine_info['engine_type']}")
    
except Exception as e:
    print(f"✗ 语音识别器失败: {e}")

print("\n" + "=" * 50)
print("交互式测试")
print("=" * 50)

try:
    from nlp.analyzer_simple import TextAnalyzer
    analyzer = TextAnalyzer()
    
    print("请输入文本进行分析（输入 'q' 退出）:")
    
    while True:
        try:
            text = input("\n输入文本> ").strip()
            if text.lower() in ['q', 'quit', 'exit']:
                break
                
            if not text:
                continue
                
            result = analyzer.analyze(text)
            print(f"\n分析结果:")
            print(f"  字符数: {result['char_count']}")
            print(f"  词数: {result['word_count']}")
            print(f"  语言: {result['language']}")
            print(f"  情感: {result['sentiment']}")
            if result['keywords']:
                print(f"  关键词: {', '.join(result['keywords'][:3])}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"分析出错: {e}")
            
except Exception as e:
    print(f"交互式测试失败: {e}")

print("\n系统测试完成!")
