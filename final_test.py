#!/usr/bin/env python3
"""
最终简化测试脚本 - 使用独立模块
适用于Python 3.13.2
"""

import sys
import os

def main():
    print("=" * 60)
    print("Auto Audio 系统 - Python 3.13.2 兼容版本")
    print("=" * 60)
    print(f"Python 版本: {sys.version}")
    
    # 添加路径
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    print("\\n1. 测试独立文本分析器...")
    try:
        from nlp.analyzer_independent import IndependentTextAnalyzer
        
        analyzer = IndependentTextAnalyzer()
        info = analyzer.get_analyzer_info()
        print(f"   ✓ 分析器类型: {info['tokenizer_type']}")
        print(f"   ✓ 状态: {'可用' if info['available'] else '不可用'}")
        
        # 测试分析功能
        test_cases = [
            "这是一个很好的测试文本，用来验证中文分析功能。",
            "This is an excellent test text for English analysis.",
            "这是一个中英文混合的文本 with both Chinese and English words."
        ]
        
        for i, text in enumerate(test_cases, 1):
            print(f"\\n   测试 {i}: {text[:30]}...")
            result = analyzer.analyze(text)
            print(f"     - 词数: {result['word_count']}")
            print(f"     - 语言: {result['language']}")
            print(f"     - 情感: {result['sentiment']}")
            if result['keywords']:
                print(f"     - 关键词: {', '.join(result['keywords'][:3])}")
        
    except Exception as e:
        print(f"   ✗ 文本分析器测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\\n2. 测试语音识别引擎检测...")
    try:
        # 测试各种语音识别引擎的可用性
        engines = []
        
        try:
            from faster_whisper import WhisperModel
            engines.append("faster-whisper")
        except ImportError:
            pass
        
        try:
            import whisper
            engines.append("openai-whisper")
        except ImportError:
            pass
        
        try:
            import speech_recognition as sr
            engines.append("SpeechRecognition")
        except ImportError:
            pass
        
        if engines:
            print(f"   ✓ 可用的语音识别引擎: {', '.join(engines)}")
        else:
            print("   ⚠ 没有可用的语音识别引擎")
            
    except Exception as e:
        print(f"   ✗ 语音识别引擎检测失败: {e}")
    
    print("\\n3. 启动交互式文本分析...")
    try:
        from nlp.analyzer_independent import IndependentTextAnalyzer
        analyzer = IndependentTextAnalyzer()
        
        print("\\n" + "="*50)
        print("交互式文本分析器")
        print("输入文本进行分析，输入 'quit' 退出")
        print("="*50)
        
        while True:
            try:
                text = input("\\n请输入文本> ").strip()
                
                if text.lower() in ['quit', 'q', 'exit', '退出']:
                    print("退出分析器")
                    break
                
                if not text:
                    print("请输入有效文本")
                    continue
                
                # 分析文本
                result = analyzer.analyze(text)
                
                print(f"\\n📊 分析结果:")
                print(f"   字符数: {result['char_count']}")
                print(f"   词数: {result['word_count']}")
                print(f"   语言: {result['language']}")
                print(f"   情感: {result['sentiment']}")
                
                if result['keywords']:
                    print(f"   关键词: {', '.join(result['keywords'][:5])}")
                
                if result['word_frequency']:
                    print(f"   高频词: {', '.join(list(result['word_frequency'].keys())[:3])}")
                
                # 显示分词结果（前10个）
                if result['tokens']:
                    tokens_preview = result['tokens'][:10]
                    print(f"   分词预览: {' | '.join(tokens_preview)}")
                
            except KeyboardInterrupt:
                print("\\n\\n用户中断，退出分析器")
                break
            except Exception as e:
                print(f"分析出错: {e}")
        
    except Exception as e:
        print(f"   ✗ 交互式分析器启动失败: {e}")
    
    print("\\n" + "="*60)
    print("系统功能测试完成！")
    print("="*60)

if __name__ == "__main__":
    main()
