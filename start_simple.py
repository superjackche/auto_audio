#!/usr/bin/env python3
"""
Auto Audio 系统 - 简化启动脚本
适用于 Python 3.13.2
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(project_root / "data" / "logs" / "system.log")
        ]
    )

def main():
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("Auto Audio 系统启动中...")
    print(f"Python 版本: {sys.version}")
    print("=" * 60)
    
    try:
        # 测试语音识别模块
        print("\\n1. 测试语音识别模块...")
        try:
            from audio.speech_to_text_simple import SpeechToText
            stt = SpeechToText()
            engine_info = stt.get_engine_info()
            print(f"   语音识别引擎: {engine_info['engine_type']}")
            print(f"   状态: {'✓ 可用' if engine_info['available'] else '✗ 不可用'}")
        except Exception as e:
            print(f"   ✗ 语音识别模块加载失败: {e}")
        
        # 测试文本分析模块
        print("\\n2. 测试文本分析模块...")
        try:
            from nlp.analyzer_simple import TextAnalyzer
            analyzer = TextAnalyzer()
            analyzer_info = analyzer.get_analyzer_info()
            print(f"   文本分析器: {analyzer_info['tokenizer_type']}")
            print(f"   状态: {'✓ 可用' if analyzer_info['available'] else '✗ 不可用'}")
            
            # 测试分析功能
            test_text = "这是一个测试文本，用来验证分析功能。This is a test text for analysis."
            result = analyzer.analyze(test_text)
            print(f"   测试分析: 检测到 {result['word_count']} 个词，语言: {result['language']}")
            
        except Exception as e:
            print(f"   ✗ 文本分析模块加载失败: {e}")
        
        # 测试配置加载
        print("\\n3. 测试配置模块...")
        try:
            from utils.config_loader import ConfigLoader
            config_loader = ConfigLoader()
            print("   ✓ 配置加载器初始化成功")
        except Exception as e:
            print(f"   ✗ 配置模块加载失败: {e}")
        
        # 启动Web界面
        print("\\n4. 启动Web界面...")
        try:
            from web.app import app
            
            print("\\n系统启动完成！")
            print("Web界面地址: http://localhost:5000")
            print("按 Ctrl+C 停止服务")
            print("=" * 60)
            
            app.run(host='0.0.0.0', port=5000, debug=False)
            
        except KeyboardInterrupt:
            print("\\n\\n系统已停止")
            
        except Exception as e:
            print(f"\\n✗ Web界面启动失败: {e}")
            print("\\n尝试简单测试模式...")
            
            # 简单测试模式
            simple_test_mode()
    
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        print(f"\\n✗ 系统启动失败: {e}")
        return 1
    
    return 0

def simple_test_mode():
    """简单测试模式"""
    print("\\n" + "=" * 40)
    print("进入简单测试模式")
    print("=" * 40)
    
    try:
        from nlp.analyzer_simple import TextAnalyzer
        analyzer = TextAnalyzer()
        
        print("\\n请输入要分析的文本（输入 'quit' 退出）:")
        
        while True:
            try:
                text = input("\\n> ").strip()
                if text.lower() in ['quit', 'exit', '退出']:
                    break
                
                if not text:
                    continue
                
                # 分析文本
                result = analyzer.analyze(text)
                
                print(f"\\n分析结果:")
                print(f"  字符数: {result['char_count']}")
                print(f"  词数: {result['word_count']}")
                print(f"  语言: {result['language']}")
                print(f"  情感: {result['sentiment']}")
                print(f"  关键词: {', '.join(result['keywords'][:5])}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"分析失败: {e}")
        
        print("\\n测试模式结束")
        
    except Exception as e:
        print(f"测试模式启动失败: {e}")

if __name__ == "__main__":
    sys.exit(main())
