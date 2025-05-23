#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试Whisper语音识别实现
此脚本用于验证系统中安装的Whisper语音识别实现
"""

import os
import sys
import importlib
import platform

def check_whisper_implementation():
    """检查系统中安装了哪种Whisper实现"""
    print("检查Whisper语音识别实现...\n")
    
    # 检查OpenAI Whisper
    try:
        import whisper
        print("✓ OpenAI Whisper 已安装")
        print(f"  版本: {whisper.__version__}")
        
        # 测试基本功能
        try:
            print("  正在加载模型 (这可能需要一些时间)...")
            model = whisper.load_model("tiny")
            print("  模型加载成功")
            has_openai_whisper = True
        except Exception as e:
            print(f"  警告: 模型加载失败: {str(e)}")
            has_openai_whisper = False
    except ImportError:
        print("✗ OpenAI Whisper 未安装")
        has_openai_whisper = False
    
    print()
    
    # 检查Faster Whisper
    try:
        from faster_whisper import WhisperModel
        print("✓ Faster Whisper 已安装")
        
        # 测试基本功能
        try:
            print("  正在加载模型 (这可能需要一些时间)...")
            model = WhisperModel("tiny", device="cpu")
            print("  模型加载成功")
            has_faster_whisper = True
        except Exception as e:
            print(f"  警告: 模型加载失败: {str(e)}")
            has_faster_whisper = False
    except ImportError:
        print("✗ Faster Whisper 未安装")
        has_faster_whisper = False
    
    print("\n总结:")
    if has_openai_whisper:
        print("- OpenAI Whisper 可用")
    if has_faster_whisper:
        print("- Faster Whisper 可用")
    
    if not has_openai_whisper and not has_faster_whisper:
        print("! 警告: 未安装任何Whisper实现")
        print("  请运行以下命令安装:")
        print("  - OpenAI Whisper: pip install openai-whisper>=1.0.0")
        print("  - 或Faster Whisper: pip install faster-whisper>=0.9.0 (推荐)")
    else:
        print("\n您的系统已准备好使用语音识别功能")

def main():
    """主函数"""
    print("\n===== Whisper语音识别测试 =====\n")
    
    # 系统信息
    print(f"Python版本: {sys.version}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"处理器: {platform.processor()}")
    
    # 检查Whisper实现
    check_whisper_implementation()
    
    print("\n===============================\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
