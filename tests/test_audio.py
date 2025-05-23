#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试录音和语音识别功能
"""

import time
import numpy as np
from src.audio.recorder import AudioRecorder
from src.audio.speech_to_text import SpeechToText

def test_audio_system():
    recorder = AudioRecorder()
    stt = SpeechToText()
    
    def process_audio(audio_data):
        # 音频预处理
        audio_data = stt.normalize_audio(audio_data)
        
        # 语音识别
        text = stt.transcribe(audio_data)
        if text:
            print(f"识别结果: {text}")
    
    try:
        print("开始录音测试 (10秒)...")
        recorder.start(callback=process_audio)
        time.sleep(10)  # 录音10秒
        
    except KeyboardInterrupt:
        print("\n用户中断测试")
    
    finally:
        print("停止录音...")
        recorder.stop()
        stt.release()

if __name__ == '__main__':
    test_audio_system()
