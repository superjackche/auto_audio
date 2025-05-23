#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音频录制模块
"""

import os
import time
import wave
import threading
import queue
import numpy as np
import pyaudio
import sounddevice as sd
import logging
from collections import deque
from typing import Callable, Optional

logger = logging.getLogger(__name__)

class AudioBuffer:
    """音频缓冲池，用于优化内存使用"""
    def __init__(self, maxlen: int = 100):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()
    
    def append(self, data: np.ndarray):
        with self.lock:
            self.buffer.append(data)
    
    def get_all(self) -> list:
        with self.lock:
            data = list(self.buffer)
            self.buffer.clear()
            return data

class AudioRecorder:
    """优化的音频录制器"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.sample_rate = 16000
            self.channels = 1
            self.chunk_size = 1024
            self.audio_buffer = AudioBuffer()
            self.processing_queue = queue.Queue()
            self.is_recording = False
            self.stream = None
            self.processing_thread = None
            self.vad_threshold = 0.01
            self.initialized = True
    
    def audio_callback(self, indata: np.ndarray, frames: int, 
                      time_info: dict, status: sd.CallbackFlags) -> None:
        """音频回调函数，进行初步的降噪和语音活动检测"""
        if status:
            print(f'音频回调状态: {status}')
            return
            
        # 将音频数据转换为单声道浮点数
        audio_data = indata[:, 0] if self.channels > 1 else indata.flatten()
        
        # 简单的噪声门限
        energy = np.mean(np.abs(audio_data))
        if energy > self.vad_threshold:
            # 简单的噪声消除
            audio_data = np.where(
                np.abs(audio_data) < self.vad_threshold,
                np.zeros_like(audio_data),
                audio_data
            )
            self.audio_buffer.append(audio_data)
    
    def process_audio(self, callback: Callable):
        """在单独的线程中处理音频数据"""
        while self.is_recording:
            try:
                # 获取累积的音频数据
                audio_data = self.audio_buffer.get_all()
                if audio_data:
                    # 合并音频片段
                    combined_audio = np.concatenate(audio_data)
                    # 调用回调函数处理音频
                    if callback:
                        callback(combined_audio)
            except Exception as e:
                print(f'音频处理错误: {str(e)}')
            threading.Event().wait(0.1)  # 短暂休眠以降低CPU使用率
    
    def start(self, callback: Optional[Callable] = None):
        """启动录音"""
        if self.is_recording:
            return
            
        self.is_recording = True
        
        try:
            self.stream = sd.InputStream(
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self.audio_callback
            )
            self.stream.start()
            
            # 启动处理线程
            self.processing_thread = threading.Thread(
                target=self.process_audio,
                args=(callback,),
                daemon=True
            )
            self.processing_thread.start()
            
        except Exception as e:
            self.is_recording = False
            raise RuntimeError(f'启动录音失败: {str(e)}')
    
    def stop(self):
        """停止录音"""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None
        
        # 清空缓冲区
        self.audio_buffer = AudioBuffer()
    
    @property
    def is_active(self):
        """检查录音是否活跃"""
        return self.is_recording and self.stream is not None
