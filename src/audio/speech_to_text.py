#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音转文本模块
"""

import os
import io
import logging
import tempfile
import numpy as np
import torch
import whisper
from typing import Optional
from threading import Lock
from functools import lru_cache

logger = logging.getLogger(__name__)

class SpeechToText:
    """优化的语音识别模块"""
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = self._load_model()
            self.initialized = True
    
    @lru_cache(maxsize=1)
    def _load_model(self, model_name: str = 'base') -> whisper.Whisper:
        """加载并缓存Whisper模型"""
        return whisper.load_model(model_name, device=self.device)
    
    def transcribe(self, audio_data: np.ndarray, 
                  language: str = 'zh') -> Optional[str]:
        """转录音频为文本"""
        try:
            # 确保音频数据类型正确
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # 使用Whisper进行转录
            result = self.model.transcribe(
                audio_data,
                language=language,
                task='transcribe',
                fp16=torch.cuda.is_available()
            )
            
            return result['text'].strip()
            
        except Exception as e:
            print(f'语音识别错误: {str(e)}')
            return None
    
    @staticmethod
    def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
        """音频数据归一化"""
        if len(audio_data) == 0:
            return audio_data
        
        # 去除直流偏置
        audio_data = audio_data - np.mean(audio_data)
        
        # 归一化到 [-1, 1] 范围
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
            
        return audio_data
    
    def release(self):
        """释放资源"""
        if hasattr(self, 'model'):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    def __del__(self):
        self.release()
