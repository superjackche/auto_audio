#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音转文本模块 - 完整实现版本，支持多种语音识别库
"""

import os
import io
import logging
import tempfile
import numpy as np
import warnings
from typing import Optional, Union
from threading import Lock
import wave

# 配置日志
logger = logging.getLogger(__name__)

# 语音识别库优先级检测
WHISPER_AVAILABLE = False
WHISPER_TYPE = None
SPEECH_RECOGNITION_AVAILABLE = False

# 1. 尝试导入 faster-whisper (推荐，性能更好)
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
    WHISPER_TYPE = "faster"
    logger.info("✓ 使用 faster-whisper 进行语音识别")
except ImportError:
    # 2. 尝试导入 openai-whisper (备选)
    try:
        import whisper
        WHISPER_AVAILABLE = True
        WHISPER_TYPE = "openai"
        logger.info("✓ 使用 openai-whisper 进行语音识别")
    except ImportError:
        logger.warning("⚠ 未找到whisper库")

# 3. 尝试导入 speech_recognition 作为备用方案
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
    logger.info("✓ speech_recognition 库可用作备用方案")
except ImportError:
    logger.warning("⚠ speech_recognition 库不可用")

# 检查是否有可用的语音识别方案
if not WHISPER_AVAILABLE and not SPEECH_RECOGNITION_AVAILABLE:
    logger.error("✗ 未找到任何可用的语音识别库")
    logger.info("请安装以下库之一：")
    logger.info("  pip install faster-whisper  # 推荐")
    logger.info("  pip install openai-whisper")
    logger.info("  pip install SpeechRecognition")


class SpeechToText:
    """多引擎语音识别模块"""
    
    def __init__(self, model_name: str = "base", device: str = "auto"):
        """
        初始化语音识别器
        
        Args:
            model_name: Whisper模型名称 ("tiny", "base", "small", "medium", "large")
            device: 计算设备 ("auto", "cpu", "cuda")
        """
        self.model_name = model_name
        self.device = self._detect_device(device)
        self.whisper_model = None
        self.sr_recognizer = None
        self._lock = Lock()
        
        # 初始化可用的识别引擎
        self._init_engines()
        
    def _detect_device(self, device: str) -> str:
        """检测最佳计算设备"""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def _init_engines(self):
        """初始化语音识别引擎"""
        # 初始化 Whisper 模型
        if WHISPER_AVAILABLE:
            try:
                self._init_whisper()
                logger.info(f"✓ Whisper模型已加载: {self.model_name}")
            except Exception as e:
                logger.error(f"✗ Whisper初始化失败: {e}")
                
        # 初始化 SpeechRecognition
        if SPEECH_RECOGNITION_AVAILABLE:
            try:
                self.sr_recognizer = sr.Recognizer()
                logger.info("✓ SpeechRecognition已初始化")
            except Exception as e:
                logger.error(f"✗ SpeechRecognition初始化失败: {e}")
    
    def _init_whisper(self):
        """初始化Whisper模型"""
        with self._lock:
            if self.whisper_model is None:
                if WHISPER_TYPE == "faster":
                    # faster-whisper
                    self.whisper_model = WhisperModel(
                        self.model_name,
                        device=self.device,
                        compute_type="float16" if self.device == "cuda" else "int8"
                    )
                elif WHISPER_TYPE == "openai":
                    # openai-whisper
                    self.whisper_model = whisper.load_model(self.model_name, device=self.device)
    def transcribe(self, audio_data: Union[np.ndarray, str], language: str = "auto") -> Optional[str]:
        """
        转录音频为文本
        
        Args:
            audio_data: 音频数据(numpy数组)或音频文件路径
            language: 语言代码 ("zh", "en", "auto")，默认为自动检测
            
        Returns:
            转录的文本或None
        """
        try:
            # 如果是文件路径，直接处理
            if isinstance(audio_data, str) and os.path.exists(audio_data):
                return self._transcribe_file(audio_data, language)
            
            # 如果是numpy数组，先保存为临时文件
            elif isinstance(audio_data, np.ndarray):
                return self._transcribe_array(audio_data, language)
            
            else:
                logger.error("无效的音频数据格式")
                return None
                
        except Exception as e:
            logger.error(f"语音识别错误: {e}")
            return f"[错误] 语音识别失败: {str(e)}"
    
    def _transcribe_file(self, file_path: str, language: str) -> Optional[str]:
        """转录音频文件"""
        # 优先使用 Whisper
        if WHISPER_AVAILABLE and self.whisper_model is not None:
            return self._whisper_transcribe_file(file_path, language)
        
        # 备用：使用 SpeechRecognition
        elif SPEECH_RECOGNITION_AVAILABLE and self.sr_recognizer is not None:
            return self._sr_transcribe_file(file_path, language)
        
        else:
            logger.error("没有可用的语音识别引擎")
            return "[错误] 没有可用的语音识别库，请安装 faster-whisper 或 openai-whisper"
    
    def _transcribe_array(self, audio_data: np.ndarray, language: str) -> Optional[str]:
        """转录numpy音频数组"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # 保存音频到临时文件
            self._save_audio_array(audio_data, temp_path)
            
            # 转录临时文件
            result = self._transcribe_file(temp_path, language)
            
            return result
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _whisper_transcribe_file(self, file_path: str, language: str) -> Optional[str]:
        """使用Whisper转录文件"""
        try:
            if WHISPER_TYPE == "faster":
                # faster-whisper
                segments, info = self.whisper_model.transcribe(
                    file_path,
                    language=language if language != "auto" else None,
                    beam_size=5,
                    best_of=5
                )
                
                # 拼接所有片段
                text = "".join([segment.text for segment in segments])
                
            elif WHISPER_TYPE == "openai":
                # openai-whisper
                result = self.whisper_model.transcribe(
                    file_path,
                    language=language if language != "auto" else None
                )
                text = result["text"]
            
            # 清理文本
            text = text.strip()
            logger.info(f"Whisper识别结果: {text[:50]}...")
            
            return text if text else None
            
        except Exception as e:
            logger.error(f"Whisper转录失败: {e}")
            return None
    
    def _sr_transcribe_file(self, file_path: str, language: str) -> Optional[str]:
        """使用SpeechRecognition转录文件"""
        try:
            with sr.AudioFile(file_path) as source:
                audio = self.sr_recognizer.record(source)
            
            # 根据语言选择识别服务
            if language.startswith("zh"):
                # 中文识别
                text = self.sr_recognizer.recognize_google(audio, language="zh-CN")
            else:
                # 英文识别
                text = self.sr_recognizer.recognize_google(audio, language="en-US")
            
            logger.info(f"SpeechRecognition识别结果: {text[:50]}...")
            return text
            
        except sr.UnknownValueError:
            logger.warning("SpeechRecognition无法理解音频")
            return "[无法识别] 音频内容不清晰"
        except sr.RequestError as e:
            logger.error(f"SpeechRecognition服务错误: {e}")
            return f"[服务错误] {str(e)}"
        except Exception as e:
            logger.error(f"SpeechRecognition转录失败: {e}")
            return None
    
    def _save_audio_array(self, audio_data: np.ndarray, file_path: str, sample_rate: int = 16000):
        """将numpy音频数组保存为WAV文件"""
        # 归一化音频数据
        audio_data = self.normalize_audio(audio_data)
        
        # 转换为16位整数
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # 保存为WAV文件
        with wave.open(file_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
    
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
    
    def is_available(self) -> bool:
        """检查语音识别是否可用"""
        return WHISPER_AVAILABLE or SPEECH_RECOGNITION_AVAILABLE
    
    def get_available_engines(self) -> list:
        """获取可用的识别引擎列表"""
        engines = []
        if WHISPER_AVAILABLE:
            engines.append(f"Whisper ({WHISPER_TYPE})")
        if SPEECH_RECOGNITION_AVAILABLE:
            engines.append("SpeechRecognition")
        return engines
    
    def release(self):
        """释放资源"""
        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None
        logger.info("语音识别资源已释放")
    
    def __del__(self):
        """析构函数"""
        try:
            self.release()
        except:
            pass


# 创建全局实例
_speech_to_text_instance = None

def get_speech_to_text(model_name: str = "base") -> SpeechToText:
    """获取语音识别实例（单例模式）"""
    global _speech_to_text_instance
    if _speech_to_text_instance is None:
        _speech_to_text_instance = SpeechToText(model_name)
    return _speech_to_text_instance


if __name__ == "__main__":
    # 测试代码
    stt = SpeechToText()
    
    print(f"可用引擎: {stt.get_available_engines()}")
    print(f"是否可用: {stt.is_available()}")
    
    # 如果有可用引擎，可以进行测试
    if stt.is_available():
        print("语音识别功能正常，可以开始使用")
    else:
        print("请安装语音识别库：")
        print("  pip install faster-whisper")
        print("  或 pip install openai-whisper")
        print("  或 pip install SpeechRecognition")
