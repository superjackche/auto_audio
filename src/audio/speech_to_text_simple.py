"""
简化的语音识别模块 - 为Python 3.13.2优化
使用可用的最佳语音识别引擎
"""

import os
import tempfile
import logging
from typing import Union, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

class SpeechToText:
    """简化的语音识别类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化语音识别器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.engine = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """初始化最佳可用的语音识别引擎"""
        
        # 尝试faster-whisper（推荐）
        try:
            from faster_whisper import WhisperModel
            model_size = self.config.get('whisper_model', 'base')
            self.engine = WhisperModel(model_size, device="cpu", compute_type="int8")
            self.engine_type = 'faster_whisper'
            logger.info(f"使用faster-whisper引擎，模型: {model_size}")
            return
        except ImportError:
            logger.warning("faster-whisper不可用")
        except Exception as e:
            logger.warning(f"faster-whisper初始化失败: {e}")
        
        # 尝试openai-whisper
        try:
            import whisper
            model_size = self.config.get('whisper_model', 'base')
            self.engine = whisper.load_model(model_size)
            self.engine_type = 'openai_whisper'
            logger.info(f"使用openai-whisper引擎，模型: {model_size}")
            return
        except ImportError:
            logger.warning("openai-whisper不可用")
        except Exception as e:
            logger.warning(f"openai-whisper初始化失败: {e}")
        
        # 尝试SpeechRecognition
        try:
            import speech_recognition as sr
            self.engine = sr.Recognizer()
            self.engine_type = 'speech_recognition'
            logger.info("使用SpeechRecognition引擎")
            return
        except ImportError:
            logger.warning("SpeechRecognition不可用")
        except Exception as e:
            logger.warning(f"SpeechRecognition初始化失败: {e}")
        
        # 如果所有引擎都不可用，使用模拟模式
        self.engine = None
        self.engine_type = 'mock'
        logger.warning("所有语音识别引擎不可用，使用模拟模式")
    
    def transcribe(self, audio_data: Union[str, np.ndarray], **kwargs) -> str:
        """
        转录音频为文本
        
        Args:
            audio_data: 音频数据（文件路径或numpy数组）
            **kwargs: 其他参数
            
        Returns:
            识别的文本
        """
        if self.engine_type == 'mock':
            return "语音识别引擎未可用，这是模拟输出"
        
        try:
            # 处理音频数据
            audio_file = self._prepare_audio(audio_data)
            
            if self.engine_type == 'faster_whisper':
                return self._transcribe_faster_whisper(audio_file)
            elif self.engine_type == 'openai_whisper':
                return self._transcribe_openai_whisper(audio_file)
            elif self.engine_type == 'speech_recognition':
                return self._transcribe_speech_recognition(audio_file)
            
        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            return f"语音识别失败: {str(e)}"
    
    def _prepare_audio(self, audio_data: Union[str, np.ndarray]) -> str:
        """准备音频数据"""
        if isinstance(audio_data, str):
            # 如果是文件路径，直接返回
            if os.path.exists(audio_data):
                return audio_data
            else:
                raise FileNotFoundError(f"音频文件不存在: {audio_data}")
        
        elif isinstance(audio_data, np.ndarray):
            # 如果是numpy数组，保存为临时文件
            try:
                import soundfile as sf
            except ImportError:
                # 如果soundfile不可用，尝试使用scipy
                try:
                    from scipy.io import wavfile
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    # 确保数据在正确范围内
                    if audio_data.dtype != np.int16:
                        audio_data = (audio_data * 32767).astype(np.int16)
                    wavfile.write(temp_file.name, 16000, audio_data)
                    temp_file.close()
                    return temp_file.name
                except ImportError:
                    raise ImportError("需要安装soundfile或scipy来处理numpy音频数据")
            
            # 使用soundfile保存
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(temp_file.name, audio_data, 16000)
            temp_file.close()
            return temp_file.name
        
        else:
            raise ValueError("不支持的音频数据类型")
    
    def _transcribe_faster_whisper(self, audio_file: str) -> str:
        """使用faster-whisper进行转录"""
        segments, info = self.engine.transcribe(audio_file, beam_size=5)
        
        text = ""
        for segment in segments:
            text += segment.text
        
        # 清理临时文件
        if audio_file.startswith(tempfile.gettempdir()):
            try:
                os.unlink(audio_file)
            except:
                pass
        
        return text.strip()
    
    def _transcribe_openai_whisper(self, audio_file: str) -> str:
        """使用openai-whisper进行转录"""
        result = self.engine.transcribe(audio_file)
        
        # 清理临时文件
        if audio_file.startswith(tempfile.gettempdir()):
            try:
                os.unlink(audio_file)
            except:
                pass
        
        return result["text"].strip()
    
    def _transcribe_speech_recognition(self, audio_file: str) -> str:
        """使用SpeechRecognition进行转录"""
        import speech_recognition as sr
        
        with sr.AudioFile(audio_file) as source:
            audio = self.engine.record(source)
        
        try:
            # 优先使用Google Web Speech API
            text = self.engine.recognize_google(audio, language='zh-CN')
        except sr.UnknownValueError:
            try:
                # 备用：使用英文识别
                text = self.engine.recognize_google(audio, language='en-US')
            except sr.UnknownValueError:
                text = "无法识别音频内容"
        except sr.RequestError as e:
            text = f"语音识别服务错误: {e}"
        
        # 清理临时文件
        if audio_file.startswith(tempfile.gettempdir()):
            try:
                os.unlink(audio_file)
            except:
                pass
        
        return text.strip()
    
    def is_available(self) -> bool:
        """检查语音识别是否可用"""
        return self.engine_type != 'mock'
    
    def get_engine_info(self) -> Dict[str, str]:
        """获取引擎信息"""
        return {
            'engine_type': self.engine_type,
            'available': self.is_available()
        }
