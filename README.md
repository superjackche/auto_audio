# Auto Audio 系统

**Python 3.13.2 兼容版本**

一个简化的音频分析和文本处理系统，专为 Python 3.13.2 环境优化。

## 🚀 快速开始

### 方法1：直接运行（推荐）

```bash
python launch.py
```

### 方法2：使用原始启动脚本

```bash  
python run.py
```

## 🎯 主要功能

- **文本分析**: 中英文分词、关键词提取、情感分析
- **语言检测**: 自动识别中文、英文或混合文本
- **词频统计**: 高频词汇和词频分析
- **交互式界面**: 命令行交互式文本分析
- **Web界面**: 基于Flask的Web应用（可选）

## 📋 系统要求

- Python 3.13.2+
- Windows/Linux/macOS

## 📦 依赖安装

### 基础依赖（必需）
```bash
pip install numpy
```

### 语音识别依赖（可选）
```bash
# 推荐：faster-whisper（性能最佳）
pip install faster-whisper

# 或者：OpenAI Whisper
pip install openai-whisper

# 或者：SpeechRecognition
pip install SpeechRecognition
```

### 全部依赖
```bash
pip install -r requirements.txt
```

## 🎮 使用方法

### 1. 交互式文本分析

运行 `python launch.py` 并选择模式1：

```
📝 请输入文本 > 这是一个测试文本

📊 分析结果:
   📏 字符数: 8
   🔤 词数: 4  
   🌐 语言: chinese
   😊 情感: neutral
   🔑 关键词: 测试, 文本
   📈 高频词: 测试, 文本
   ✂️ 分词: 这是 | 一个 | 测试 | 文本
```

### 2. Web界面模式

运行 `python launch.py` 并选择模式2，然后访问 http://localhost:5000

### 3. 可用命令

在交互模式中：
- `help` - 显示帮助信息
- `quit` - 退出程序
- 直接输入文本进行分析

## 🛠️ 核心模块

### 文本分析器 (`src/nlp/analyzer_independent.py`)

- **独立设计**: 不依赖外部NLP库
- **中英文支持**: 原生支持中英文混合文本
- **词频统计**: 智能词频分析和关键词提取
- **情感分析**: 基于词典的情感倾向分析
- **语言检测**: 自动识别文本语言

### 语音识别 (`src/audio/speech_to_text_simple.py`)

- **多引擎支持**: faster-whisper、openai-whisper、SpeechRecognition
- **自动降级**: 智能选择最佳可用引擎
- **音频格式**: 支持多种音频输入格式

## 📁 项目结构

```
auto_audio/
├── launch.py              # 主启动脚本 ⭐
├── run.py                 # 原始启动脚本
├── requirements.txt       # 依赖列表
├── cleanup.py            # 项目清理脚本
├── src/                  # 源代码
│   ├── nlp/
│   │   ├── analyzer_independent.py  # 独立文本分析器 ⭐
│   │   └── analyzer_simple.py       # 简化分析器
│   ├── audio/
│   │   └── speech_to_text_simple.py # 简化语音识别 ⭐
│   ├── web/
│   │   └── app.py                   # Web应用
│   └── utils/
│       └── config_loader.py         # 配置加载器
├── config/               # 配置文件
├── data/                # 数据目录
└── docs/                # 文档
```

## 🔧 故障排除

### 1. 语音识别模块加载失败

如果看到 `module 'pkgutil' has no attribute 'ImpImporter'` 错误：

这是某些第三方库与Python 3.13兼容性问题。系统会自动降级到独立模式。

### 2. 中文分词效果不佳

系统使用独立分词器，如需更好效果可尝试：
```bash
pip install jieba --upgrade
```

### 3. Web界面无法启动

检查Flask依赖：
```bash
pip install flask flask-socketio
```

## 🆕 更新说明

**v2.0 - Python 3.13.2 兼容版本**

- ✅ 完全兼容 Python 3.13.2
- ✅ 独立文本分析器，无外部依赖
- ✅ 简化的启动流程
- ✅ 智能引擎选择
- ✅ 清理了冗余文件
- ✅ 改进的用户界面

## 📄 许可证

本课堂语义行为实时分析系统根据 MIT 许可证授权。

Copyright (c) 2025 superjackche

有关完整的许可证文本，请参阅项目根目录下的 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交Issues和Pull Requests！

