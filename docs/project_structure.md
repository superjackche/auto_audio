# 课堂语义行为实时分析系统 - 项目结构

本文档介绍了系统的整体结构，以帮助用户和开发人员了解各个组件和文件的作用。

## 项目目录结构

```
auto_audio/                    # 项目根目录
├── main.py                    # 核心分析引擎入口
├── run.py                     # 完整系统入口（含Web界面）
├── setup.py                   # 包安装配置文件
├── requirements.txt           # 依赖库列表
├── README.md                  # 项目说明文档
├── start.bat                  # Windows批处理启动脚本
├── start.ps1                  # PowerShell启动脚本
│
├── config/                    # 配置文件目录
│   ├── auth.yaml              # 认证配置
│   ├── config.yaml            # 系统配置
│   ├── keywords.txt           # 中文关键词库
│   └── english_keywords.txt   # 英文关键词库
│
├── data/                      # 数据目录
│   ├── logs/                  # 日志文件
│   ├── models/                # 模型文件
│   ├── reports/               # 分析报告
│   └── test_audio/            # 测试音频
│
├── docs/                      # 文档目录
│   ├── user_guide.md          # 用户指南
│   └── entry_files.md         # 入口文件说明
│
├── scripts/                   # 脚本工具
│   └── setup_dev.ps1          # 开发环境设置脚本
│
├── src/                       # 源代码目录
│   ├── analysis/              # 分析模块
│   │   └── risk_assessment.py # 风险评估
│   │
│   ├── audio/                 # 音频处理模块
│   │   ├── recorder.py        # 音频录制
│   │   └── speech_to_text.py  # 语音识别
│   │
│   ├── nlp/                   # 自然语言处理模块
│   │   ├── analyzer.py        # 文本分析
│   │   └── bilingual_analyzer.py # 双语分析
│   │
│   ├── utils/                 # 工具类
│   │   └── config_loader.py   # 配置加载器
│   │
│   └── web/                   # Web界面模块
│       ├── app.py             # Flask应用
│       ├── static/            # 静态文件
│       └── templates/         # 页面模板
│
├── tests/                     # 测试目录
│   └── test_audio.py          # 音频测试
│
└── tools/                     # 工具脚本
    ├── download_test_audio.py # 下载测试音频
    └── test_speech_recognition.py # 测试语音识别
```

## 主要入口文件

系统提供多个入口文件，选择合适的方式启动系统：

- **start.bat**: Windows用户的一键启动脚本（双击运行）
- **start.ps1**: PowerShell启动脚本（右键选择"使用PowerShell运行"）
- **run.py**: 完整系统入口，包含Web界面和核心功能
- **main.py**: 核心分析引擎入口，命令行模式

详细的启动说明请参见 [入口文件说明](entry_files.md)。

## 核心模块说明

### 音频处理模块 (src/audio)

- **recorder.py**: 负责从麦克风捕获音频数据
- **speech_to_text.py**: 将音频转换为文本的语音识别模块

### 文本分析模块 (src/nlp)

- **analyzer.py**: 基础文本分析功能
- **bilingual_analyzer.py**: 支持中英文双语的分析模块

### 风险评估模块 (src/analysis)

- **risk_assessment.py**: 基于分析结果进行风险评估

### Web界面模块 (src/web)

- **app.py**: Flask Web应用
- **templates/**: HTML模板文件
- **static/**: CSS、JavaScript等静态资源

## 配置与数据

### 配置文件 (config/)

系统配置存储在 `config/` 目录中：

- **config.yaml**: 主要系统配置
- **auth.yaml**: 用户认证配置
- **keywords.txt**: 中文关键词列表
- **english_keywords.txt**: 英文关键词列表

### 数据目录 (data/)

系统运行过程中产生的数据存储在 `data/` 目录中：

- **logs/**: 系统日志
- **models/**: 下载的模型文件
- **reports/**: 分析报告
- **test_audio/**: 测试用的音频文件

## 系统启动流程

1. 用户通过入口文件启动系统
2. 系统加载配置文件
3. 初始化各个模块（音频、语音识别、分析、评估）
4. 如果启用Web界面，启动Flask服务器
5. 开始捕获和分析音频
6. 实时显示分析结果并进行风险评估

---

*文档最后更新: 2025年5月23日*
