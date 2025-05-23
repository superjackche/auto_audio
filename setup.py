import os
from setuptools import setup, find_packages

# 版本信息
VERSION = '0.1.0'

# 获取长描述
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="auto_audio",
    version=VERSION,    author="SuperJackChe",
    author_email="superjackche@example.com",
    description="课堂语义行为实时分析系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/superjackche/auto_audio",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "pyaudio>=0.2.11",
        "librosa>=0.8.1",
        "sounddevice>=0.4.4",
        "SpeechRecognition>=3.8.1",
        "openai-whisper>=1.0.0",
        "torch>=1.10.0",
        "spacy>=3.2.0",
        "transformers>=4.18.0",
        "jieba>=0.42.1",
        "flask>=2.0.1",
        "flask-socketio>=5.1.1",
        "plotly>=5.6.0",
        "sqlalchemy>=1.4.32"
    ],
    entry_points={
        'console_scripts': [
            'auto_audio=auto_audio.main:main',
        ],
    },
    include_package_data=True,
    zip_safe=False
)
