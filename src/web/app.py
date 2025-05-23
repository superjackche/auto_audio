#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Web界面主模块
"""

import os
import json
import logging
import threading
import time
import sys
import importlib.util
from datetime import datetime, timedelta
from functools import wraps
import yaml
import tempfile
import uuid
import numpy as np

from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from flask_socketio import SocketIO, emit
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

# 导入项目中的其他模块
from src.audio.recorder import AudioRecorder
from src.audio.speech_to_text import SpeechToText
from src.nlp.analyzer import SemanticAnalyzer
from src.nlp.educational_analyzer import EducationalSemanticAnalyzer
from src.nlp.unified_language_analyzer import UnifiedLanguageAnalyzer
from src.analysis.risk_assessment import RiskAssessor
from src.utils.config_loader import ConfigLoader

# 项目根目录
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

logger = logging.getLogger(__name__)

# 存储最近的分析结果
analysis_history = []
MAX_HISTORY_SIZE = 100
analysis_lock = threading.Lock()

# 初始化Flask应用
app = Flask(__name__, 
    template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), 'static')
)
socketio = SocketIO(app)

# 加载管理员配置
def load_admin_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              'config', 'auth.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

admin_config = load_admin_config()
app.secret_key = admin_config['session_secret_key']

def login_required(f):
    """登录检查装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('请先登录', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """管理员权限检查装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            flash('需要管理员权限', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def generate_demo_data():
    """生成演示数据用于监控大屏"""
    import random
    
    # 实时风险评分
    risk_scores = {
        'current': random.randint(15, 85),
        'average': random.randint(20, 50),
        'trend': random.choice(['up', 'down', 'stable'])
    }
    
    # 关键词监控数据
    keywords_data = [
        {'keyword': '意识形态', 'count': random.randint(0, 5), 'risk': 'high'},
        {'keyword': '价值观', 'count': random.randint(1, 8), 'risk': 'medium'},
        {'keyword': '文化交流', 'count': random.randint(3, 12), 'risk': 'low'},
        {'keyword': '学术自由', 'count': random.randint(0, 6), 'risk': 'medium'},
        {'keyword': '批判思维', 'count': random.randint(2, 10), 'risk': 'low'},
    ]
    
    # 实时活动流
    activities = [
        f"{random.choice(['张教授', '李老师', '王博士'])}在{random.choice(['国际贸易课堂', '文化比较课程', '政治经济学'])}中提到了敏感词汇",
        f"检测到{random.choice(['高风险', '中风险', '低风险'])}语义表达",
        f"AI分析完成，风险评分：{random.randint(10, 90)}",
        f"实时监控：{random.choice(['正常', '需关注', '高度警惕'])}状态"
    ]
    
    # 教师行为分析
    teacher_stats = {
        'total_sessions': random.randint(150, 300),
        'risk_sessions': random.randint(5, 25),
        'avg_risk_score': random.randint(20, 45),
        'improvement_rate': f"{random.randint(5, 25)}%"
    }
    
    return {
        'risk_scores': risk_scores,
        'keywords_data': keywords_data,
        'activities': activities[:4],  # 只显示最近4条
        'teacher_stats': teacher_stats,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def add_analysis_result(result):
    """添加新的分析结果到历史记录"""
    with analysis_lock:
        analysis_history.append(result)
        if len(analysis_history) > MAX_HISTORY_SIZE:
            analysis_history.pop(0)
        
        # 通过WebSocket发送实时更新
        socketio.emit('new_analysis', result)

def start_web_server(config):
    """
    启动Web服务器
    
    Args:
        config: Web服务器配置
        
    Returns:
        WebServer: Web服务器实例
    """
    if not config.get('enabled', True):
        logger.info("Web界面已禁用")
        return None
    
    host = config.get('host', '127.0.0.1')
    port = config.get('port', 5000)
    debug = config.get('debug', False)
    secret_key = config.get('secret_key', 'development_key')
    
    # 配置Flask应用
    app.config['SECRET_KEY'] = secret_key
    app.config['WEB_CONFIG'] = config
    
    # 启动Web服务器线程
    server_thread = threading.Thread(target=lambda: socketio.run(app, host=host, port=port, debug=debug))
    server_thread.daemon = True
    server_thread.start()
    
    logger.info(f"Web服务器已启动: http://{host}:{port}")
    
    return WebServer(server_thread)

class WebServer:
    """Web服务器类，用于管理Web服务的生命周期"""
    
    def __init__(self, thread):
        self.thread = thread
        self.running = True
    
    def stop(self):
        """停止Web服务器"""
        if self.running:
            self.running = False
            logger.info("正在关闭Web服务器...")
            # 实际上我们不能优雅地关闭线程中的Flask服务器
            # 这里依赖于守护线程在主程序结束时自动终止

# 路由定义
@app.route('/')
def index():
    """主页"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """登录页面"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == admin_config['admin_username'] and check_password_hash(
                admin_config['admin_password_hash'], password):
            session['user_id'] = username
            session['role'] = 'admin'
            flash('登录成功！', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('用户名或密码错误！', 'danger')
    
    return render_template('login.html', now=datetime.now())

@app.route('/logout')
def logout():
    """退出登录"""
    session.clear()
    flash('您已成功退出登录', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """仪表盘页面"""
    stats = {
        'total_records': len(analysis_history),
        'high_risk_count': sum(1 for item in analysis_history if item.get('risk_score', 0) > 70),
        'medium_risk_count': sum(1 for item in analysis_history if 30 < item.get('risk_score', 0) <= 70),
        'low_risk_count': sum(1 for item in analysis_history if item.get('risk_score', 0) <= 30),
        'recent_records': analysis_history[-5:] if analysis_history else []
    }
    return render_template('dashboard.html', stats=stats, now=datetime.now())

@app.route('/realtime')
@login_required
def realtime():
    """实时监控页面"""
    keywords = []
    keywords_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              'config', 'keywords.txt')
    if os.path.exists(keywords_path):
        with open(keywords_path, 'r', encoding='utf-8') as f:
            keywords = [line.strip() for line in f.readlines() if line.strip()]
    
    return render_template('realtime.html', keywords=keywords, now=datetime.now())

@app.route('/reports')
@login_required
def reports():
    """风险报告页面"""
    return render_template('reports.html', reports=analysis_history, now=datetime.now())

@app.route('/monitor')
@login_required
def monitor_dashboard():
    """实时监控大屏页面"""
    # 生成演示数据用于展示
    demo_data = generate_demo_data()
    return render_template('monitor-dashboard.html', 
                         demo_data=demo_data, 
                         now=datetime.now())

@app.route('/audio-upload')
@login_required
def audio_upload():
    """音频文件上传分析页面"""
    return render_template('audio-upload.html', now=datetime.now())

@app.route('/settings')
@login_required
@admin_required
def settings():
    """系统设置页面"""
    return render_template('settings.html', now=datetime.now())

# API路由
@app.route('/api/analysis/recent')
@login_required
def api_recent_analysis():
    """获取最近的分析结果"""
    with analysis_lock:
        return jsonify(analysis_history[-20:])

@app.route('/api/analysis/stats')
@login_required
def api_analysis_stats():
    """获取分析统计数据"""
    with analysis_lock:
        if not analysis_history:
            return jsonify({
                'total': 0,
                'risk_levels': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
                'trend': []
            })
        
        # 计算各风险级别的数量
        risk_levels = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for item in analysis_history:
            risk_level = item.get('risk_level', 'low')
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
        
        # 计算风险趋势（最近1小时的5分钟间隔）
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # 按时间分组
        timestamps = []
        scores = []
        for item in analysis_history:
            try:
                timestamp = datetime.fromisoformat(item.get('timestamp'))
                if timestamp >= hour_ago:
                    timestamps.append(timestamp)
                    scores.append(item.get('risk_score', 0))
            except (ValueError, TypeError):
                pass
        
        # 创建趋势数据
        trend = []
        if timestamps:
            df = pd.DataFrame({'timestamp': timestamps, 'score': scores})
            df['interval'] = df['timestamp'].dt.floor('5min')
            grouped = df.groupby('interval')['score'].mean().reset_index()
            
            for _, row in grouped.iterrows():
                trend.append({
                    'time': row['interval'].isoformat(),
                    'score': float(row['score'])
                })
        
        return jsonify({
            'total': len(analysis_history),
            'risk_levels': risk_levels,
            'trend': trend
        })

@app.route('/api/reports/list')
@login_required
def api_report_list():
    """获取报告列表"""
    reports_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'reports')
    reports = []
    
    try:
        for filename in os.listdir(reports_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(reports_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        report = json.load(f)
                        
                    reports.append({
                        'id': filename,
                        'timestamp': report.get('timestamp'),
                        'risk_level': report.get('risk_level'),
                        'risk_score': report.get('risk_score'),
                        'text': report.get('text', '')[:100]
                    })
                except:
                    pass
    except Exception as e:
        logger.exception(f"获取报告列表失败: {str(e)}")
    
    return jsonify(sorted(reports, key=lambda x: x.get('timestamp', ''), reverse=True))

@app.route('/api/reports/detail/<report_id>')
@login_required
def api_report_detail(report_id):
    """获取报告详情"""
    reports_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'reports')
    file_path = os.path.join(reports_dir, report_id)
    
    try:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            return jsonify(report)
    except Exception as e:
        logger.exception(f"获取报告详情失败: {str(e)}")
    
    return jsonify({'error': '报告不存在'}), 404

@app.route('/api/settings', methods=['POST'])
@admin_required
def api_update_settings():
    """更新系统设置"""
    # 这里可以实现设置更新逻辑
    return jsonify({'status': 'success'})

@app.route('/api/reports')
@login_required
def api_reports():
    """获取报告数据API"""
    return jsonify(analysis_history)

# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    """处理WebSocket连接"""
    if 'user_id' not in session:
        return False
    logger.info(f"用户 {session['user_id']} 已连接")
    emit('status', {'message': '已连接'})

@socketio.on('disconnect')
def handle_disconnect():
    """处理WebSocket断开连接"""
    if 'user_id' in session:
        logger.info(f"用户 {session['user_id']} 已断开连接")
    emit('status', {'message': '已断开连接'})

@socketio.on('toggle_recording')
def handle_toggle_recording(data):
    """处理录制开关切换"""
    if 'user_id' not in session:
        return
    
    recording = data.get('recording', False)
    user_id = session['user_id']
    
    if recording:
        try:
            # 启动录音和分析线程
            recorder_thread = threading.Thread(
                target=start_audio_processing,
                args=(user_id,),
                daemon=True
            )
            recorder_thread.start()
            emit('status', {'message': '开始录制'})
        except Exception as e:
            logger.error(f"启动录音失败: {str(e)}")
            emit('error', {'message': '启动录音失败'})
    else:
        try:
            # 停止录音和分析线程
            stop_audio_processing(user_id)
            emit('status', {'message': '停止录制'})
        except Exception as e:
            logger.error(f"停止录音失败: {str(e)}")
            emit('error', {'message': '停止录音失败'})

@socketio.on('start_monitoring')
def handle_start_monitoring():
    """开始麦克风监控"""
    if 'user_id' not in session:
        return
    
    user_id = session['user_id']
    logger.info(f"用户 {user_id} 开始麦克风监控")
    
    try:
        # 启动录音和分析线程
        recorder_thread = threading.Thread(
            target=start_audio_processing,
            args=(user_id,),
            daemon=True
        )
        recorder_thread.start()
        emit('status', {'message': '麦克风监控已开始'})
        
    except Exception as e:
        logger.error(f"启动麦克风监控失败: {str(e)}")
        emit('error', {'message': f'启动麦克风监控失败: {str(e)}'})

@socketio.on('stop_monitoring')
def handle_stop_monitoring():
    """停止麦克风监控"""
    if 'user_id' not in session:
        return
    
    user_id = session['user_id']
    logger.info(f"用户 {user_id} 停止麦克风监控")
    
    try:
        # 停止录音流程
        stop_audio_processing(user_id)
        emit('status', {'message': '麦克风监控已停止'})
        
    except Exception as e:
        logger.error(f"停止麦克风监控失败: {str(e)}")
        emit('error', {'message': f'停止麦克风监控失败: {str(e)}'})

def start_audio_processing(user_id):
    """启动音频处理流程"""
    from ..audio.recorder import AudioRecorder
    from ..audio.speech_to_text import SpeechToText
    from ..nlp.bilingual_analyzer import BilingualTextAnalyzer
    from ..analysis.risk_assessment import RiskAssessor
    try:
        recorder = AudioRecorder()
        stt = SpeechToText()
        analyzer = BilingualTextAnalyzer()
        risk_assessor = RiskAssessor()
        
        def process_audio(audio_data):
            # 语音识别
            text = stt.transcribe(audio_data)
            if text:
                # 发送转录文本
                socketio.emit('transcript', {
                    'text': text,
                    'timestamp': time.time()
                })
                
                # 语义分析
                analysis_result = analyzer.analyze(text)
                
                # 风险评估
                risk_score, risk_factors = risk_assessor.assess(
                    text=text,
                    analysis_result=analysis_result
                )
                  # 确定风险等级
                if risk_score < 0.3:
                    risk_level = "低风险"
                elif risk_score < 0.6:
                    risk_level = "中风险"
                elif risk_score < 0.8:
                    risk_level = "高风险"
                else:
                    risk_level = "极高风险"
                
                # 发送风险评估结果
                socketio.emit('risk_assessment', {
                    'risk_score': float(risk_score),
                    'risk_level': risk_level,
                    'risk_factors': risk_factors,
                    'timestamp': time.time()
                })
                
                # 同时发送一个统一的结果用于前端显示
                socketio.emit('speech_result', {
                    'text': text,
                    'risk_score': float(risk_score),
                    'risk_level': risk_level,
                    'risk_factors': risk_factors
                })
                  # 保存分析历史
                with analysis_lock:
                    analysis_history.append({
                        'text': text,
                        'risk_score': float(risk_score),
                        'risk_level': risk_level,
                        'risk_factors': risk_factors,
                        'timestamp': time.time()
                    })
                    if len(analysis_history) > MAX_HISTORY_SIZE:
                        analysis_history.pop(0)
        
        recorder.start(callback=process_audio)
        
    except Exception as e:
        logger.error(f"音频处理流程启动失败: {str(e)}")
        socketio.emit('error', {'message': '音频处理流程启动失败'})

def stop_audio_processing(user_id):
    """停止音频处理流程"""
    from ..audio.recorder import AudioRecorder
    
    try:
        recorder = AudioRecorder()
        recorder.stop()
    except Exception as e:
        logger.error(f"音频处理流程停止失败: {str(e)}")
        socketio.emit('error', {'message': '音频处理流程停止失败'})

@app.route('/api/audio/analyze', methods=['POST'])
@login_required
def api_audio_analyze():
    """音频文件分析API - 使用教育语义分析器"""
    if 'file' not in request.files:
        return jsonify({'error': '未提供音频文件'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '未选择音频文件'}), 400
    
    # 检查文件类型
    allowed_extensions = {'mp3', 'wav', 'ogg', 'm4a', 'flac'}
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
    if file_ext not in allowed_extensions:
        return jsonify({'error': f'不支持的文件格式。支持的格式: {", ".join(allowed_extensions)}'}), 400
    
    try:
        # 保存上传的文件
        filename = secure_filename(file.filename)
        temp_dir = os.path.join(ROOT_DIR, 'data', 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{filename}")
        file.save(file_path)
        
        logger.info(f"开始分析上传的音频文件: {filename}")
          # 使用演示分析器
        demo_analyzer_path = os.path.join(ROOT_DIR, 'tools', 'demo_analyzer.py')
        sys.path.insert(0, os.path.join(ROOT_DIR, 'tools'))
        
        # 动态导入
        spec = importlib.util.spec_from_file_location("demo_analyzer", demo_analyzer_path)
        demo_analyzer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo_analyzer_module)
        
        demo_analyzer = demo_analyzer_module.DemoAudioAnalyzer()
        
        # 设置分析上下文
        context = {
            'teacher': request.form.get('teacher', '演示教师'),
            'course': request.form.get('course', '演示课程'),
            'upload_mode': True,
            'filename': filename
        }
        
        # 进行分析
        result = demo_analyzer.analyze_audio_file(file_path, context)
        
        # 检查是否有错误
        if 'error' in result:
            logger.error(f"音频分析失败: {result['error']}")
            return jsonify(result), 500
        
        # 添加到历史记录
        analysis_record = {
            'timestamp': result['timestamp'],
            'filename': filename,
            'text': result['transcript']['text'],
            'risk_level': result['risk_assessment']['risk_level'],
            'risk_score': result['risk_assessment']['risk_score'],
            'teacher': context['teacher'],
            'course': context['course'],
            'analysis_type': 'file_upload'
        }
        
        add_analysis_result(analysis_record)
        
        logger.info(f"音频文件分析完成: {filename}, 风险等级: {result['risk_assessment']['risk_level']}")
        
        # 清理临时文件
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        logger.exception(f"音频文件分析失败: {str(e)}")
        
        # 清理临时文件
        try:
            if 'file_path' in locals():
                os.remove(file_path)
        except:
            pass
            
        return jsonify({'error': f'音频文件分析失败: {str(e)}'}), 500
        
        def process_result(result):
            results.append(result)
        
        # 启动录音和分析线程
        recorder = AudioRecorder()
        stt = SpeechToText()
        analyzer = SemanticAnalyzer()
        risk_assessor = RiskAssessor()
        
        def process_audio(audio_data):
            # 语音识别
            text = stt.transcribe(audio_data)
            if text:
                # 语义分析
                analysis_result = analyzer.analyze(text)
                
                # 风险评估
                risk_score, risk_factors = risk_assessor.assess(
                    text=text,
                    analysis_result=analysis_result
                )
                
                # 保存分析结果
                process_result({
                    'text': text,
                    'score': risk_score,
                    'factors': risk_factors,
                    'timestamp': time.time()
                })
        
        recorder.record(file_path, callback=process_audio)
        
        # 等待处理完成
        time.sleep(5)
        
        return jsonify({
            'status': 'success',
            'results': results
        })
    except Exception as e:
        logger.error(f"音频文件分析失败: {str(e)}")
        return jsonify({'error': '音频文件分析失败'}), 500

@app.route('/analyze_audio_file', methods=['POST'])
@login_required
def analyze_audio_file():
    """处理上传的音频文件并分析"""
    if 'audio_file' not in request.files:
        return jsonify({'error': '未找到音频文件'}), 400
        
    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    # 检查文件类型
    allowed_extensions = {'mp3', 'wav', 'ogg'}
    if not ('.' in audio_file.filename and 
            audio_file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return jsonify({'error': '不支持的文件格式'}), 400
    
    try:
        # 为上传的文件创建一个唯一的临时文件名并保存
        temp_dir = os.path.join(ROOT_DIR, 'data', 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        filename = secure_filename(audio_file.filename)
        file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{filename}")
        audio_file.save(file_path)
        
        # 使用音频文件分析器进行分析
        sys.path.insert(0, os.path.join(ROOT_DIR, 'tools'))
        from tools.analyze_audio_file import AudioFileAnalyzer
        analyzer = AudioFileAnalyzer()
        result = analyzer.analyze(file_path)
        
        # 如果分析成功，将结果添加到历史记录
        if 'error' not in result:
            result['timestamp'] = datetime.now().isoformat()
            add_analysis_result(result)
            logger.info(f"文件分析完成: {filename}, 风险等级: {result['risk_level']}")
        
        # 分析完成后删除临时文件
        try:
            os.remove(file_path)
        except:
            pass
            
        return jsonify(result)
        
    except Exception as e:
        logger.exception(f"音频文件分析失败: {str(e)}")
        return jsonify({'error': f'音频文件分析失败: {str(e)}'}), 500

# 如果作为主程序运行
if __name__ == '__main__':
    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'web.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("启动Web服务...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
