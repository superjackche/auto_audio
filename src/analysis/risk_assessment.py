#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
风险评估模块
"""

import os
import json
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class RiskAssessor:
    """风险评估器，评估语义分析结果并生成风险报告"""
    
    def __init__(self, config=None):
        """
        初始化风险评估器
        
        Args:
            config: 风险评估配置字典
        """
        self.config = config or {}
        
        # 风险阈值设置
        self.low_threshold = self.config.get('low_threshold', 0.3)
        self.medium_threshold = self.config.get('medium_threshold', 0.6)
        self.high_threshold = self.config.get('high_threshold', 0.8)
        self.alert_threshold = self.config.get('alert_threshold', 0.7)
        
        # 预警方法
        self.alert_method = self.config.get('alert_method', ['console'])
        
        # 初始化报告路径
        self.report_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'reports')
        os.makedirs(self.report_path, exist_ok=True)
        
        logger.info("风险评估器初始化完成")
    
    def assess(self, text, analysis_result=None):
        """
        评估语义分析结果
        
        Args:
            text: 原始文本内容
            analysis_result: 语义分析结果
            
        Returns:
            tuple: (风险评分, 风险因素列表)
        """
        # 如果没有传入分析结果，返回零风险
        if not analysis_result:
            logger.warning("无效的分析结果")
            return 0.0, []
        
        # 提取风险分数和因素
        risk_score = analysis_result.get('risk_score', 0.0)
        risk_factors = analysis_result.get('risk_factors', [])
        
        # 确定风险级别
        risk_level = self._determine_risk_level(risk_score)
        
        # 生成风险结果
        result = {
            'timestamp': datetime.now().isoformat(),
            'text': text,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'keyword_matches': analysis_result.get('keyword_matches', []),
            'details': {
                'keyword_score': analysis_result.get('keyword_score', 0.0),
                'semantic_score': analysis_result.get('semantic_score', 0.0),
                'consistency_score': analysis_result.get('consistency_score', 0.0),
            }
        }
        
        # 如果风险超过警报阈值，触发警报
        if risk_score >= self.alert_threshold:
            self._trigger_alert(result)
            
        # 保存风险报告
        if self.config.get('save_reports', True):
            self._save_report(result)
        
        # 返回风险评分和风险因素
        return risk_score, risk_factors
    
    def _determine_risk_level(self, risk_score):
        """
        确定风险级别
        
        Args:
            risk_score: 风险分数
            
        Returns:
            str: 风险级别 (low, medium, high, critical)
        """
        if risk_score < self.low_threshold:
            return 'low'
        elif risk_score < self.medium_threshold:
            return 'medium'
        elif risk_score < self.high_threshold:
            return 'high'
        else:
            return 'critical'
    
    def _trigger_alert(self, result):
        """
        触发风险预警
        
        Args:
            result: 风险评估结果
        """
        alert_message = self._format_alert_message(result)
        
        # 根据配置的预警方法发送警报
        for method in self.alert_method:
            if method == 'console':
                logger.warning(alert_message)
            elif method == 'web':
                # 这里应该实现Web预警，在实际应用中可以通过WebSocket发送到前端
                pass
            elif method == 'email':
                self._send_email_alert(alert_message, result)
            elif method == 'sms':
                self._send_sms_alert(alert_message, result)
    
    def _format_alert_message(self, result):
        """
        格式化预警消息
        
        Args:
            result: 风险评估结果
            
        Returns:
            str: 格式化的预警消息
        """
        risk_level_zh = {
            'low': '低',
            'medium': '中',
            'high': '高',
            'critical': '严重'
        }
        
        message = f"⚠️ 风险预警 ⚠️\n"
        message += f"时间: {result['timestamp']}\n"
        message += f"风险级别: {risk_level_zh.get(result['risk_level'], result['risk_level'])}"
        message += f" (分数: {result['risk_score']:.2f})\n"
        message += f"文本: {result['text'][:100]}...\n"
        
        if result['risk_factors']:
            message += "风险因素:\n"
            for factor in result['risk_factors'][:5]:  # 仅显示前5个因素
                message += f"- {factor.get('description', str(factor))}\n"
            
            if len(result['risk_factors']) > 5:
                message += f"... 以及 {len(result['risk_factors']) - 5} 个其他因素\n"
        
        return message
    
    def _send_email_alert(self, alert_message, result):
        """
        发送邮件预警
        
        Args:
            alert_message: 预警消息
            result: 风险评估结果
        """
        # 检查邮件配置
        email_config = self.config.get('email', {})
        if not email_config.get('enabled', False):
            return
            
        smtp_server = email_config.get('smtp_server')
        smtp_port = email_config.get('smtp_port', 587)
        username = email_config.get('username')
        password = email_config.get('password')
        recipients = email_config.get('recipients', [])
        
        if not smtp_server or not username or not password or not recipients:
            logger.error("邮件配置不完整，无法发送邮件预警")
            return
            
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # 创建邮件
            message = MIMEMultipart()
            message['Subject'] = f"【课堂监控】风险预警 - {result['risk_level']}级别"
            message['From'] = username
            message['To'] = ', '.join(recipients)
            
            # 添加邮件正文
            body = MIMEText(alert_message, 'plain')
            message.attach(body)
            
            # 发送邮件
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(message)
                
            logger.info(f"邮件预警已发送至 {len(recipients)} 个收件人")
            
        except Exception as e:
            logger.exception(f"发送邮件预警失败: {str(e)}")
    
    def _send_sms_alert(self, alert_message, result):
        """
        发送短信预警
        
        Args:
            alert_message: 预警消息
            result: 风险评估结果
        """
        # 检查短信配置
        sms_config = self.config.get('sms', {})
        if not sms_config.get('enabled', False):
            return
            
        api_key = sms_config.get('api_key')
        recipients = sms_config.get('recipients', [])
        
        if not api_key or not recipients:
            logger.error("短信配置不完整，无法发送短信预警")
            return
            
        try:
            # 这里应该实现SMS API调用
            # 不同的服务提供商有不同的API
            logger.info(f"短信预警已发送至 {len(recipients)} 个收件人")
            
        except Exception as e:
            logger.exception(f"发送短信预警失败: {str(e)}")
    
    def _save_report(self, result):
        """
        保存风险报告
        
        Args:
            result: 风险评估结果
        """
        try:
            # 生成文件名，使用时间戳避免文件名冲突
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            risk_level = result['risk_level']
            filename = f"risk_report_{risk_level}_{timestamp}.json"
            filepath = os.path.join(self.report_path, filename)
            
            # 保存为JSON文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            logger.debug(f"风险报告已保存至: {filepath}")
            
        except Exception as e:
            logger.exception(f"保存风险报告失败: {str(e)}")
            
    def generate_summary_report(self, start_time=None, end_time=None):
        """
        生成摘要报告
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            dict: 摘要报告
        """
        # 这个方法可以根据需要实现，用于生成一定时间范围内的风险摘要
        # 可以包括风险趋势、高频关键词、常见风险因素等
        return {
            'start_time': start_time,
            'end_time': end_time,
            'summary': '尚未实现此功能'
        }
