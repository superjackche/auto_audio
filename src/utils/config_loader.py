#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置加载器模块
"""

import os
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigLoader:
    """配置加载器类"""
    
    def __init__(self, config_dir=None):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录，如果为None，则使用默认目录
        """
        # 如果未指定配置目录，使用默认的配置目录
        if config_dir is None:
            self.config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
        else:
            self.config_dir = config_dir
            
        # 确保配置目录存在
        os.makedirs(self.config_dir, exist_ok=True)
        
        logger.info(f"配置加载器初始化完成，使用配置目录: {self.config_dir}")
    
    def load_config(self, config_name='config.yaml'):
        """
        加载配置文件
        
        Args:
            config_name: 配置文件名，默认为config.yaml
            
        Returns:
            dict: 配置字典
        """
        config_path = os.path.join(self.config_dir, config_name)
        return self._load_yaml(config_path)
    
    def _load_yaml(self, config_path):
        """
        加载YAML配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            dict: 配置字典
        """
        try:
            if not os.path.exists(config_path):
                logger.warning(f"配置文件不存在: {config_path}")
                return {}
                
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            logger.info(f"成功加载配置文件: {config_path}")
            return config
            
        except Exception as e:
            logger.exception(f"加载配置文件失败: {str(e)}")
            return {}

# 兼容旧版API
def load_config(config_path):
    """
    加载YAML配置文件（兼容旧版API）
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    try:
        if not os.path.exists(config_path):
            logger.error(f"配置文件不存在: {config_path}")
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        logger.info(f"成功加载配置文件: {config_path}")
        return config
        
    except Exception as e:
        logger.exception(f"加载配置文件失败: {str(e)}")
        raise
