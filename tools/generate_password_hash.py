#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成密码哈希值工具
用于生成新的管理员密码哈希
"""

import sys
from werkzeug.security import generate_password_hash

def generate_hash(password):
    """生成密码哈希值"""
    return generate_password_hash(password)

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: python generate_password_hash.py <密码>")
        return 1
    
    password = sys.argv[1]
    password_hash = generate_hash(password)
    
    print("\n===== 密码哈希生成器 =====\n")
    print(f"密码: {password}")
    print(f"哈希值: {password_hash}")
    print("\n请将此哈希值复制到 config/auth.yaml 文件中的 admin_password_hash 字段")
    print("\n===========================\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
