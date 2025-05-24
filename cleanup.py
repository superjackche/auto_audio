#!/usr/bin/env python3
"""
项目清理脚本 - 删除不必要的文件，保留核心功能
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """清理项目文件"""
    project_root = Path(__file__).parent
    
    # 要删除的文件
    files_to_remove = [
        'fix_script.py',
        'fix_speech_to_text.py', 
        'initialize.py',
        'install_dependencies.py',
        'install_whisper.py',
        'main.py',
        'python313_compatibility_check.py',
        'PYTHON313_COMPATIBILITY.md',
        'run_optimized.py',
        'simple_test.py',
        'start_simple.py',
        'start_windows.ps1',
        'system_status_check.py',
        'test_core.py',
        'test_imports.py',
        'final_test.py',
        'start.bat',
        'start.ps1', 
        'start.sh',
        'run.py'
    ]
    
    # 要删除的目录
    dirs_to_remove = [
        'auto_audio.egg-info'
    ]
    
    print("开始清理项目...")
    print("="*50)
    
    # 删除文件
    for file_name in files_to_remove:
        file_path = project_root / file_name
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"✓ 已删除文件: {file_name}")
            except Exception as e:
                print(f"✗ 删除文件失败 {file_name}: {e}")
    
    # 删除目录
    for dir_name in dirs_to_remove:
        dir_path = project_root / dir_name
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"✓ 已删除目录: {dir_name}")
            except Exception as e:
                print(f"✗ 删除目录失败 {dir_name}: {e}")
    
    # 清理src目录中的备份文件
    src_dir = project_root / 'src'
    if src_dir.exists():
        print("\\n清理源代码备份文件...")
        for backup_file in src_dir.rglob('*.bak'):
            try:
                backup_file.unlink()
                print(f"✓ 已删除备份: {backup_file.relative_to(project_root)}")
            except Exception as e:
                print(f"✗ 删除备份失败: {e}")
        
        for new_file in src_dir.rglob('*.new'):
            try:
                new_file.unlink()
                print(f"✓ 已删除临时文件: {new_file.relative_to(project_root)}")
            except Exception as e:
                print(f"✗ 删除临时文件失败: {e}")
    
    print("\\n="*50)
    print("清理完成！")
    print("\\n保留的核心文件:")
    print("- launch.py (主启动脚本)")
    print("- requirements.txt (依赖列表)")
    print("- README.md (项目说明)")
    print("- setup.py (安装脚本)")
    print("- src/ (源代码目录)")
    print("- config/ (配置文件)")
    print("- data/ (数据目录)")
    print("- docs/ (文档)")
    print("- tests/ (测试)")
    print("- tools/ (工具)")

if __name__ == "__main__":
    cleanup_project()
