#!/bin/bash
# 课堂语义行为实时分析系统 - Linux/MacOS启动脚本

# 显示启动标志
echo ""
echo "  课堂语义行为实时分析系统启动器  "
echo "  ============================  "
echo ""

# 创建必要的目录
mkdir -p data/logs data/models data/reports data/test_audio
echo "检查必要的目录..."

# 检查虚拟环境
if [ -f ".venv/bin/activate" ]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
fi

# 显示启动信息
echo ""
echo "+---------------------------------------------+"
echo "|    课堂语义行为实时分析系统              |"
echo "+---------------------------------------------+"
echo "| 管理员账号: admin                          |"
echo "| 管理员密码: admin123                       |"
echo "| Web界面: http://localhost:5000             |"
echo "+---------------------------------------------+"
echo ""

# 启动系统
echo "正在启动系统..."
python run.py

# 系统关闭时的提示
echo ""
echo "系统已关闭。"
echo ""
