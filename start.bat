@echo off
echo 正在启动课堂语义行为实时分析系统...
echo.

:: 确保存在必要的目录
if not exist "data\logs" mkdir "data\logs"
if not exist "data\models" mkdir "data\models"
if not exist "data\reports" mkdir "data\reports" 
if not exist "data\test_audio" mkdir "data\test_audio"

:: 检查虚拟环境
if exist ".venv\Scripts\activate.bat" (
    echo 激活虚拟环境...
    call .venv\Scripts\activate.bat
)

:: 显示启动信息
echo.
echo +---------------------------------------------+
echo ^|    课堂语义行为实时分析系统启动器    ^|
echo +---------------------------------------------+
echo ^| 管理员账号: admin                          ^|
echo ^| 管理员密码: admin123                       ^|
echo ^| Web界面: http://localhost:5000             ^|
echo +---------------------------------------------+
echo.

:: 启动系统
python run.py

:: 系统关闭后的提示
echo.
echo 系统已关闭。按任意键退出...
pause > nul
