# 课堂语义行为实时分析系统 - PowerShell启动脚本
# 此脚本用于在Windows系统上以PowerShell环境启动系统
# 推荐使用方式: 右键点击此文件，选择"使用PowerShell运行"

# 设置控制台标题
$host.UI.RawUI.WindowTitle = "课堂语义行为实时分析系统"

# 显示欢迎标志
Write-Host ""
Write-Host "  课堂语义行为实时分析系统启动器  " -ForegroundColor White -BackgroundColor DarkBlue
Write-Host "  ============================  " -ForegroundColor Cyan
Write-Host ""
Write-Host "正在启动系统..." -ForegroundColor Cyan

# 确保存在必要的目录
$dirs = @(
    ".\data\logs",
    ".\data\models",
    ".\data\reports",
    ".\data\test_audio"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "创建目录: $dir" -ForegroundColor Yellow
    }
}

# 检查虚拟环境
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "激活虚拟环境..." -ForegroundColor Green
    try {
        & .\.venv\Scripts\Activate.ps1
    }
    catch {
        Write-Host "无法激活虚拟环境，将使用系统Python环境" -ForegroundColor Yellow
    }
}

# 检查Python是否安装
try {
    $pythonVersion = (python --version 2>&1)
    Write-Host "检测到Python: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "未检测到Python，请确保已安装Python 3.11或更高版本" -ForegroundColor Red
    Write-Host "按任意键退出..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# 显示启动信息
Write-Host ""
Write-Host "+---------------------------------------------+" -ForegroundColor Cyan
Write-Host "|    课堂语义行为实时分析系统              |" -ForegroundColor Cyan
Write-Host "+---------------------------------------------+" -ForegroundColor Cyan
Write-Host "| 管理员账号: admin                          |" -ForegroundColor White
Write-Host "| 管理员密码: admin123                       |" -ForegroundColor White
Write-Host "| Web界面: http://localhost:5000             |" -ForegroundColor White
Write-Host "+---------------------------------------------+" -ForegroundColor Cyan
Write-Host ""
Write-Host "系统启动中，请稍候..." -ForegroundColor Yellow

# 启动系统
try {
    Write-Host "正在启动Web界面和分析引擎..." -ForegroundColor Green
    python run.py
}
catch {
    Write-Host "启动系统时出错: $_" -ForegroundColor Red
}

# 系统关闭后的提示
Write-Host ""
Write-Host "系统已关闭。" -ForegroundColor Cyan
Write-Host "按任意键退出..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
