# 开发环境初始化脚本
Write-Host "正在设置开发环境..." -ForegroundColor Green

# 创建虚拟环境
python -m venv .venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "创建虚拟环境失败" -ForegroundColor Red
    exit 1
}

# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 升级pip
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "升级pip失败" -ForegroundColor Red
    exit 1
}

# 安装开发依赖
pip install -e .
if ($LASTEXITCODE -ne 0) {
    Write-Host "安装开发依赖失败" -ForegroundColor Red
    exit 1
}

# 下载一些测试音频
$testAudioDir = ".\data\test_audio"
if (-not (Test-Path $testAudioDir)) {
    New-Item -ItemType Directory -Path $testAudioDir | Out-Null
}

# 创建管理员配置
$authConfig = @"
admin:
  username: admin
  # 默认密码：admin123
  password_hash: pbkdf2:sha256:260000$GAzrggwEMrEfhFkZ$7d5808fcf6e46f2bb54c8fc486db3235f96d352404934d4d3c428bdc43ef5f1d
"@

$authConfigPath = ".\config\auth.yaml"
if (-not (Test-Path $authConfigPath)) {
    New-Item -ItemType File -Path $authConfigPath -Force
    $authConfig | Out-File -FilePath $authConfigPath -Encoding UTF8
}

Write-Host "开发环境设置完成！" -ForegroundColor Green
Write-Host "管理员账号：admin" -ForegroundColor Yellow
Write-Host "管理员密码：admin123" -ForegroundColor Yellow
Write-Host "请运行 'python src/web/app.py' 启动系统" -ForegroundColor Yellow
