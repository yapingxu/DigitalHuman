@echo off
title EchoMimic Web界面启动器
echo 🎭 EchoMimic 数字人生成器
echo =============================

REM 检查是否在正确的目录
if not exist "infer.py" (
    echo ❌ 错误: 请确保在EchoMimic项目根目录下运行此脚本
    echo 当前目录: %CD%
    pause
    exit /b 1
)

REM 激活conda环境
call conda activate echomimic
if errorlevel 1 (
    echo ❌ 错误: 无法激活echomimic环境
    echo 请先确保已创建echomimic conda环境
    pause
    exit /b 1
)

echo ✅ echomimic环境已激活

REM 检查是否已安装Web依赖
python -c "import gradio" 2>nul
if errorlevel 1 (
    echo 📦 检测到缺少Web界面依赖，开始安装...
    python setup_web.py
    if errorlevel 1 (
        echo ❌ 依赖安装失败
        pause
        exit /b 1
    )
)

REM 启动Web界面
echo 🌐 启动Web界面...
python start_web.py

pause