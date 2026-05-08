@echo off
chcp 65001 >nul
title ChannelHub - 一键启动
echo ============================================
echo   ChannelHub · 信道数据工场
echo ============================================
echo.

:: 检测 Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 显示 Python 版本
python --version

:: 设置项目根目录
set MSG_REPO_ROOT=%~dp0
set MSG_REPO_ROOT=%MSG_REPO_ROOT:~0,-1%
echo 项目目录: %MSG_REPO_ROOT%

:: 如果没有虚拟环境就创建
if not exist "%MSG_REPO_ROOT%\.venv\Scripts\python.exe" (
    echo.
    echo [1/3] 创建虚拟环境...
    python -m venv "%MSG_REPO_ROOT%\.venv"
    if %errorlevel% neq 0 (
        echo [错误] 创建虚拟环境失败
        pause
        exit /b 1
    )
    echo.
    echo [2/3] 安装依赖（首次需要几分钟）...
    "%MSG_REPO_ROOT%\.venv\Scripts\pip" install -e "%MSG_REPO_ROOT%[platform]" --quiet
    if %errorlevel% neq 0 (
        echo [错误] 安装依赖失败，请检查网络连接
        pause
        exit /b 1
    )
    echo 依赖安装完成
) else (
    echo 虚拟环境已存在，跳过安装
)

:: 初始化数据库
echo.
echo [3/3] 初始化数据库...
"%MSG_REPO_ROOT%\.venv\Scripts\python" -c "import sys; sys.path.insert(0,'src'); from platform.backend.db import init_db; init_db()" 2>nul
echo 数据库就绪

:: 启动服务
echo.
echo ============================================
echo   启动中... 浏览器访问 http://localhost:8000
echo   按 Ctrl+C 停止
echo ============================================
echo.

"%MSG_REPO_ROOT%\.venv\Scripts\python" -m uvicorn platform.backend.main:app --host 0.0.0.0 --port 8000

pause
