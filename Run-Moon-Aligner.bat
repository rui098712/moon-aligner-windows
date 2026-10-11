@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo 首次运行，正在创建独立的 Python 环境……
    where py >nul 2>nul
    if errorlevel 1 (
        echo.
        echo 没有找到 Python。请安装 64 位 Python 3.12，并勾选“Add python.exe to PATH”。
        echo 下载地址：https://www.python.org/downloads/windows/
        pause
        exit /b 1
    )
    py -3.12 -m venv .venv
    if errorlevel 1 py -3 -m venv .venv
    if errorlevel 1 goto :failed
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :failed
)

start "" ".venv\Scripts\pythonw.exe" main.py
exit /b 0

:failed
echo.
echo 安装失败，请检查网络连接和 Python 安装。
pause
exit /b 1

