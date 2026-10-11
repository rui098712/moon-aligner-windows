@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo     月亮裁切与堆栈 Windows EXE 构建工具
echo ========================================
echo.

where py >nul 2>nul
if errorlevel 1 (
    echo 没有找到 Python。
    echo 请安装 64 位 Python 3.12，并勾选“Add python.exe to PATH”。
    echo 下载地址：https://www.python.org/downloads/windows/
    pause
    exit /b 1
)

if not exist ".build-venv\Scripts\python.exe" (
    echo [1/4] 创建构建环境……
    py -3.12 -m venv .build-venv
    if errorlevel 1 py -3 -m venv .build-venv
    if errorlevel 1 goto :failed
) else (
    echo [1/4] 使用已有构建环境。
)

echo [2/4] 安装或更新依赖……
".build-venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".build-venv\Scripts\python.exe" -m pip install -r requirements-build.txt
if errorlevel 1 goto :failed

echo [3/4] 生成独立 EXE，这一步可能需要几分钟……
".build-venv\Scripts\python.exe" -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onefile ^
    --windowed ^
    --name "月亮裁切与堆栈" ^
    main.py
if errorlevel 1 goto :failed

echo [4/4] 构建完成。
echo.
echo EXE 位置：%CD%\dist\月亮裁切与堆栈.exe
echo 可以把这个 EXE 复制到其他 64 位 Windows 10/11 电脑使用。
echo.
explorer "%CD%\dist"
pause
exit /b 0

:failed
echo.
echo 构建失败。请保留这个窗口中的错误信息以便排查。
pause
exit /b 1

