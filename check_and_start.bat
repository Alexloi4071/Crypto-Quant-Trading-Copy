@echo off
chcp 65001 >nul
echo ============================================================
echo   Optuna 9層優化系統 - 自動檢查與啟動
echo ============================================================
echo.

REM 檢查 Python 是否安裝
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 錯誤: 未找到 Python
    echo.
    echo 請先安裝 Python 3.8+
    echo 下載地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python 已安裝
python --version
echo.

REM 檢查虛擬環境
if exist venv\Scripts\activate.bat (
    echo 🔧 啟動虛擬環境...
    call venv\Scripts\activate.bat
    echo.
)

REM 步驟 1: 檢查依賴
echo ============================================================
echo 步驟 1/3: 檢查依賴
echo ============================================================
echo.
python check_dependencies.py
if errorlevel 1 (
    echo.
    echo ⚠️  發現缺失的依賴
    echo.
    choice /C YN /M "是否自動安裝缺失的依賴"
    if errorlevel 2 goto :manual
    if errorlevel 1 goto :install
) else (
    echo.
    echo ✅ 所有依賴都已就緒！
    goto :start
)

:install
echo.
echo ============================================================
echo 步驟 2/3: 安裝依賴
echo ============================================================
echo.
python install_ui_dependencies.py
if errorlevel 1 (
    echo.
    echo ❌ 安裝失敗，請手動安裝
    echo.
    echo 運行: pip install -r requirements.txt
    pause
    exit /b 1
)
goto :start

:manual
echo.
echo 請手動運行以下命令安裝依賴：
echo   pip install -r requirements.txt
echo.
echo 或安裝最小依賴：
echo   python install_ui_dependencies.py
echo.
pause
exit /b 0

:start
echo.
echo ============================================================
echo 步驟 3/3: 啟動 UI
echo ============================================================
echo.
python start_ui.py
pause

