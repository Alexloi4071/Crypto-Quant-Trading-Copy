@echo off
chcp 65001 >nul
echo ============================================================
echo   啟動 Optuna 9層優化控制台
echo ============================================================
echo.
echo 📡 服務信息：
echo    ┌─ 主頁面: http://localhost:8000
echo    ├─ Optuna控制台: http://localhost:8000/static/optuna.html
echo    ├─ API文檔: http://localhost:8000/docs
echo    ├─ 健康檢查: http://localhost:8000/api/v1/health
echo    └─ WebSocket: ws://localhost:8000/ws
echo.
echo 💡 提示：按 Ctrl+C 停止服務
echo ============================================================
echo.

REM 檢查 Python 是否安裝
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 錯誤: 未找到 Python，請先安裝 Python 3.8+
    pause
    exit /b 1
)

REM 檢查虛擬環境
if exist venv\Scripts\activate.bat (
    echo 🔧 啟動虛擬環境...
    call venv\Scripts\activate.bat
)

REM 啟動服務
echo 🚀 正在啟動服務器...
echo.
python api\main.py

if errorlevel 1 (
    echo.
    echo ❌ 啟動失敗！
    echo.
    echo 可能的原因：
    echo   1. 端口 8000 已被佔用
    echo   2. 缺少必要的依賴包 (運行: pip install -r requirements.txt)
    echo   3. 配置文件錯誤
    echo.
    pause
    exit /b 1
)

pause

