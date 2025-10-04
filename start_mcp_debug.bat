@echo off
echo 🚀 启动 MCP Python Debug 系统...

REM 激活虚拟环境 (如果存在)
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo ✅ 虚拟环境已激活
) else if exist "crypto_trading_env\Scripts\activate.bat" (
    call crypto_trading_env\Scripts\activate.bat
    echo ✅ 虚拟环境已激活
)

REM 检查环境
echo 🔍 检查环境状态...
python check_environment.py

REM 启动 MCP 服务器
echo 🤖 启动 MCP Debug 服务器...
python mcp_python_debug_server.py

pause
