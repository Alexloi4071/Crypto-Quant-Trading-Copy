  # !/bin/bash
echo "🚀 啟動 MCP Python Debug 系統..."

# 激活虛擬環境 (如果存在)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ 虛擬環境已激活"
elif [ -f "crypto_trading_env/bin/activate" ]; then
    source crypto_trading_env/bin/activate
    echo "✅ 虛擬環境已激活"
fi

# 檢查環境
echo "🔍 檢查環境狀態..."
python check_environment.py

# 啟動 MCP 服務器
echo "🤖 啟動 MCP Debug 服務器..."
python mcp_python_debug_server.py
