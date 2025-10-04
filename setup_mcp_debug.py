#!/usr/bin/env python3
"""
MCP Python Debug 系統設置腳本
加密貨幣量化交易系統專用

此腳本將：
1. 檢查並安裝必要的依賴包
2. 設置環境變量
3. 配置 Cursor IDE 的 MCP 設置
4. 測試 MCP 服務器連接
"""

import sys
import subprocess
import json
import os
from pathlib import Path
import shutil
import platform

class MCPDebugSetup:
    def __init__(self):
        self.project_root = Path.cwd()
        self.cursor_config_dir = self._get_cursor_config_dir()

    def _get_cursor_config_dir(self) -> Path:
        """獲取 Cursor IDE 配置目錄"""
        system = platform.system()

        if system == "Windows":
            return Path.home() / "AppData" / "Roaming" / "Cursor" / "User"
        elif system == "Darwin":  # macOS
            return Path.home() / "Library" / "Application Support" / "Cursor" / "User"
        else:  # Linux
            return Path.home() / ".config" / "Cursor" / "User"

    def check_python_version(self) -> bool:
        """檢查 Python 版本"""
        print("🐍 檢查 Python 版本...")
        version = sys.version_info

        if version.major == 3 and version.minor >= 8:
            print(f"✅ Python {version.major}.{version.minor}.{version.micro} - 版本符合要求")
            return True
        else:
            print(f"❌ Python 版本過舊：{version.major}.{version.minor}.{version.micro}")
            print("需要 Python 3.8+ (推薦 3.10+)")
            return False

    def install_dependencies(self) -> bool:
        """安裝 MCP Debug 系統依賴"""
        print("📦 安裝 MCP Debug 依賴包...")

        required_packages = [
            "anthropic>=0.7.0",
            "openai>=1.0.0",
            "google-generativeai>=0.3.0",
            "aiohttp>=3.8.0",
            "pyyaml>=6.0",
            "mcp>=0.4.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0"
        ]

        success = True
        for package in required_packages:
            try:
                print(f"Installing {package}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"✅ {package} - 安裝成功")
            except subprocess.CalledProcessError as e:
                print(f"❌ {package} - 安裝失敗: {e}")
                print(f"Error output: {e.stderr}")
                success = False

        return success

    def setup_environment_variables(self) -> bool:
        """設置環境變量"""
        print("🔐 設置環境變量...")

        env_file = self.project_root / ".env"

        # 檢查是否已存在 .env 文件
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                env_content = f.read()
        else:
            env_content = ""

        # 需要添加的 MCP 相關環境變量
        mcp_vars = {
            "CLAUDE_API_KEY": "your_claude_api_key_here",
            "OPENAI_API_KEY": "your_openai_api_key_here",
            "GEMINI_API_KEY": "your_gemini_api_key_here",
            "MCP_DEBUG_ENABLED": "true",
            "MCP_LOG_LEVEL": "INFO"
        }

        updated = False
        for var, default_value in mcp_vars.items():
            if var not in env_content:
                env_content += f"\n# MCP Debug Configuration\n{var}={default_value}\n"
                updated = True
                print(f"➕ 已添加環境變量: {var}")

        if updated:
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(env_content)
            print(f"✅ 環境變量已更新到 {env_file}")
        else:
            print("✅ 環境變量已存在，無需更新")

        # 提醒用戶設置實際的 API Keys
        print("\n⚠️  重要提醒：")
        print("請在 .env 文件中設置實際的 API Keys：")
        for var in ["CLAUDE_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]:
            if f"{var}=your_" in env_content:
                print(f"   - {var} (目前為示例值)")

        return True

    def configure_cursor_mcp(self) -> bool:
        """配置 Cursor IDE 的 MCP 設置"""
        print("⚙️  配置 Cursor IDE MCP 設置...")

        # Cursor 配置文件路径
        settings_file = self.cursor_config_dir / "settings.json"

        # MCP 服務器配置
        mcp_config = {
            "mcpServers": {
                "python-debug-aggregator": {
                    "command": "python",
                    "args": [str(self.project_root / "mcp_python_debug_server.py")],
                    "env": {
                        "PYTHONPATH": str(self.project_root),
                        "PROJECT_ROOT": str(self.project_root)
                    }
                }
            }
        }

        try:
            # 讀取現有設置
            if settings_file.exists():
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
            else:
                settings = {}

            # 合併 MCP 配置
            if "mcpServers" not in settings:
                settings["mcpServers"] = {}

            settings["mcpServers"].update(mcp_config["mcpServers"])

            # 創建配置目錄（如果不存在）
            settings_file.parent.mkdir(parents=True, exist_ok=True)

            # 寫回配置文件
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)

            print(f"✅ Cursor MCP 配置已更新: {settings_file}")
            return True

        except Exception as e:
            print(f"❌ Cursor 配置失敗: {e}")
            print("請手動配置 Cursor IDE 的 MCP 設置")
            self._print_manual_config()
            return False

    def _print_manual_config(self):
        """打印手動配置說明"""
        print("\n📋 手動配置 Cursor IDE:")
        print("1. 打開 Cursor IDE")
        print("2. 按 Ctrl+, (Cmd+, on Mac) 打開設置")
        print("3. 搜索 'MCP' 或找到 'Model Context Protocol'")
        print("4. 添加新的 MCP 服務器:")
        print(f"   - Name: python-debug-aggregator")
        print(f"   - Command: python")
        print(f"   - Args: {self.project_root / 'mcp_python_debug_server.py'}")
        print(f"   - Working Directory: {self.project_root}")

    def test_mcp_server(self) -> bool:
        """測試 MCP 服務器連接"""
        print("🧪 測試 MCP 服務器...")

        # 檢查服務器文件是否存在
        server_file = self.project_root / "mcp_python_debug_server.py"
        if not server_file.exists():
            print(f"❌ MCP 服務器文件不存在: {server_file}")
            return False

        # 嘗試導入和基本測試
        try:
            print("測試基本導入...")
            subprocess.run([
                sys.executable, "-c",
                "import asyncio, json, sys; print('✅ 基本導入測試通過')"
            ], check=True, capture_output=True)

            print("✅ MCP 服務器基本測試通過")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ MCP 服務器測試失敗: {e}")
            print(f"Error: {e.stderr.decode()}")
            return False

    def create_startup_script(self):
        """創建快速啟動腳本"""
        print("📝 創建啟動腳本...")

        # Windows 腳本
        windows_script = self.project_root / "start_mcp_debug.bat"
        windows_content = f"""@echo off
echo 🚀 启动 MCP Python Debug 系统...

REM 激活虚拟环境 (如果存在)
if exist "venv\\Scripts\\activate.bat" (
    call venv\\Scripts\\activate.bat
    echo ✅ 虚拟环境已激活
) else if exist "crypto_trading_env\\Scripts\\activate.bat" (
    call crypto_trading_env\\Scripts\\activate.bat
    echo ✅ 虚拟环境已激活
)

REM 检查环境
echo 🔍 检查环境状态...
python check_environment.py

REM 启动 MCP 服务器
echo 🤖 启动 MCP Debug 服务器...
python mcp_python_debug_server.py

pause
"""

        with open(windows_script, 'w', encoding='utf-8') as f:
            f.write(windows_content)

        # Linux/Mac 腳本
        unix_script = self.project_root / "start_mcp_debug.sh"
        unix_content = f"""  # !/bin/bash
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
"""

        with open(unix_script, 'w', encoding='utf-8') as f:
            f.write(unix_content)

        # 設置執行權限 (Unix/Mac)
        if platform.system() != "Windows":
            os.chmod(unix_script, 0o755)

        print("✅ 啟動腳本已創建")
        print(f"   - Windows: {windows_script.name}")
        print(f"   - Unix/Mac: {unix_script.name}")

    def print_usage_guide(self):
        """打印使用指南"""
        print("\n" + "="*50)
        print("🎉 MCP Python Debug 系統設置完成!")
        print("="*50)

        print("\n📋 使用指南:")
        print("1. 重啟 Cursor IDE 以加載 MCP 配置")
        print("2. 在 Cursor 中使用以下命令:")
        print("   @mcp scan_environment")
        print("   @mcp analyze_error <錯誤詳情>")
        print("   @mcp batch_scan")
        print("\n3. 或直接使用:")
        print("   @mcp 請掃描環境並分析所有錯誤")

        print("\n🔧 配置文件:")
        print(f"   - MCP 服務器: mcp_python_debug_server.py")
        print(f"   - 配置文件: mcp_servers_config.yaml")
        print(f"   - 環境變量: .env")

        print("\n⚠️  注意事項:")
        print("   - 請在 .env 文件中設置實際的 AI API Keys")
        print("   - 首次使用前請運行: python check_environment.py")
        print("   - 如遇問題請查看日誌: mcp_debug.log")

        print("\n🚀 快速開始:")
        if platform.system() == "Windows":
            print("   運行: start_mcp_debug.bat")
        else:
            print("   運行: ./start_mcp_debug.sh")

def main():
    """主函數"""
    print("🔧 MCP Python Debug 系統設置")
    print("=" * 40)

    setup = MCPDebugSetup()

    # 檢查步驟
    steps = [
        ("Python 版本檢查", setup.check_python_version),
        ("安裝依賴包", setup.install_dependencies),
        ("環境變量設置", setup.setup_environment_variables),
        ("Cursor MCP 配置", setup.configure_cursor_mcp),
        ("MCP 服務器測試", setup.test_mcp_server),
    ]

    success_count = 0
    for step_name, step_func in steps:
        print(f"\n🔄 執行: {step_name}")
        try:
            if step_func():
                success_count += 1
                print(f"✅ {step_name} - 完成")
            else:
                print(f"⚠️  {step_name} - 部分完成或需要手動處理")
        except Exception as e:
            print(f"❌ {step_name} - 失敗: {e}")

    # 創建輔助文件
    setup.create_startup_script()

    # 總結
    print(f"\n📊 設置結果: {success_count}/{len(steps)} 步驟成功")

    if success_count >= len(steps) - 1:  # 允許一個步驟失敗
        setup.print_usage_guide()
    else:
        print("❌ 設置未完成，請檢查錯誤並重新運行")
        return False

    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⛔ 用戶中斷設置")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n⛔ 設置過程中出現錯誤: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)
