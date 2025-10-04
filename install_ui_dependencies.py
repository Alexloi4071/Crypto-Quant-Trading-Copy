"""
安裝 UI 運行所需的最小依賴
僅安裝運行 Web UI 必需的包，不包含完整的機器學習包
"""

import subprocess
import sys

# UI 運行的最小依賴列表
MINIMAL_DEPENDENCIES = [
    # Web 框架
    "fastapi>=0.103.0",
    "uvicorn[standard]>=0.23.0",
    "pydantic>=2.0.0",
    
    # 基礎數據處理
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    
    # 工具庫
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0",
    "requests>=2.31.0",
    "aiohttp>=3.8.5",
    
    # 安全認證
    "PyJWT>=2.8.0",
    "cryptography>=41.0.0",
    
    # 數據庫（輕量級）
    "sqlalchemy>=2.0.0",
    
    # 交易所 API
    "python-binance>=1.0.16",
    "ccxt>=4.0.0",
]

def install_packages(packages: list[str]) -> bool:
    """安裝包列表"""
    print("=" * 70)
    print("📦 安裝 UI 最小依賴包")
    print("=" * 70)
    print()
    print("即將安裝以下包：")
    for pkg in packages:
        print(f"  - {pkg}")
    print()
    
    try:
        # 升級 pip
        print("⬆️  升級 pip...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ])
        print("✅ pip 升級完成")
        print()
        
        # 安裝依賴
        print("📥 開始安裝依賴包...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + packages)
        
        print()
        print("=" * 70)
        print("✅ 所有依賴安裝完成！")
        print("=" * 70)
        print()
        print("🚀 現在可以運行以下命令啟動 UI：")
        print()
        print("   python start_ui.py")
        print()
        return True
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 70)
        print("❌ 安裝失敗！")
        print("=" * 70)
        print()
        print(f"錯誤信息: {e}")
        print()
        print("可能的解決方案：")
        print("1. 確保有網絡連接")
        print("2. 使用國內鏡像：")
        print("   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>")
        print("3. 檢查 Python 版本是否 >= 3.8")
        print("4. 嘗試使用管理員/root 權限運行")
        print()
        return False

def main():
    """主函數"""
    print()
    print("🔍 檢查 Python 版本...")
    python_version = sys.version_info
    print(f"   Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 版本過低，需要 Python 3.8+")
        sys.exit(1)
    
    print("✅ Python 版本符合要求")
    print()
    
    # 詢問用戶確認
    response = input("是否繼續安裝？(y/n): ").strip().lower()
    if response != 'y':
        print("取消安裝")
        sys.exit(0)
    
    print()
    
    # 安裝依賴
    success = install_packages(MINIMAL_DEPENDENCIES)
    
    if success:
        # 運行依賴檢查
        print("🔍 驗證安裝...")
        try:
            subprocess.check_call([sys.executable, "check_dependencies.py"])
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("ℹ️  跳過驗證檢查")
        
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()

