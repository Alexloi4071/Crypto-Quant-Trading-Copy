"""
依賴檢查腳本
檢查所有必需的 Python 模組是否已安裝
"""

import sys
import importlib
from pathlib import Path

# 定義依賴檢查列表
DEPENDENCIES = {
    "核心框架": {
        "fastapi": "FastAPI Web 框架",
        "uvicorn": "ASGI 服務器",
        "pydantic": "數據驗證",
    },
    "數據處理": {
        "pandas": "數據分析",
        "numpy": "數值計算",
        "scipy": "科學計算",
    },
    "機器學習": {
        "sklearn": "scikit-learn 機器學習",
        "lightgbm": "LightGBM 模型",
        "xgboost": "XGBoost 模型",
        "optuna": "超參數優化",
    },
    "交易相關": {
        "binance": "Binance API (python-binance)",
        "ccxt": "加密貨幣交易所 API",
    },
    "工具庫": {
        "dotenv": "環境變數 (python-dotenv)",
        "yaml": "YAML 配置 (pyyaml)",
        "requests": "HTTP 請求",
        "aiohttp": "異步 HTTP",
    },
    "數據庫": {
        "sqlalchemy": "ORM 框架",
    },
    "安全與認證": {
        "jwt": "JWT Token (PyJWT)",
        "cryptography": "加密庫",
    },
}

# 可選依賴（不影響 UI 運行）
OPTIONAL_DEPENDENCIES = {
    "進階功能": {
        "telegram": "Telegram 通知 (python-telegram-bot)",
        "redis": "Redis 緩存",
        "celery": "任務隊列",
    },
    "可視化": {
        "plotly": "交互式圖表",
        "matplotlib": "靜態圖表",
    },
    "深度學習": {
        "tensorflow": "TensorFlow",
        "torch": "PyTorch",
    },
}

def check_module(module_name: str) -> tuple[bool, str]:
    """檢查單個模組是否已安裝"""
    try:
        importlib.import_module(module_name)
        return True, "✅"
    except ImportError:
        return False, "❌"
    except Exception as e:
        # 某些模組導入時可能有版本衝突或其他錯誤，但已安裝
        # 例如 tensorflow 與 numpy 版本衝突
        return True, "⚠️ "

def check_file_exists(file_path: str) -> tuple[bool, str]:
    """檢查文件是否存在"""
    if Path(file_path).exists():
        return True, "✅"
    else:
        return False, "❌"

def main():
    """主檢查函數"""
    print("=" * 70)
    print("🔍 Crypto Quant Trading System - 依賴檢查")
    print("=" * 70)
    print()
    
    # 檢查 Python 版本
    print("📌 Python 版本檢查:")
    python_version = sys.version_info
    print(f"   當前版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major >= 3 and python_version.minor >= 8:
        print("   ✅ Python 版本符合要求 (>= 3.8)")
    else:
        print("   ❌ Python 版本過低，需要 Python 3.8+")
    print()
    
    # 檢查必需依賴
    print("📦 必需依賴檢查:")
    print("-" * 70)
    
    missing_required = []
    for category, modules in DEPENDENCIES.items():
        print(f"\n{category}:")
        for module_name, description in modules.items():
            installed, status = check_module(module_name)
            print(f"  {status} {module_name:20} - {description}")
            if not installed:
                missing_required.append(module_name)
    
    print()
    print("-" * 70)
    
    # 檢查可選依賴
    print("\n📦 可選依賴檢查 (不影響基本功能):")
    print("-" * 70)
    
    missing_optional = []
    for category, modules in OPTIONAL_DEPENDENCIES.items():
        print(f"\n{category}:")
        for module_name, description in modules.items():
            installed, status = check_module(module_name)
            print(f"  {status} {module_name:20} - {description}")
            if not installed:
                missing_optional.append(module_name)
    
    print()
    print("-" * 70)
    
    # 檢查核心文件
    print("\n📁 核心文件檢查:")
    print("-" * 70)
    
    core_files = {
        "api/main.py": "API 主程序",
        "api/config.py": "API 配置",
        "api/dependencies.py": "依賴注入",
        "frontend/static/optuna.html": "Optuna UI",
        "frontend/js/optuna_client.js": "前端客戶端",
        "src/utils/logger.py": "日誌工具",
        "src/trading/trading_system.py": "交易系統",
        "config/settings.py": "系統配置",
    }
    
    missing_files = []
    for file_path, description in core_files.items():
        exists, status = check_file_exists(file_path)
        print(f"  {status} {file_path:40} - {description}")
        if not exists:
            missing_files.append(file_path)
    
    print()
    print("=" * 70)
    
    # 總結報告
    print("\n📊 檢查總結:")
    print("-" * 70)
    
    if not missing_required and not missing_files:
        print("✅ 所有必需依賴和核心文件都已就緒！")
        print("🚀 您可以運行以下命令啟動 UI：")
        print()
        print("   python start_ui.py")
        print("   或")
        print("   python api/main.py")
        print()
        return 0
    
    else:
        print("⚠️ 發現缺失項目：")
        print()
        
        if missing_required:
            print("❌ 缺失的必需依賴：")
            for module in missing_required:
                print(f"   - {module}")
            print()
            print("📝 安裝命令：")
            print(f"   pip install {' '.join(missing_required)}")
            print()
            print("   或者安裝所有依賴：")
            print("   pip install -r requirements.txt")
            print()
        
        if missing_files:
            print("❌ 缺失的核心文件：")
            for file in missing_files:
                print(f"   - {file}")
            print()
        
        if missing_optional:
            print("ℹ️  可選依賴缺失（不影響基本功能）：")
            for module in missing_optional:
                print(f"   - {module}")
            print()
        
        return 1
    
    print("=" * 70)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

