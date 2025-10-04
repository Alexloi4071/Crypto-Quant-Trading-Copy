"""
啟動 Optuna 9層優化 UI 系統
簡易啟動腳本 - 啟動 FastAPI 後端 + 前端 UI
"""

import sys
import os
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """檢查必要的依賴"""
    required_packages = ['fastapi', 'uvicorn', 'pydantic']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ 缺少必要的依賴包: {', '.join(missing)}")
        print(f"   請運行: pip install {' '.join(missing)}")
        return False
    
    return True

def main():
    """啟動 UI 系統"""
    print("=" * 60)
    print("🚀 啟動 Optuna 9層優化控制台")
    print("=" * 60)
    print()
    
    # 檢查依賴
    if not check_dependencies():
        sys.exit(1)
    
    print("📡 服務信息：")
    print("   ┌─ 主頁面: http://localhost:8000")
    print("   ├─ Optuna控制台: http://localhost:8000/static/optuna.html")
    print("   ├─ API文檔: http://localhost:8000/docs")
    print("   ├─ 健康檢查: http://localhost:8000/api/v1/health")
    print("   └─ WebSocket: ws://localhost:8000/ws")
    print()
    print("💡 提示：按 Ctrl+C 停止服務")
    print("=" * 60)
    print()
    
    # 啟動 FastAPI 服務
    from api.main import main as run_api
    
    try:
        run_api()
    except KeyboardInterrupt:
        print("\n\n🛑 服務已停止")
    except ModuleNotFoundError as e:
        print(f"\n❌ 模組導入錯誤: {e}")
        print("\n可能的原因：")
        print("1. 缺少必要的 Python 包")
        print("2. 項目結構不完整")
        print("\n建議操作：")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 啟動失敗: {e}")
        print("\n可能的原因：")
        print("1. 端口 8000 已被佔用")
        print("2. 缺少必要的配置文件")
        print("3. 數據庫連接失敗")
        print("\n請檢查錯誤信息並修復後重試。")
        sys.exit(1)

if __name__ == "__main__":
    main()

