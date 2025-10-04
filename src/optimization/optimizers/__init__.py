#!/usr/bin/env python3
"""
模組化Optuna優化系統

提供標籤優化、特徵選擇、模型超參數優化的完整解決方案
支持從方案B到方案C的漸進式升級路徑
"""

__version__ = "2.0.0"
__author__ = "AI優化團隊"
__description__ = "模組化Optuna優化系統 - 方案B到C漸進式升級"

# 導入主要類
try:
    from .config import OptimizationConfig, get_config
    from .label_optimizer import LabelOptimizer
    from .feature_selector import FeatureSelector
    from .model_optimizer import ModelOptimizer
    from .main_optimizer import ModularOptunaOptimizer
except ImportError as e:
    # 處理相對導入問題，提供友好的錯誤信息
    import warnings
    warnings.warn(f"部分模組導入失敗: {e}. 請確保所有依賴已正確安裝。", ImportWarning)
    
    # 嘗試絕對導入作為後備方案
    try:
        from src.optimization.config import OptimizationConfig, get_config
        from src.optimization.label_optimizer import LabelOptimizer
        from src.optimization.feature_selector import FeatureSelector
        from src.optimization.model_optimizer import ModelOptimizer
        from src.optimization.main_optimizer import ModularOptunaOptimizer
    except ImportError:
        # 如果仍然失敗，則設置為 None
        OptimizationConfig = None
        LabelOptimizer = None
        FeatureSelector = None
        ModelOptimizer = None
        ModularOptunaOptimizer = None
        print("❌ 警告: 無法導入優化模組，請檢查Python路徑和依賴")

# 定義公共API
__all__ = [
    # 主要類
    "ModularOptunaOptimizer",    # 主控制器
    "LabelOptimizer",            # 標籤優化器
    "FeatureSelector",           # 特徵選擇器
    "ModelOptimizer",            # 模型優化器
    
    # 配置管理
    "OptimizationConfig",        # 配置類
    "get_config",               # 配置獲取函數
    
    # 版本信息
    "__version__",
    "__author__",
    "__description__"
]

# 便捷的快速開始函數
def quick_start(symbol: str, timeframe: str = "1h"):
    """快速開始優化
    
    Args:
        symbol: 交易對符號 (如 "BTCUSDT")
        timeframe: 時間框架 (如 "1h", "4h", "15m", "1D")
        
    Returns:
        ModularOptunaOptimizer實例
        
    Example:
        >>> optimizer = quick_start("BTCUSDT", "1h")
        >>> results = optimizer.run_complete_optimization()
    """
    if ModularOptunaOptimizer is None:
        raise ImportError("優化模組導入失敗，無法創建優化器")
        
    return ModularOptunaOptimizer(symbol, timeframe)

# 添加快速開始函數到公共API
__all__.append("quick_start")

# 包級別的配置常量
DEFAULT_CONFIG = {
    "supported_symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT"],
    "supported_timeframes": ["15m", "1h", "4h", "1D"],
    "min_python_version": "3.8",
    "required_packages": [
        "optuna>=3.0.0",
        "lightgbm>=3.3.0", 
        "scikit-learn>=1.2.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0"
    ]
}

__all__.append("DEFAULT_CONFIG")

# 版本檢查函數
def check_dependencies():
    """檢查必需依賴是否已安裝
    
    Returns:
        dict: 依賴檢查結果
    """
    import sys
    
    results = {
        "python_version": sys.version,
        "python_version_ok": sys.version_info >= (3, 8),
        "packages": {}
    }
    
    for package in DEFAULT_CONFIG["required_packages"]:
        package_name = package.split(">=")[0]
        try:
            __import__(package_name)
            results["packages"][package_name] = "✅ 已安裝"
        except ImportError:
            results["packages"][package_name] = f"❌ 未安裝 - 需要: {package}"
    
    return results

__all__.append("check_dependencies")

# 如果直接運行此模組，顯示包信息
if __name__ == "__main__":
    print(f"📦 {__description__}")
    print(f"🔢 版本: {__version__}")
    print(f"👥 作者: {__author__}")
    print("\n📋 支持的功能:")
    print("  ├─ 標籤多目標優化 (NSGA-II)")
    print("  ├─ 智能特徵選擇 (4種方法)")
    print("  ├─ 模型超參數優化 (TPE + Pruning)")
    print("  └─ 端到端自動化流程")
    
    print("\n🔍 依賴檢查:")
    deps = check_dependencies()
    print(f"  Python版本: {deps['python_version_ok'] and '✅' or '❌'} {deps['python_version']}")
    for pkg, status in deps["packages"].items():
        print(f"  {pkg}: {status}")
    
    print(f"\n🚀 快速開始:")
    print("  from src.optimization import quick_start")
    print("  optimizer = quick_start('BTCUSDT', '1h')")
    print("  results = optimizer.run_complete_optimization()")
