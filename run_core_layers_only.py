from pathlib import Path
import os
import sys

from optuna_system.coordinator import OptunaCoordinator
from config.timeframe_scaler import MultiTimeframeCoordinator


def ensure_utf8_console() -> None:
    """強制 Windows 主控台與 Python I/O 使用 UTF-8，避免中文/符號亂碼"""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")

    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        except Exception:
            pass

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


ensure_utf8_console()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data"

# Force reload and print optimizer module info to ensure latest version
try:
    import importlib
    import optuna_system.optimizers.optuna_feature as _ofeat
    print("FeatureOptimizer file =>", _ofeat.__file__)
    print("FeatureOptimizer PATCH_ID =>", getattr(_ofeat, "PATCH_ID", None))
    importlib.reload(_ofeat)
except Exception as _e:
    print("[WARN] Could not pre-load FeatureOptimizer:", _e)

multi_scaler = MultiTimeframeCoordinator(symbol='BTCUSDT', data_path=str(DATA_PATH))
scaled_config = multi_scaler.get_scaled_config_for_timeframe('15m')

coordinator = OptunaCoordinator(
    symbol='BTCUSDT',
    timeframe='15m',
    data_path=str(DATA_PATH),
    scaled_config=scaled_config
)

print('--- Layer0 ---')
processed_cleaned_dir = DATA_PATH / 'processed' / 'cleaned' / 'BTCUSDT_15m'
need_layer0 = True
try:
    if processed_cleaned_dir.exists():
        has_any = any(processed_cleaned_dir.glob('cleaned_ohlcv*.parquet')) or \
                  any(processed_cleaned_dir.glob('cleaned_ohlcv*.pkl')) or \
                  any(processed_cleaned_dir.glob('cleaned_ohlcv*.pickle'))
        need_layer0 = not has_any
except Exception:
    pass

if need_layer0:
    layer0_result = coordinator.run_layer0_data_cleaning(n_trials=50)
    print(layer0_result)
else:
    print(f"跳過 Layer0，已存在清洗檔: {processed_cleaned_dir}")

print('--- Layer1 ---')
layer1_result = coordinator.run_layer1_label_optimization(n_trials=150)
print(layer1_result)

# ✅ Layer1結果驗證
if 'best_score' in layer1_result:
    print(f"\n📊 Layer1分析:")
    print(f"  F1分數: {layer1_result['best_score']:.4f}")
    if layer1_result['best_score'] < 0.45:
        print(f"  ⚠️ 警告: 分數低於閾值0.45")
    
    # 檢查標籤分布
    metadata = layer1_result.get('metadata', {})
    if 'label_distribution' in metadata or 'data_columns' in metadata:
        print(f"  特徵信息: {metadata.get('data_shape', 'N/A')}")

print('\n' + '='*60)
print('--- Layer2 ---')
layer2_result = coordinator.run_layer2_feature_optimization(n_trials=300)
print(layer2_result)

# ✅ Layer2結果驗證
if 'best_score' in layer2_result:
    print(f"\n📊 Layer2分析:")
    print(f"  F1分數: {layer2_result['best_score']:.4f}")
    if layer2_result['best_score'] < 0.50:
        print(f"  ⚠️ 警告: 分數低於閾值0.50")
    
    # 檢查選擇的特徵
    if 'best_params' in layer2_result and 'selected_features' in layer2_result['best_params']:
        selected = layer2_result['best_params']['selected_features']
        print(f"  選擇特徵數: {len(selected)}")
        
        # 檢查策略特徵保留情況
        strategy_features = [f for f in selected if any(p in f for p in ['wyk_', 'td_', 'micro_'])]
        print(f"  策略特徵保留: {len(strategy_features)}個")
        if len(strategy_features) > 0:
            print(f"    ✅ Wyckoff/TD/Micro策略特徵已保留")
        else:
            print(f"    ⚠️ 警告: 未選中策略特徵")

print('\n' + '='*60)
print('✅ 優化完成！')