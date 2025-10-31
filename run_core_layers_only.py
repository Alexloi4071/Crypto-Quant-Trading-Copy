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

# 🎯 Trials数量配置（基于学术计算）
# 
# 参数空间分析：
# - Layer1 Primary: 10维参数 → 最小100 trials (Bergstra: 10×d)
# - Layer1 Meta: 6维参数，双目标 → NSGA-II推荐400-600 trials
# - Layer2: 20维参数 (7特征+13模型) → 最小200 trials
# 
# 学术依据：
# - Bergstra & Bengio (2012): n_trials ≥ 10 × 参数维度
# - Akiba et al. (2019): TPE采样器30+ trials后效果显著
# - Deb et al. (2002): NSGA-II推荐 population(30-40) × generations(10-20)
# 
# 三种运行模式：
#   1. BUG检查:  L1=5,   L2=5   (3-5分钟，验证无错误)
#   2. 标准测试: L1=50,  L2=100 (30-40分钟，充分验证) ✅ 推荐
#   3. 完整优化: L1=600, L2=250 (2-4小时，生产环境)
# 
# 当前配置: 完整优化模式
TEST_MODE = False  # 改为 True 使用标准测试模式
if TEST_MODE:
    # 🎯 标准测试模式：充分验证所有修复
    DEFAULT_L1_TRIALS = 50   # Primary最优(from 10) + Meta NSGA-II(40, pop=30)
    DEFAULT_L2_TRIALS = 100  # 5种特征方法 × 20 trials = 充分探索
    print("🎯 标准测试模式: L1=50 trials, L2=100 trials")
    print("   预计时间: 30-40分钟")
    print("   目标: 充分验证所有修复效果")
    print("   学术依据: Bergstra (10×d), Akiba (TPE 30+)")
else:
    # 🚀 完整优化模式（生产环境）
    DEFAULT_L1_TRIALS = 600  # Primary最优(from 200) + Meta NSGA-II(400, pop=40×gen=15)
    DEFAULT_L2_TRIALS = 250  # 5种特征方法 × 50 trials = 充分探索
    print("🚀 完整优化模式: L1=600 trials, L2=250 trials")
    print("   预计时间: 2-4小时")
    print("   目标: 最终生产环境优化")
    print("   NSGA-II配置: population=40, generations=15")

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
L1_TRIALS = int(os.getenv('L1_TRIALS', str(DEFAULT_L1_TRIALS)))
L2_TRIALS = int(os.getenv('L2_TRIALS', str(DEFAULT_L2_TRIALS)))
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
layer1_result = coordinator.run_layer1_label_optimization(n_trials=L1_TRIALS)
print(layer1_result)

# ✅ Layer1結果驗證
if isinstance(layer1_result, dict):
    if 'best_score' in layer1_result:
        print(f"\n📊 Layer1分析:")
        print(f"  F1分數: {layer1_result['best_score']:.4f}")
        if layer1_result['best_score'] < 0.45:
            print(f"  ⚠️ 警告: 分數低於閾值0.45")

        # 檢查標籤分布
        metadata = layer1_result.get('metadata', {})
        if 'label_distribution' in metadata or 'data_columns' in metadata:
            print(f"  特徵信息: {metadata.get('data_shape', 'N/A')}")
    else:
        print("⚠️ Layer1 未提供 best_score，輸出: ", layer1_result)

print('\n' + '='*60)

print('--- Layer2 ---')
layer2_result = coordinator.run_layer2_feature_optimization(n_trials=L2_TRIALS)
print(layer2_result)

if isinstance(layer2_result, dict):
    if 'best_score' in layer2_result:
        print(f"\n📊 Layer2分析:")
        print(f"  最佳分數: {layer2_result['best_score']:.4f}")
        mat_path = layer2_result.get('materialized_path') or layer2_result.get('metadata', {}).get('materialized_path')
        if mat_path:
            print(f"  物化特徵檔案: {mat_path}")
    else:
        print("⚠️ Layer2 未提供 best_score，輸出: ", layer2_result)

print('\n' + '='*60)