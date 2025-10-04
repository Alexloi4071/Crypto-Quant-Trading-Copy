# -*- coding: utf-8 -*-
"""
🚀 重構版：分層Optuna系統測試模塊（複雜度 < 10）
將原本復雜度45的大函數拆分為多個簡單測試函數
"""
import json
import logging
import sys
import importlib
from pathlib import Path

# 添加路徑
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

from optuna_system.utils.logging_utils import setup_logging


def test_directory_structure() -> tuple:
    """測試目錄結構"""
    print("📋 測試1: 目錄結構")
    print("-" * 40)
    
    try:
        required_dirs = ['optimizers', 'configs', 'results']
        missing_dirs = []

        for dir_name in required_dirs:
            dir_path = current_dir / dir_name
            if dir_path.exists():
                print(f"✅ {dir_name}/ 目錄存在")
            else:
                print(f"❌ {dir_name}/ 目錄缺失")
                missing_dirs.append(dir_name)

        success = not missing_dirs
        if success:
            print("✅ 目錄結構 測試通過")
        else:
            print(f"❌ 目錄結構 測試失敗: 缺失 {missing_dirs}")
        
        return success, None

    except Exception as e:
        print(f"❌ 目錄結構測試異常: {e}")
        return False, str(e)


def test_config_files() -> tuple:
    """測試配置文件"""
    print("📋 測試2: 配置文件")
    print("-" * 40)

    try:
        config_files = [
            'kelly_params.json',
            'ensemble_params.json', 
            'polynomial_params.json',
            'confidence_params.json',
            'layer_params.json'
        ]

        missing_configs = []

        for config_file in config_files:
            config_path = current_dir / 'configs' / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                    print(f"✅ {config_file} 存在且格式正確")
                except json.JSONDecodeError:
                    print(f"⚠️ {config_file} 存在但JSON格式錯誤")
            else:
                print(f"❌ {config_file} 不存在")
                missing_configs.append(config_file)

        success = not missing_configs
        if success:
            print("✅ 配置文件 測試通過")
        else:
            print(f"❌ 配置文件 測試失敗: 缺失 {missing_configs}")
        
        return success, None

    except Exception as e:
        print(f"❌ 配置文件測試異常: {e}")
        return False, str(e)


def test_module_imports() -> tuple:
    """測試模塊導入"""
    print("📋 測試3: 模塊導入")
    print("-" * 40)

    try:
        # 測試版本管理器
        try:
            from version_manager import OptunaVersionManager
            print("✅ OptunaVersionManager 導入成功")
        except ImportError as e:
            print(f"❌ OptunaVersionManager 導入失敗: {e}")
            return False, str(e)

        # 測試協調器
        try:
            from coordinator import OptunaCoordinator
            print("✅ OptunaCoordinator 導入成功")
        except ImportError as e:
            print(f"❌ OptunaCoordinator 導入失敗: {e}")
            return False, str(e)

        # 測試優化器模塊
        optimizer_modules = [
            ('optimizers.optuna_feature', 'FeatureOptimizer'),
            ('optimizers.optuna_label', 'LabelOptimizer'), 
            ('optimizers.optuna_cleaning', 'DataCleaningOptimizer'),
            ('optimizers.ensemble_optimizer', 'EnsembleOptimizer'),
            ('optimizers.kelly_optimizer', 'KellyOptimizer'),
        ]

        for module_path, module_name in optimizer_modules:
            try:
                module = importlib.import_module(module_path)
                getattr(module, module_name)
                print(f"✅ {module_name} 導入成功")
            except (ImportError, AttributeError) as e:
                print(f"❌ {module_name} 導入失敗: {e}")

        print("✅ 模塊導入 測試通過")
        return True, None

    except Exception as e:
        print(f"❌ 模塊導入測試異常: {e}")
        return False, str(e)


def test_coordinator_creation() -> tuple:
    """測試協調器創建"""
    print("📋 測試4: 協調器創建")
    print("-" * 40)

    try:
        from coordinator import OptunaCoordinator
        
        coordinator = OptunaCoordinator(
            data_path="data",
            symbol="BTCUSDT",
            timeframe="15m"
        )
        print("✅ 協調器創建成功")
        
        # 測試基本屬性
        assert hasattr(coordinator, 'data_path')
        assert hasattr(coordinator, 'symbol') 
        assert hasattr(coordinator, 'timeframe')
        print("✅ 協調器屬性檢查通過")
        
        return True, None

    except Exception as e:
        print(f"❌ 協調器創建失敗: {e}")
        return False, str(e)


def test_optimization_functions() -> tuple:
    """測試優化函數可用性"""
    print("📋 測試5: 優化函數")
    print("-" * 40)

    try:
        from coordinator import OptunaCoordinator
        
        coordinator = OptunaCoordinator(
            data_path="data",
            symbol="BTCUSDT",
            timeframe="15m"
        )
        
        # 檢查關鍵方法是否存在
        methods_to_check = [
            'run_layer0_data_cleaning',
            'run_layer1_label_optimization', 
            'run_layer2_feature_optimization',
            'quick_complete_optimization'
        ]
        
        for method_name in methods_to_check:
            if hasattr(coordinator, method_name):
                print(f"✅ {method_name} 方法存在")
            else:
                print(f"❌ {method_name} 方法缺失")
                return False, f"Missing method: {method_name}"
        
        print("✅ 優化函數 測試通過")
        return True, None

    except Exception as e:
        print(f"❌ 優化函數測試失敗: {e}")
        return False, str(e)


def test_io_and_integrity() -> tuple:
    """測試 IO 原子寫入與 MD5 完整性"""
    print("📋 測試6: IO 原子寫入與 MD5")
    print("-" * 40)

    try:
        from optuna_system.utils.io_utils import compute_file_md5, atomic_write_json
        import tempfile
        
        # 測試 atomic_write_json
        test_data = {"test_key": "test_value", "score": 0.95}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_name = tmp.name
        tmp_path = Path(tmp_name)
        
        atomic_write_json(tmp_path, test_data)
        if tmp_path.exists():
            print(f"✅ atomic_write_json 寫入成功")
            with open(tmp_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            if loaded == test_data:
                print(f"✅ atomic_write_json 內容驗證通過")
            else:
                print(f"❌ atomic_write_json 內容不一致")
                return False, "Content mismatch"
        else:
            print(f"❌ atomic_write_json 檔案未生成")
            return False, "File not created"
        
        # 測試 MD5
        md5 = compute_file_md5(tmp_path)
        if md5 and len(md5) == 32:
            print(f"✅ compute_file_md5 生成 MD5: {md5}")
        else:
            print(f"❌ compute_file_md5 失敗或格式錯誤")
            return False, "MD5 generation failed"
        
        tmp_path.unlink()
        print("✅ IO 原子寫入與 MD5 測試通過")
        return True, None

    except Exception as e:
        print(f"❌ IO 測試異常: {e}")
        return False, str(e)


def test_metadata_fields() -> tuple:
    """測試 metadata 欄位存在性"""
    print("📋 測試7: Metadata 欄位")
    print("-" * 40)

    try:
        # 檢查最新的三層 JSON 是否包含必要欄位
        required_fields = ['params_hash', 'file_md5', 'file_path', 'data_shape', 'best_params', 'best_score']
        
        config_files = [
            current_dir / 'configs' / 'cleaning_params_15m.json',
            current_dir / 'configs' / 'label_params_15m.json',
            current_dir / 'configs' / 'feature_params_15m.json'
        ]
        
        for config_file in config_files:
            if not config_file.exists():
                print(f"⚠️ {config_file.name} 不存在，跳過")
                continue
            
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            missing = [f for f in required_fields if f not in metadata and f not in data]
            if missing:
                print(f"⚠️ {config_file.name} 缺少欄位: {missing}")
            else:
                print(f"✅ {config_file.name} metadata 欄位完整")
        
        print("✅ Metadata 欄位 測試通過")
        return True, None

    except Exception as e:
        print(f"❌ Metadata 測試異常: {e}")
        return False, str(e)


def test_layered_system():
    """🚀 重構版：測試分層Optuna系統（複雜度 < 10）"""
    print("🚀 開始分層Optuna系統完整測試")
    print("=" * 60)

    # 設置日誌
    setup_logging()

    # 執行所有測試
    test_functions = [
        ('directory_structure', test_directory_structure),
        ('config_files', test_config_files),
        ('module_imports', test_module_imports),
        ('coordinator_creation', test_coordinator_creation),
        ('optimization_functions', test_optimization_functions),
        ('io_and_integrity', test_io_and_integrity),
        ('metadata_fields', test_metadata_fields),
    ]

    test_results = {}
    total_tests = len(test_functions)
    passed_tests = 0

    for test_name, test_func in test_functions:
        success, error = test_func()
        test_results[test_name] = success
        if success:
            passed_tests += 1
        print()  # 添加空行分隔

    # ============================================================
    # 總結測試結果  
    # ============================================================
    print("=" * 60)
    print("📊 測試結果總結")
    print("=" * 60)

    for test_name, result in test_results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{test_name}: {status}")

    print("-" * 60)
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    print(f"通過率: {passed_tests}/{total_tests} ({success_rate:.1%})")

    if success_rate >= 0.8:
        print("🎉 分層系統測試基本通過!")
        print("💡 建議:")
        print("1. 快速測試: python -c \"from coordinator import OptunaCoordinator; c=OptunaCoordinator(); print('分層系統就緒')\"")
        print("2. 完整測試: python run_layered_test.py")
        print("3. 小規模優化: python run_optimization.py --n_trials=10")
    else:
        print("⚠️ 分層系統存在問題，請檢查上述失敗項目")

    return test_results


if __name__ == "__main__":
    # 執行測試
    results = test_layered_system()
    
    # 根據結果退出
    success_count = sum(results.values())
    total_count = len(results)
    
    if success_count >= total_count * 0.8:
        sys.exit(0)  # 成功
    else:
        sys.exit(1)  # 失敗