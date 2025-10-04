# -*- coding: utf-8 -*-
"""
Optuna系統測試腳本
快速驗證所有組件是否正常工作
"""
import sys
from pathlib import Path

from optuna_system.utils.logging_utils import setup_logging

# 添加路徑
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def test_directory_structure():
    """測試目錄結構"""
    print("\n🔍 測試目錄結構...")
    
    required_dirs = [
        'optimizers',
        'configs', 
        'results'
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            print(f"✅ {dir_name}/ 目錄存在")
        else:
            print(f"❌ {dir_name}/ 目錄不存在")
            all_exist = False
    
    return all_exist

def test_configs():
    """測試配置文件"""
    print("\n🔍 測試配置文件...")
    
    configs_path = Path("configs")
    required_configs = [
        'kelly_params.json',
        'ensemble_params.json',
        'polynomial_params.json', 
        'confidence_params.json'
    ]
    
    all_exist = True
    for config_file in required_configs:
        config_path = configs_path / config_file
        if config_path.exists():
            print(f"✅ {config_file} 存在")
            
            # 測試載入
            try:
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"   - 格式正確，包含 {len(data)} 個配置項")
            except Exception as e:
                print(f"   ❌ 格式錯誤: {e}")
                all_exist = False
        else:
            print(f"❌ {config_file} 不存在")
            all_exist = False
    
    return all_exist

def test_imports():
    """測試模塊導入"""
    print("\n🔍 測試模塊導入...")
    
    try:
        from version_manager import OptunaVersionManager
        print("✅ 版本管理器導入成功")
    except Exception as e:
        print(f"❌ 版本管理器導入失敗: {e}")
        return False
    
    try:
        from coordinator import OptunaCoordinator
        print("✅ 協調器導入成功")
    except Exception as e:
        print(f"❌ 協調器導入失敗: {e}")
        return False
    
    # 測試優化器導入
    optimizer_modules = [
        ('optimizers.kelly_optimizer', 'KellyOptimizer'),
        ('optimizers.ensemble_optimizer', 'EnsembleOptimizer'), 
        ('optimizers.polynomial_optimizer', 'PolynomialOptimizer'),
        ('optimizers.confidence_optimizer', 'ConfidenceOptimizer')
    ]
    
    for module_name, class_name in optimizer_modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {class_name} 導入成功")
        except Exception as e:
            print(f"⚠️ {class_name} 導入失敗: {e}")
    
    return True

def test_version_manager():
    """測試版本管理器"""
    print("\n🔍 測試版本管理器...")
    
    try:
        from version_manager import OptunaVersionManager
        
        vm = OptunaVersionManager("results")
        
        # 測試創建版本
        version = vm.create_new_version()
        print(f"✅ 創建版本成功: {version}")
        
        # 測試保存結果
        test_results = {
            'kelly': {
                'best_params': {'kelly_fraction': 0.25},
                'best_score': 0.85,
                'n_trials': 10
            }
        }
        
        success = vm.save_results(version, test_results)
        if success:
            print("✅ 保存結果成功")
        else:
            print("❌ 保存結果失敗")
        
        return True
        
    except Exception as e:
        print(f"❌ 版本管理器測試失敗: {e}")
        return False

def test_coordinator():
    """測試協調器"""
    print("\n🔍 測試協調器...")
    
    try:
        from coordinator import OptunaCoordinator
        
        # 創建協調器
        coordinator = OptunaCoordinator(
            symbol='BTCUSDT',
            timeframe='15m',
            data_path='../data'
        )
        print("✅ 協調器創建成功")
        
        # 測試加載配置
        configs = coordinator.load_default_configs()
        print(f"✅ 加載配置成功，模塊數: {len(configs)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 協調器測試失敗: {e}")
        return False

def run_full_test():
    """運行完整測試"""
    setup_logging()
    print("🚀 開始Optuna系統完整測試\n")
    print("="*60)
    
    tests = [
        ("目錄結構", test_directory_structure),
        ("配置文件", test_configs),
        ("模塊導入", test_imports),
        ("版本管理器", test_version_manager),
        ("協調器", test_coordinator)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 測試: {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                print(f"✅ {test_name} 測試通過")
                passed += 1
            else:
                print(f"❌ {test_name} 測試失敗")
        except Exception as e:
            print(f"💥 {test_name} 測試異常: {e}")
    
    # 生成總結報告
    print("\n" + "="*60)
    print("📊 測試總結")
    print("="*60)
    print(f"通過率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有測試通過！Optuna系統可以正常使用。")
    elif passed >= total * 0.8:
        print("✅ 大部分測試通過，系統基本可用。")
    else:
        print("❌ 測試失敗較多，請檢查系統配置。")
    
    return passed == total

if __name__ == "__main__":
    run_full_test()
