#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理旧的Layer0-2优化结果，强制重新运行
"""
import shutil
from pathlib import Path

def delete_if_exists(path, description):
    """删除路径如果存在"""
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
            print(f"✅ 已删除: {description}")
            print(f"   路径: {path}")
        else:
            path.unlink()
            print(f"✅ 已删除文件: {description}")
            print(f"   路径: {path}")
        return True
    else:
        print(f"⚠️  不存在: {description}")
        print(f"   路径: {path}")
        return False

def main():
    print("="*60)
    print("开始清理旧的Layer0-2优化结果")
    print("="*60)
    
    base_path = Path(".")
    
    # 1. 删除Layer0清洗数据
    print("\n【Layer0 清洗数据】")
    cleaned_dir = base_path / "data" / "processed" / "cleaned" / "BTCUSDT_15m"
    delete_if_exists(cleaned_dir, "Layer0清洗数据目录")
    
    # 也删除configs下的旧清洗文件（如果有）
    old_cleaned = base_path / "configs" / "cleaned_ohlcv_15m.parquet"
    delete_if_exists(old_cleaned, "configs下的旧清洗文件")
    
    # 2. 删除Layer1标签数据
    print("\n【Layer1 标签数据】")
    labels_dir = base_path / "data" / "processed" / "labels" / "BTCUSDT_15m"
    delete_if_exists(labels_dir, "Layer1标签数据目录")
    
    # 3. 删除Layer2特征数据（如果有）
    print("\n【Layer2 特征数据】")
    features_dir = base_path / "data" / "processed" / "features" / "BTCUSDT_15m"
    delete_if_exists(features_dir, "Layer2特征数据目录")
    
    # 4. 删除旧的优化参数文件（可选，保留参数历史可能有用）
    print("\n【优化参数文件】")
    print("注意：保留参数文件以便对比，不删除")
    
    configs = base_path / "configs"
    if configs.exists():
        cleaning_params = configs / "cleaning_params.json"
        label_params = configs / "label_params.json"
        feature_params = configs / "feature_params.json"
        
        for param_file in [cleaning_params, label_params, feature_params]:
            if param_file.exists():
                print(f"   保留: {param_file.name}")
    
    print("\n" + "="*60)
    print("清理完成！")
    print("="*60)
    print("\n现在可以运行: python run_core_layers_only.py")
    print("\n预计时间:")
    print("  - Layer0:  1-2小时  (50 trials)")
    print("  - Layer1:  2-3小时  (150 trials)")
    print("  - Layer2:  4-6小时  (300 trials)")
    print("  - 总计:    7-11小时")
    print("\n建议在晚上或周末运行！")

if __name__ == "__main__":
    main()

