#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查Layer3运行结果"""

import pandas as pd
from pathlib import Path
import json

def check_results():
    """检查Layer3生成的结果"""
    
    print("=" * 80)
    print("Layer3 结果检查")
    print("=" * 80)
    
    # 1. 检查预测文件
    results_path = Path("optuna_system/results/BTCUSDT_15m")
    
    models = ['lgb', 'xgb', 'cat', 'rf', 'et']
    model_names = {
        'lgb': 'LightGBM',
        'xgb': 'XGBoost',
        'cat': 'CatBoost',
        'rf': 'RandomForest',
        'et': 'ExtraTrees'
    }
    
    print("\n=== 预测文件检查 ===")
    predictions_found = {}
    for model in models:
        pred_file = results_path / f"{model}_predictions.parquet"
        if pred_file.exists():
            df = pd.read_parquet(pred_file)
            print(f"[OK] {model_names[model]:15s}: {pred_file.name}")
            print(f"     形状: {df.shape} (样本数 x 列数)")
            print(f"     列: {list(df.columns)}")
            predictions_found[model] = True
        else:
            print(f"[--] {model_names[model]:15s}: 未找到")
            predictions_found[model] = False
    
    # 2. 检查配置文件
    print("\n=== 配置文件检查 ===")
    config_file = Path("configs/model_params.json")
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"[OK] model_params.json")
        print(f"     包含模型数: {len(config)}")
        
        for model, params in config.items():
            if 'best_score' in params:
                print(f"\n  {model_names.get(model, model)}:")
                print(f"    最佳分数: {params['best_score']:.4f}")
                if 'n_trials' in params:
                    print(f"    Trials数: {params['n_trials']}")
    else:
        print("[--] model_params.json: 未找到")
    
    # 3. 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    
    completed = sum(predictions_found.values())
    total = len(models)
    
    print(f"已完成模型: {completed}/{total}")
    
    if completed == 0:
        print("\n状态: 优化尚未开始或刚刚开始")
        print("建议: 等待优化完成")
    elif completed < total:
        print(f"\n状态: 优化进行中 ({completed/total*100:.0f}%)")
        print(f"已完成: {[m for m, f in predictions_found.items() if f]}")
        print(f"待完成: {[m for m, f in predictions_found.items() if not f]}")
    else:
        print("\n状态: 全部完成! ✓")
        print("\n下一步:")
        print("  1. 运行Layer6集成优化")
        print("  2. 或继续运行Layer4-9完整流程")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    check_results()

