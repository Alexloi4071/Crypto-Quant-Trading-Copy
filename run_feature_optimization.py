#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
執行 BTCUSDT_15m 特徵選擇超參數優化
"""

import sys
import os
sys.path.append('.')

from optuna_system.coordinator import OptunaCoordinator

def main():
    print("🚀 開始 BTCUSDT_15m 特徵選擇超參數優化...")
    
    # 檢查數據文件是否存在
    feature_file = 'data/processed/features/BTCUSDT_15m/v55/BTCUSDT_15m_selected_features.parquet'
    
    if not os.path.exists(feature_file):
        print(f"❌ 特徵文件不存在: {feature_file}")
        return
    
    print(f"✅ 找到特徵文件: {feature_file}")
    
    # 使用分層協調器進行特徵優化
    coordinator = OptunaCoordinator(
        symbol="BTCUSDT",
        timeframe="15m",
        data_path="data/processed",
        version="v55"
    )
    
    # 僅執行第2層特徵工程優化
    print("📊 執行第2層：特徵工程參數優化...")
    result = coordinator.run_layer2_feature_optimization(n_trials=20)
    
    if 'error' in result:
        print(f"❌ 優化失敗: {result['error']}")
    else:
        print(f"✅ 優化完成! 最佳得分: {result.get('best_score', 'N/A')}")
        print(f"📊 最優參數: {result.get('best_params', 'N/A')}")
        
        # 保存結果
        result_file = f"optuna_system/results/feature_optimization_BTCUSDT_15m.json"
        os.makedirs("optuna_system/results", exist_ok=True)
        
        import json
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"💾 結果已保存至: {result_file}")

if __name__ == "__main__":
    main()
