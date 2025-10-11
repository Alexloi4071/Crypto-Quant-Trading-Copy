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
    # 讓協調器自動尋找/回退上一層資料（不再硬性依賴固定版本特徵檔）
    # 使用分層協調器進行特徵優化
    coordinator = OptunaCoordinator(
        symbol="BTCUSDT",
        timeframe="15m",
        data_path="data"
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
