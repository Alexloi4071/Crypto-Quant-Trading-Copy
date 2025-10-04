#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多時間框架優化運行腳本
基於BTCUSDT_15m優化結果，自動縮放到其他時間框架
"""
import sys
sys.path.append('.')

from config.timeframe_scaler import MultiTimeframeCoordinator


def main():
    """運行多時間框架優化"""
    print("🚀 開始多時間框架參數自動縮放優化...")
    print("基於BTCUSDT_15m優化結果自動適配")
    print("-" * 60)
    
    # 初始化多時框協調器
    multi_coordinator = MultiTimeframeCoordinator(symbol='BTCUSDT')
    
    # 顯示縮放映射
    print("📊 參數縮放映射：")
    for timeframe in multi_coordinator.supported_timeframes:
        config = multi_coordinator.get_scaled_config_for_timeframe(timeframe)
        scale = multi_coordinator.scaler.get_scale_factor(timeframe)
        print(f"  {timeframe}: 縮放係數={scale:.2f}, lag={config.get('label_lag', 12)}, "
              f"purge={config.get('purge_period', 192)}, pos_quantile={config.get('pos_quantile', 0.85)}")
    
    print("-" * 60)
    
    # 一鍵優化所有時間框架（文檔目標）
    results = multi_coordinator.optimize_all_timeframes(n_trials=50)  # 每個時框50次試驗
    
    print("\n🎉 多時間框架優化完成！")
    
    # 顯示結果對比
    print("\n📈 最終結果對比：")
    for timeframe, result in results.items():
        if 'error' not in result:
            summary = result.get('optimization_summary', {})
            score = summary.get('average_score', result.get('best_score', 'N/A'))
            print(f"  {timeframe:>4}: F1={score:.3f} (Layer2特徵優化)")
        else:
            print(f"  {timeframe:>4}: 失敗 - {result['error']}")
    
    return results


if __name__ == "__main__":
    results = main()
