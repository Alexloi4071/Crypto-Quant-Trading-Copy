#!/usr/bin/env python3
"""
🧪 測試新標籤邏輯
快速測試基於盈利的標籤生成是否正常工作
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.optimization.label_optimizer import LabelOptimizer

def test_new_label_logic():
    """測試新的標籤生成邏輯"""
    print("🧪 測試新標籤邏輯 - 基於未來實際盈利能力")
    print("=" * 60)
    
    # 1. 創建測試數據 (模擬BTCUSDT價格)
    print("📊 1. 創建測試數據...")
    dates = pd.date_range('2023-01-01', periods=1000, freq='15min')
    
    # 模擬價格：基礎趨勢 + 隨機波動
    base_price = 20000
    trend = np.linspace(0, 0.2, 1000)  # 20%上漲趨勢
    noise = np.random.normal(0, 0.02, 1000)  # 2%隨機波動
    prices = base_price * (1 + trend + noise)
    
    price_series = pd.Series(prices, index=dates)
    print(f"   ✅ 生成 {len(price_series)} 條測試價格數據")
    print(f"   💰 價格範圍: ${price_series.min():.0f} - ${price_series.max():.0f}")
    
    # 2. 初始化標籤優化器
    print("\n🔧 2. 初始化標籤優化器...")
    optimizer = LabelOptimizer("BTCUSDT", "15m")
    
    # 3. 測試不同參數組合
    test_cases = [
        {
            "name": "保守型",
            "lag": 8,  # 2小時
            "profit_threshold": 0.005,  # 0.5%
            "loss_threshold": -0.005,   # -0.5%
            "threshold_method": "fixed"
        },
        {
            "name": "積極型", 
            "lag": 12,  # 3小時
            "profit_threshold": 0.008,  # 0.8%
            "loss_threshold": -0.008,   # -0.8%
            "threshold_method": "fixed"
        },
        {
            "name": "自適應型",
            "lag": 6,   # 1.5小時
            "profit_threshold": 0.003,  # 0.3%
            "loss_threshold": -0.003,   # -0.3%
            "threshold_method": "adaptive"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📈 3.{i} 測試 {case['name']} 參數...")
        print(f"   ⏰ 預測週期: {case['lag']} × 15分鐘 = {case['lag']*15/60:.1f}小時")
        print(f"   📊 盈利閾值: {case['profit_threshold']:.3f} ({case['profit_threshold']*100:.1f}%)")
        print(f"   📉 虧損閾值: {case['loss_threshold']:.3f} ({case['loss_threshold']*100:.1f}%)")
        
        try:
            labels = optimizer.generate_labels(
                price_data=price_series,
                lag=case['lag'],
                profit_threshold=case['profit_threshold'],
                loss_threshold=case['loss_threshold'],
                label_type="ternary",
                threshold_method=case['threshold_method']
            )
            
            print(f"   ✅ 成功生成 {len(labels)} 個標籤")
            
            # 計算理論盈利率
            future_returns = (price_series.shift(-case['lag']) / price_series - 1)[:-case['lag']]
            actual_profit = future_returns - 0.0006  # 扣除交易成本
            
            # 統計實際盈利情況
            profitable_trades = actual_profit > case['profit_threshold']
            losing_trades = actual_profit < case['loss_threshold']
            
            print(f"   📊 理論盈利分析:")
            print(f"      🟢 盈利交易: {profitable_trades.sum():,} ({profitable_trades.mean()*100:.1f}%)")
            print(f"      🔴 虧損交易: {losing_trades.sum():,} ({losing_trades.mean()*100:.1f}%)")
            print(f"      🟡 中性交易: {(~profitable_trades & ~losing_trades).sum():,}")
            
            # 計算如果按標籤交易的理論收益
            buy_signals = labels == 2
            sell_signals = labels == 0
            
            if buy_signals.any():
                buy_returns = actual_profit[buy_signals]
                avg_buy_return = buy_returns.mean()
                print(f"      💰 買入信號平均收益: {avg_buy_return:.4f} ({avg_buy_return*100:.2f}%)")
            
            if sell_signals.any():
                # 對於賣出信號，收益應該是反向的
                sell_returns = -actual_profit[sell_signals]  # 做空收益
                avg_sell_return = sell_returns.mean()
                print(f"      💰 賣出信號平均收益: {avg_sell_return:.4f} ({avg_sell_return*100:.2f}%)")
            
        except Exception as e:
            print(f"   ❌ 測試失敗: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 新標籤邏輯測試完成！")
    print("👆 請檢查上面的標籤分佈和理論收益是否合理")

if __name__ == "__main__":
    test_new_label_logic()
