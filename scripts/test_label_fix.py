#!/usr/bin/env python3
"""
測試標籤修復邏輯 - 使用模擬資料
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.optimization.label_optimizer import LabelOptimizer


def create_btc_like_price_series(n_samples=50000, seed=42):
    """創建類似 BTC 15m 的價格序列"""
    np.random.seed(seed)
    
    # 模擬 3 年 15m 資料（約 105,120 個 bar）
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='15T')
    
    # 基礎價格走勢（趨勢 + 噪音）
    trend = np.linspace(20000, 60000, n_samples)  # BTC 從 2萬到 6萬
    
    # 加入波動性（隨機遊走 + 趨勢回歸）
    returns = np.random.normal(0, 0.015, n_samples)  # 1.5% 標準差（15m）
    
    # 偶發大波動（模擬新聞事件）
    big_moves = np.random.choice([0, 1], n_samples, p=[0.995, 0.005])
    returns += big_moves * np.random.normal(0, 0.05, n_samples)
    
    # 累積價格
    prices = trend[0] * np.exp(np.cumsum(returns))
    
    # 平滑回趨勢線（避免過度偏離）
    prices = 0.7 * prices + 0.3 * trend
    
    return pd.Series(prices, index=dates, name='close')


def main():
    print('🔄 建立模擬 BTC 15m 價格資料...')
    price_data = create_btc_like_price_series(n_samples=50000)
    print(f'✅ 生成價格序列: {len(price_data)} 個樣本')
    print(f'📊 價格範圍: ${price_data.min():.0f} - ${price_data.max():.0f}')
    print(f'📈 平均日波動: {price_data.pct_change().std() * np.sqrt(96):.2%}')  # 96 個 15m bar = 1天
    
    # 測試修復後的標籤生成
    symbol, timeframe = 'BTCUSDT', '15m'
    lo = LabelOptimizer(symbol, timeframe)
    
    print('\n🎯 測試 quantile 標籤生成（修復後）...')
    labels = lo.generate_labels(
        price_data=price_data,
        lag=3,
        profit_threshold=0.75,  # 75% 分位數
        loss_threshold=0.25,    # 25% 分位數
        label_type='multiclass',
        threshold_method='quantile'
    )
    
    # 分佈分析
    vc = labels.value_counts(normalize=True).sort_index()
    dist = {int(k): float(v) for k, v in vc.items()}
    
    print('\n✅ 標籤分佈結果:')
    print('   label_dist:', dist)
    print('   count:', len(labels))
    
    if len(dist) >= 3:
        sell_pct = dist.get(0, 0) * 100
        hold_pct = dist.get(1, 0) * 100
        buy_pct = dist.get(2, 0) * 100
        print(f'\n📈 詳細分佈:')
        print(f'   賣出 (0): {sell_pct:.1f}%')
        print(f'   持有 (1): {hold_pct:.1f}%') 
        print(f'   買入 (2): {buy_pct:.1f}%')
        
        # 評估分佈合理性
        target_msg = '目標: 多/空各 25~35%, 中性 30~50%'
        if 20 <= sell_pct <= 40 and 20 <= buy_pct <= 40 and 20 <= hold_pct <= 60:
            print(f'✅ 標籤分佈合理！({target_msg})')
        else:
            print(f'⚠️ 標籤分佈可能需要調整 ({target_msg})')
    
    # 測試其他分位數設定
    print('\n🔬 測試不同分位數設定:')
    test_configs = [
        (0.8, 0.2, '80/20'),
        (0.7, 0.3, '70/30'), 
        (0.65, 0.35, '65/35'),
    ]
    
    for pos_q, neg_q, name in test_configs:
        labels_test = lo.generate_labels(
            price_data=price_data,
            lag=3,
            profit_threshold=pos_q,
            loss_threshold=neg_q,
            label_type='multiclass',
            threshold_method='quantile'
        )
        vc_test = labels_test.value_counts(normalize=True).sort_index()
        dist_test = {int(k): float(v) for k, v in vc_test.items()}
        
        if len(dist_test) >= 3:
            sell_pct = dist_test.get(0, 0) * 100
            hold_pct = dist_test.get(1, 0) * 100
            buy_pct = dist_test.get(2, 0) * 100
            print(f'   {name}: 賣{sell_pct:.0f}% 持{hold_pct:.0f}% 買{buy_pct:.0f}%')
    
    print('\n🎉 標籤修復測試完成！')


if __name__ == '__main__':
    main()
