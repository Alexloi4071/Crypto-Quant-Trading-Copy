# -*- coding: utf-8 -*-
"""
Primary Label Optimizer 單元測試
測試 Layer 1A: Primary Model 的各項功能
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# 添加項目根目錄到路徑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optuna_system.optimizers.optuna_primary_label import PrimaryLabelOptimizer


class TestPrimaryLabelOptimizer:
    """Primary Label Optimizer 測試類"""
    
    @pytest.fixture
    def optimizer(self):
        """創建測試用的優化器實例"""
        return PrimaryLabelOptimizer(
            data_path='data',
            config_path='configs',
            symbol='BTCUSDT',
            timeframe='15m'
        )
    
    @pytest.fixture
    def sample_params(self):
        """測試用的參數"""
        return {
            'lag': 12,
            'atr_period': 14,
            'profit_multiplier': 2.0,
            'stop_multiplier': 1.5,
            'max_holding': 20,
            'enable_trailing_stop': True,
            'trailing_activation_ratio': 0.5,
            'trailing_distance_ratio': 0.7,
            'trailing_lock_min_profit': 0.3,
            'transaction_cost_bps': 10.0,
        }
    
    def test_init(self, optimizer):
        """測試初始化"""
        assert optimizer.symbol == 'BTCUSDT'
        assert optimizer.timeframe == '15m'
        assert optimizer.price_data is not None
        assert len(optimizer.price_data) > 0
    
    def test_timeframe_conversion(self, optimizer):
        """測試時間框轉換"""
        assert optimizer._timeframe_to_minutes('15m') == 15.0
        assert optimizer._timeframe_to_minutes('1h') == 60.0
        assert optimizer._timeframe_to_minutes('4h') == 240.0
        assert optimizer._timeframe_to_minutes('1d') == 1440.0
    
    def test_atr_calculation(self, optimizer):
        """測試 ATR 計算"""
        high = pd.Series([100, 102, 101, 103, 102])
        low = pd.Series([98, 99, 99, 100, 99])
        close = pd.Series([99, 101, 100, 102, 101])
        
        atr = optimizer.calculate_atr(high, low, close, period=3)
        
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(close)
        assert not atr.isna().all()  # 至少有一些非 NaN 值
    
    def test_triple_barrier_labels(self, optimizer, sample_params):
        """測試 Triple Barrier 標籤生成"""
        price_data = optimizer.price_data['close'].iloc[:1000]
        
        labels = optimizer.generate_triple_barrier_labels(price_data, sample_params)
        
        # 基本驗證
        assert isinstance(labels, pd.Series)
        assert len(labels) > 0
        assert labels.isin([0, 1, 2]).all()  # 只包含 0, 1, 2
        
        # 統計驗證
        label_counts = labels.value_counts()
        assert len(label_counts) >= 2  # 至少有兩種標籤
    
    def test_primary_signals_generation(self, optimizer, sample_params):
        """測試 Primary 信號生成（二分類）"""
        price_data = optimizer.price_data['close'].iloc[:1000]
        
        signals = optimizer.generate_primary_signals(price_data, sample_params)
        
        # 基本驗證
        assert isinstance(signals, pd.Series)
        assert len(signals) > 0
        assert signals.isin([1, -1]).all()  # 只包含 1 (買入) 和 -1 (賣出)
        
        # 平衡性驗證（應該接近 50/50）
        buy_count = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        total = len(signals)
        buy_ratio = buy_count / total
        
        print(f"   買入比例: {buy_ratio:.2%}")
        print(f"   賣出比例: {(1-buy_ratio):.2%}")
        
        # 允許一定偏差（30% - 70%）
        assert 0.3 <= buy_ratio <= 0.7, f"買入比例 {buy_ratio:.2%} 偏離 50% 過多"
    
    def test_objective_function(self, optimizer):
        """測試目標函數"""
        import optuna
        
        study = optuna.create_study(direction='maximize')
        
        # 運行 3 個試驗
        study.optimize(optimizer.objective, n_trials=3, show_progress_bar=False)
        
        # 驗證
        assert study.best_value is not None
        assert study.best_value > -999  # 不應該是錯誤值
        assert len(study.best_params) > 0
    
    def test_optimize_integration(self, optimizer):
        """測試完整優化流程（集成測試）"""
        result = optimizer.optimize(n_trials=5)
        
        # 驗證結果結構
        assert 'best_params' in result
        assert 'best_score' in result
        assert 'accuracy' in result
        assert 'sharpe' in result
        assert 'buy_ratio' in result
        
        # 驗證結果範圍
        assert result['best_score'] > 0, "最佳得分應該為正數"
        assert 0 <= result['accuracy'] <= 1, "準確率應該在 0-1 之間"
        assert 0 <= result['buy_ratio'] <= 1, "買入比例應該在 0-1 之間"
        
        print(f"\n   ✅ 優化完成:")
        print(f"      最佳得分: {result['best_score']:.4f}")
        print(f"      方向準確率: {result['accuracy']:.3f}")
        print(f"      Sharpe: {result['sharpe']:.2f}")
        print(f"      買入比例: {result['buy_ratio']:.2%}")
    
    def test_apply_labels(self, optimizer, sample_params):
        """測試標籤應用"""
        data = optimizer.price_data.iloc[:1000].copy()
        
        result = optimizer.apply_labels(data, sample_params)
        
        # 驗證輸出
        assert 'primary_signal' in result.columns
        assert result['primary_signal'].isin([1, -1]).all()
        assert len(result) > 0


def test_quick_validation():
    """快速驗證測試（獨立運行）"""
    print("\n🚀 Primary Label Optimizer 快速驗證")
    
    optimizer = PrimaryLabelOptimizer(
        data_path='data',
        config_path='configs'
    )
    
    print(f"✅ 數據載入: {len(optimizer.price_data)} 行")
    
    # 測試信號生成
    params = {
        'lag': 12,
        'atr_period': 14,
        'profit_multiplier': 2.0,
        'stop_multiplier': 1.5,
        'max_holding': 20,
        'enable_trailing_stop': True,
        'trailing_activation_ratio': 0.5,
        'trailing_distance_ratio': 0.7,
        'trailing_lock_min_profit': 0.3,
        'transaction_cost_bps': 10.0,
    }
    
    signals = optimizer.generate_primary_signals(
        optimizer.price_data['close'].iloc[:1000],
        params
    )
    
    buy_ratio = (signals == 1).sum() / len(signals)
    print(f"✅ 信號生成: {len(signals)} 個信號")
    print(f"   買入: {buy_ratio:.1%}, 賣出: {(1-buy_ratio):.1%}")
    
    assert 0.2 <= buy_ratio <= 0.8, "信號分佈異常"
    print("✅ 所有驗證通過!")


if __name__ == "__main__":
    # 獨立運行測試
    test_quick_validation()

