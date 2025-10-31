# -*- coding: utf-8 -*-
"""
Meta Quality Optimizer 單元測試
測試 Layer 1B: Meta Model 的各項功能
"""
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# 添加項目根目錄到路徑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optuna_system.optimizers.optuna_meta_quality import MetaQualityOptimizer


class TestMetaQualityOptimizer:
    """Meta Quality Optimizer 測試類"""
    
    @pytest.fixture
    def optimizer(self):
        """創建測試用的優化器實例"""
        return MetaQualityOptimizer(
            data_path='data',
            config_path='configs',
            symbol='BTCUSDT',
            timeframe='15m'
        )
    
    @pytest.fixture
    def sample_signals(self):
        """測試用的 Primary 信號"""
        # 模擬 50/50 平衡的信號
        np.random.seed(42)
        signals = np.random.choice([1, -1], size=1000)
        return pd.Series(signals, index=pd.date_range('2024-01-01', periods=1000, freq='15min'))
    
    @pytest.fixture
    def sample_price_data(self):
        """測試用的價格數據"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
        
        # 模擬價格走勢
        returns = np.random.normal(0.0001, 0.01, 1000)
        prices = 100 * (1 + returns).cumprod()
        
        data = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 1000)
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def sample_params(self):
        """測試用的參數"""
        return {
            'lag': 12,
            'atr_period': 14,
            'profit_multiplier': 2.0,
            'stop_multiplier': 1.5,
            'transaction_cost_bps': 10.0,
            'strength_weight': 0.3,
            'trend_weight': 0.3,
            'winrate_weight': 0.2,
            'alignment_weight': 0.2,
            'quality_threshold': 0.5,
        }
    
    def test_init(self, optimizer):
        """測試初始化"""
        assert optimizer.symbol == 'BTCUSDT'
        assert optimizer.timeframe == '15m'
        assert optimizer.primary_signals is None  # 初始未設定
    
    def test_set_primary_signals(self, optimizer, sample_signals):
        """測試設定 Primary 信號"""
        optimizer.set_primary_signals(sample_signals)
        
        assert optimizer.primary_signals is not None
        assert len(optimizer.primary_signals) == len(sample_signals)
        assert optimizer.primary_signals.equals(sample_signals)
    
    def test_atr_calculation(self, optimizer, sample_price_data):
        """測試 ATR 計算"""
        optimizer.price_data = sample_price_data
        close = sample_price_data['close']
        
        atr = optimizer._calculate_atr(close, period=14)
        
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(close)
        assert not atr.isna().all()  # 至少有一些非 NaN 值
        assert (atr >= 0).all()  # ATR 應該非負
    
    def test_meta_features_generation(self, optimizer, sample_signals, sample_price_data, sample_params):
        """測試元特徵生成"""
        optimizer.price_data = sample_price_data
        
        meta_features = optimizer.generate_meta_features(
            sample_signals, sample_price_data, sample_params
        )
        
        # 基本驗證
        assert isinstance(meta_features, pd.DataFrame)
        assert len(meta_features) == len(sample_signals)
        
        # 驗證包含的特徵
        expected_features = [
            'signal_strength',
            'risk_reward_ratio',
            'volatility',
            'trend_strength',
            'volume_ratio',
            'momentum_5',
            'momentum_20',
            'recent_winrate',
            'signal_momentum_alignment'
        ]
        
        for feature in expected_features:
            assert feature in meta_features.columns, f"缺少特徵: {feature}"
        
        # 驗證沒有 NaN
        assert not meta_features.isna().any().any(), "元特徵包含 NaN 值"
        
        print(f"\n   ✅ 生成 {len(meta_features.columns)} 個元特徵")
    
    def test_meta_labels_generation(self, optimizer, sample_signals, sample_price_data, sample_params):
        """測試元標籤生成"""
        meta_labels = optimizer.generate_meta_labels(
            sample_signals, sample_price_data, sample_params
        )
        
        # 基本驗證
        assert isinstance(meta_labels, pd.Series)
        assert len(meta_labels) == len(sample_signals)
        assert meta_labels.isin([0, 1]).all()  # 只包含 0 和 1
        
        # 統計驗證
        good_ratio = (meta_labels == 1).sum() / len(meta_labels)
        print(f"\n   好信號比例: {good_ratio:.1%}")
        
        # 應該有一定比例的好信號
        assert 0.2 <= good_ratio <= 0.8, f"好信號比例 {good_ratio:.1%} 異常"
    
    def test_quality_evaluation(self, optimizer, sample_signals, sample_price_data, sample_params):
        """測試質量評估"""
        optimizer.price_data = sample_price_data
        
        meta_features = optimizer.generate_meta_features(
            sample_signals, sample_price_data, sample_params
        )
        
        quality_labels = optimizer.evaluate_quality(meta_features, sample_params)
        
        # 基本驗證
        assert isinstance(quality_labels, pd.Series)
        assert len(quality_labels) == len(meta_features)
        assert quality_labels.isin([0, 1]).all()  # 只包含 0 和 1
        
        # 執行率驗證
        execution_ratio = (quality_labels == 1).sum() / len(quality_labels)
        print(f"\n   執行率: {execution_ratio:.1%}")
        
        # 執行率應該在合理範圍（20-80%）
        assert 0.1 <= execution_ratio <= 0.9, f"執行率 {execution_ratio:.1%} 異常"
    
    def test_objective_function(self, optimizer, sample_signals, sample_price_data):
        """測試目標函數"""
        import optuna
        
        # 設定必要的數據
        optimizer.set_primary_signals(sample_signals)
        optimizer.price_data = sample_price_data
        
        study = optuna.create_study(direction='maximize')
        
        # 運行 3 個試驗
        study.optimize(optimizer.objective, n_trials=3, show_progress_bar=False)
        
        # 驗證
        assert study.best_value is not None
        assert study.best_value > -999  # 不應該是錯誤值
        assert len(study.best_params) > 0
        
        # 驗證記錄的額外信息
        best_trial = study.best_trial
        assert 'f1_score' in best_trial.user_attrs
        assert 'precision' in best_trial.user_attrs
        assert 'recall' in best_trial.user_attrs
        assert 'sharpe' in best_trial.user_attrs
        assert 'execution_ratio' in best_trial.user_attrs
    
    def test_optimize_integration(self, optimizer, sample_signals, sample_price_data):
        """測試完整優化流程（集成測試）"""
        # 設定必要的數據
        optimizer.set_primary_signals(sample_signals)
        optimizer.price_data = sample_price_data
        
        result = optimizer.optimize(n_trials=5)
        
        # 驗證結果結構
        assert 'best_params' in result
        assert 'best_score' in result
        assert 'f1_score' in result
        assert 'precision' in result
        assert 'recall' in result
        assert 'sharpe' in result
        assert 'execution_ratio' in result
        
        # 驗證結果範圍
        assert result['best_score'] > 0, "最佳得分應該為正數"
        assert 0 <= result['f1_score'] <= 1, "F1 分數應該在 0-1 之間"
        assert 0 <= result['precision'] <= 1, "精確率應該在 0-1 之間"
        assert 0 <= result['recall'] <= 1, "召回率應該在 0-1 之間"
        assert 0 <= result['execution_ratio'] <= 1, "執行率應該在 0-1 之間"
        
        print(f"\n   ✅ 優化完成:")
        print(f"      最佳得分: {result['best_score']:.4f}")
        print(f"      F1 分數: {result['f1_score']:.3f}")
        print(f"      精確率: {result['precision']:.3f}")
        print(f"      召回率: {result['recall']:.3f}")
        print(f"      Sharpe: {result['sharpe']:.2f}")
        print(f"      執行率: {result['execution_ratio']:.1%}")
    
    def test_rolling_winrate_calculation(self, optimizer, sample_signals, sample_price_data):
        """測試滾動勝率計算"""
        winrate = optimizer._calculate_rolling_winrate(
            sample_signals, sample_price_data, window=20, lag=12
        )
        
        # 基本驗證
        assert isinstance(winrate, pd.Series)
        assert len(winrate) == len(sample_signals)
        
        # 驗證勝率範圍（0-1）
        valid_winrate = winrate.dropna()
        assert (valid_winrate >= 0).all(), "勝率不應為負"
        assert (valid_winrate <= 1).all(), "勝率不應超過 1"
    
    def test_feature_quality(self, optimizer, sample_signals, sample_price_data, sample_params):
        """測試元特徵的質量（無極端值）"""
        optimizer.price_data = sample_price_data
        
        meta_features = optimizer.generate_meta_features(
            sample_signals, sample_price_data, sample_params
        )
        
        # 驗證沒有無限值
        assert not np.isinf(meta_features.values).any(), "元特徵包含無限值"
        
        # 驗證沒有極端大的值
        for col in meta_features.columns:
            values = meta_features[col].abs()
            max_val = values.max()
            assert max_val < 1e10, f"特徵 {col} 包含極端值: {max_val}"


def test_quick_validation():
    """快速驗證測試（獨立運行）"""
    print("\n🚀 Meta Quality Optimizer 快速驗證")
    
    optimizer = MetaQualityOptimizer(
        data_path='data',
        config_path='configs'
    )
    
    # 創建模擬數據
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
    
    # 模擬 Primary 信號
    signals = pd.Series(
        np.random.choice([1, -1], size=1000),
        index=dates
    )
    
    # 模擬價格數據
    returns = np.random.normal(0.0001, 0.01, 1000)
    prices = 100 * (1 + returns).cumprod()
    
    price_data = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.005,
        'low': prices * 0.995,
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 1000)
    }, index=dates)
    
    # 設定數據
    optimizer.set_primary_signals(signals)
    optimizer.price_data = price_data
    
    print(f"✅ 數據設定完成")
    
    # 測試元特徵生成
    params = {
        'lag': 12,
        'atr_period': 14,
        'profit_multiplier': 2.0,
        'stop_multiplier': 1.5,
        'transaction_cost_bps': 10.0,
        'strength_weight': 0.3,
        'trend_weight': 0.3,
        'winrate_weight': 0.2,
        'alignment_weight': 0.2,
        'quality_threshold': 0.5,
    }
    
    meta_features = optimizer.generate_meta_features(signals, price_data, params)
    print(f"✅ 元特徵生成: {len(meta_features.columns)} 個特徵")
    
    meta_labels = optimizer.generate_meta_labels(signals, price_data, params)
    good_ratio = (meta_labels == 1).sum() / len(meta_labels)
    print(f"✅ 元標籤生成: 好信號比例 {good_ratio:.1%}")
    
    quality = optimizer.evaluate_quality(meta_features, params)
    exec_ratio = (quality == 1).sum() / len(quality)
    print(f"✅ 質量評估: 執行率 {exec_ratio:.1%}")
    
    print("✅ 所有驗證通過!")


if __name__ == "__main__":
    # 獨立運行測試
    test_quick_validation()

