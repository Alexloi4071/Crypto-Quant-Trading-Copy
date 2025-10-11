"""
時框特性配置
定義每個時框的基礎特性
"""

TIMEFRAME_PROFILES = {
    '5m': {
        'sample_density': 'very_high',         # 樣本密度
        'noise_level': 'high',                 # 噪聲水平
        'trend_persistence': 'low',            # 趨勢持續性
        'base_lag_range': (4, 24),             # 基礎滯後期範圍（擴大）
        'base_threshold_range': (0.003, 0.015),  # 基礎閾值範圍
        'buy_quantile_range': (0.65, 0.85),
        'sell_quantile_range': (0.15, 0.35),
        'lookback_range': (500, 1300),
        'base_feature_count': 30,              # 基礎特徵數量
        'training_period_months': 3,  # 建議訓練期
        'cv_folds': 3,  # 交叉驗證折數
        'trials_multiplier': 1.2  # trial 數量乘數
    },

    '15m': {
        'sample_density': 'high',
        'noise_level': 'medium_high',
        'trend_persistence': 'medium_low',
        'base_lag_range': (8, 36),
        'base_threshold_range': (0.005, 0.025),
        'buy_quantile_range': (0.68, 0.88),
        'sell_quantile_range': (0.12, 0.32),
        'lookback_range': (400, 900),
        'base_feature_count': 35,
        'training_period_months': 4,
        'cv_folds': 3,
        'trials_multiplier': 1.1
    },

    '1h': {
        'sample_density': 'medium',
        'noise_level': 'medium',
        'trend_persistence': 'medium',
        'base_lag_range': (12, 64),
        'base_threshold_range': (0.008, 0.035),
        'buy_quantile_range': (0.70, 0.90),
        'sell_quantile_range': (0.10, 0.30),
        'lookback_range': (240, 720),
        'base_feature_count': 40,
        'training_period_months': 6,
        'cv_folds': 5,
        'trials_multiplier': 1.0
    },

    '4h': {
        'sample_density': 'low',
        'noise_level': 'medium_low',
        'trend_persistence': 'medium_high',
        'base_lag_range': (18, 96),
        'base_threshold_range': (0.012, 0.055),
        'buy_quantile_range': (0.72, 0.92),
        'sell_quantile_range': (0.08, 0.28),
        'lookback_range': (160, 480),
        'base_feature_count': 35,
        'training_period_months': 12,
        'cv_folds': 5,
        'trials_multiplier': 0.9
    },

    '1d': {
        'sample_density': 'very_low',
        'noise_level': 'low',
        'trend_persistence': 'high',
        'base_lag_range': (24, 120),
        'base_threshold_range': (0.020, 0.080),
        'buy_quantile_range': (0.75, 0.95),
        'sell_quantile_range': (0.05, 0.25),
        'lookback_range': (120, 360),
        'base_feature_count': 30,
        'training_period_months': 24,
        'cv_folds': 4,
        'trials_multiplier': 0.8
    },

    '1D': {  # 支援兩種格式
        'sample_density': 'very_low',
        'noise_level': 'low',
        'trend_persistence': 'high',
        'base_lag_range': (24, 120),
        'base_threshold_range': (0.020, 0.080),
        'buy_quantile_range': (0.75, 0.95),
        'sell_quantile_range': (0.05, 0.25),
        'lookback_range': (120, 360),
        'base_feature_count': 30,
        'training_period_months': 24,
        'cv_folds': 4,
        'trials_multiplier': 0.8
    },

    '1W': {
        'sample_density': 'extremely_low',
        'noise_level': 'very_low',
        'trend_persistence': 'very_high',
        'base_lag_range': (36, 200),      # 放寬至 65 週，捕捉更長週期
        'base_threshold_range': (0.030, 0.300),  # 放寬至 30% 波動率
        'buy_quantile_range': (0.78, 0.97),
        'sell_quantile_range': (0.03, 0.22),
        'lookback_range': (80, 220),
        'base_feature_count': 25,
        'training_period_months': 36,
        'cv_folds': 3,
        'trials_multiplier': 0.7
    }
}

# 樣本密度調整係數
SAMPLE_DENSITY_ADJUSTMENTS = {
    'extremely_low': {'feature_reduction': 0.7, 'regularization_boost': 1.4},
    'very_low': {'feature_reduction': 0.8, 'regularization_boost': 1.3},
    'low': {'feature_reduction': 0.9, 'regularization_boost': 1.2},
    'medium': {'feature_reduction': 1.0, 'regularization_boost': 1.0},
    'high': {'feature_reduction': 1.1, 'regularization_boost': 0.9},
    'very_high': {'feature_reduction': 1.2, 'regularization_boost': 0.8}
}
