"""
主策略配置
整合幣種和時框配置，生成最終參數
"""

from .symbol_profiles import (VOLATILITY_MULTIPLIERS,
                              LIQUIDITY_ADJUSTMENTS)
from .timeframe_profiles import SAMPLE_DENSITY_ADJUSTMENTS

# 默認試驗數配置
DEFAULT_TRIALS_CONFIG = {
    'layer1_base_trials': 120,
    'layer2_base_trials': 100,
    'stage1_ratio': 0.6,  # 第一層階段1占比
    'stage2_ratio': 0.4   # 第一層階段2占比
}

# LightGBM 參數範圍配置
LIGHTGBM_PARAM_RANGES = {
    'num_leaves': (20, 100),
    'max_depth': (4, 15),
    'learning_rate': (0.01, 0.3),
    'n_estimators': (100, 1000),
    'reg_alpha': (0.0, 1.0),
    'reg_lambda': (0.0, 1.0),
    'feature_fraction': (0.6, 1.0),
    'bagging_fraction': (0.6, 1.0),
    'bagging_freq': (1, 7),
    'min_child_samples': (5, 100)
}

# 特徵數量偏好映射
FEATURE_COUNT_PREFERENCES = {
    'low': 0.7,      # 減少30%特徵
    'medium': 1.0,   # 標準特徵數
    'high': 1.3      # 增加30%特徵
}

# 滯後期敏感度映射
LAG_SENSITIVITY_MULTIPLIERS = {
    'low': 0.7,
    'medium': 1.0,
    'high': 1.3,
    'very_high': 1.6
}
