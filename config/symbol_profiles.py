"""
幣種特性配置
定義每個幣種的基礎特性，影響參數計算
"""

SYMBOL_PROFILES = {
    'BTCUSDT': {
        'volatility_level': 'medium',  # 波動率等級: low/medium/high
        'liquidity_level': 'very_high',  # 流動性等級
        'market_cap_tier': 1,  # 市值等級: 1(大)/2(中)/3(小)
        'correlation_group': 'crypto_major',  # 相關性群組
        'base_threshold_multiplier': 1.0,  # 基礎閾值乘數
        'feature_count_preference': 'medium',  # 特徵數量偏好
        'lag_sensitivity': 'medium'  # 滯後期敏感度
    },

    'ETHUSDT': {
        'volatility_level': 'high',
        'liquidity_level': 'very_high',
        'market_cap_tier': 1,
        'correlation_group': 'crypto_major',
        'base_threshold_multiplier': 1.1,  # ETH 波動較大
        'feature_count_preference': 'medium',
        'lag_sensitivity': 'high'  # 對滯後期較敏感
    },

    'SOLUSDT': {
        'volatility_level': 'very_high',
        'liquidity_level': 'high',
        'market_cap_tier': 2,
        'correlation_group': 'crypto_alt',
        'base_threshold_multiplier': 1.3,  # SOL 波動更大
        'feature_count_preference': 'low',  # 避免過擬合
        'lag_sensitivity': 'very_high'
    },

    'ADAUSDT': {
        'volatility_level': 'high',
        'liquidity_level': 'medium',
        'market_cap_tier': 2,
        'correlation_group': 'crypto_alt',
        'base_threshold_multiplier': 1.15,
        'feature_count_preference': 'medium',
        'lag_sensitivity': 'high'
    },

    # 模板：新幣種配置
    'TEMPLATE': {
        'volatility_level': 'medium',
        'liquidity_level': 'medium',
        'market_cap_tier': 2,
        'correlation_group': 'crypto_alt',
        'base_threshold_multiplier': 1.0,
        'feature_count_preference': 'medium',
        'lag_sensitivity': 'medium'
    }
}


# 波動率等級映射
VOLATILITY_MULTIPLIERS = {
    'low': 0.7,
    'medium': 1.0,
    'high': 1.3,
    'very_high': 1.6
}


# 流動性等級映射
LIQUIDITY_ADJUSTMENTS = {
    'low': {
        'min_samples_multiplier': 1.5, 'regularization_boost': 1.2
    },
    'medium': {
        'min_samples_multiplier': 1.2, 'regularization_boost': 1.1
    },
    'high': {
        'min_samples_multiplier': 1.0, 'regularization_boost': 1.0
    },
    'very_high': {
        'min_samples_multiplier': 0.8, 'regularization_boost': 0.9
    }
}
