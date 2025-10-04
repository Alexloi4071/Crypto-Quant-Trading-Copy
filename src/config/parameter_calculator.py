"""
參數計算器
根據幣種和時框特性動態計算優化參數
"""

from typing import Dict, Tuple, Any, List

from config.symbol_profiles import SYMBOL_PROFILES
from config.timeframe_profiles import TIMEFRAME_PROFILES
from config.strategy_config import (
    DEFAULT_TRIALS_CONFIG,
    LIGHTGBM_PARAM_RANGES,
    FEATURE_COUNT_PREFERENCES,
    LAG_SENSITIVITY_MULTIPLIERS,
    VOLATILITY_MULTIPLIERS,
    LIQUIDITY_ADJUSTMENTS,
    SAMPLE_DENSITY_ADJUSTMENTS
)


class ParameterCalculator:

    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.symbol_profile = SYMBOL_PROFILES.get(
            symbol, SYMBOL_PROFILES['TEMPLATE'])
        self.timeframe_profile = TIMEFRAME_PROFILES[timeframe]

    def calculate_all_parameters(self) -> Dict[str, Any]:
        """計算所有優化參數 (支援兩層優化、多目標優化)"""
        params = {
            # 標籤參數 (擴充結構)
            'label_lag_range': self._calculate_label_lag_range(),
            'label_threshold_range': self._calculate_label_threshold_range(),

            # 特徵參數
            'n_features_range': self._calculate_feature_count_range(),

            # 訓練參數
            'cv_folds': self._calculate_cv_folds(),
            'training_period': self._calculate_training_period(),

            # 🆕 兩階段多目標試驗數配置
            'trials_config': self._calculate_trials_config(),

            # 🆕 Warm-Start種子參數
            'seed_params': self._calculate_seed_params(),

            # 模型參數範圍 (自適應調整)
            'model_param_ranges': self._calculate_model_param_ranges(),

            # 🆕 多目標優化方向
            'optimization_directions': self._calculate_optimization_directions(),

            # 元數據
            'rationale': self._generate_rationale()
        }

        return params

    def _calculate_label_lag_range(self) -> Tuple[int, int, int]:
        """計算標籤滯後期範圍 (min, max, warm_start)"""
        base_min, base_max = self.timeframe_profile['base_lag_range']

        # 根據滯後期敏感度調整
        sensitivity = self.symbol_profile['lag_sensitivity']
        multiplier = LAG_SENSITIVITY_MULTIPLIERS[sensitivity]

        # 計算調整後範圍
        adjusted_min = max(1, int(base_min))
        adjusted_max = max(adjusted_min + 2, int(base_max * multiplier))
        
        # 🌱 計算warm_start經驗值（中位數偏向較小值，以供第一層優化初始種子）
        warm_start_lag = int(adjusted_min + (adjusted_max - adjusted_min) * 0.4)

        return (adjusted_min, adjusted_max, warm_start_lag)

    def _calculate_label_threshold_range(self) -> Dict[str, float]:
        """計算標籤閾值範圍 (分階段優化結構：min, mid, max)"""
        base_min, base_max = self.timeframe_profile['base_threshold_range']

        # 根據波動率調整
        volatility = self.symbol_profile['volatility_level']
        vol_multiplier = VOLATILITY_MULTIPLIERS[volatility]

        # 根據幣種特定乘數調整
        symbol_multiplier = self.symbol_profile['base_threshold_multiplier']

        # 綜合調整
        total_multiplier = vol_multiplier * symbol_multiplier

        adjusted_min = base_min * total_multiplier
        adjusted_max = base_max * total_multiplier
        
        # 🎯 計算中間值（以供分階段優化時使用固定中間值）
        adjusted_mid = (adjusted_min + adjusted_max) / 2

        return {
            'min': round(adjusted_min, 5),
            'mid': round(adjusted_mid, 5),
            'max': round(adjusted_max, 5)
        }

    def _calculate_feature_count_range(self) -> Tuple[int, int]:
        """計算特徵數量範圍"""
        base_count = self.timeframe_profile['base_feature_count']

        # 根據特徵偏好調整
        preference = self.symbol_profile['feature_count_preference']
        pref_multiplier = FEATURE_COUNT_PREFERENCES[preference]

        # 根據樣本密度調整
        density = self.timeframe_profile['sample_density']
        density_adjustment = (
            SAMPLE_DENSITY_ADJUSTMENTS[density]['feature_reduction'])

        # 計算調整後數量
        adjusted_count = int(
            base_count * pref_multiplier * density_adjustment)

        # 設置範圍（±20%）
        min_features = max(10, int(adjusted_count * 0.8))
        max_features = min(60, int(adjusted_count * 1.2))

        return (min_features, max_features)

    def _calculate_cv_folds(self) -> int:
        """計算交叉驗證折數"""
        base_folds = self.timeframe_profile['cv_folds']

        # 根據樣本密度微調
        density = self.timeframe_profile['sample_density']
        if density in ['extremely_low', 'very_low']:
            return max(3, base_folds - 1)
        elif density in ['very_high']:
            return min(7, base_folds + 1)

        return base_folds

    def _calculate_training_period(self) -> int:
        """計算建議訓練期（月）"""
        base_period = self.timeframe_profile['training_period_months']

        # 根據市值等級調整
        tier = self.symbol_profile['market_cap_tier']
        if tier == 1:  # 大市值，數據更穩定，可用較短期間
            return base_period
        elif tier == 2:  # 中市值，需要更多數據
            return int(base_period * 1.2)
        else:  # 小市值，需要更多歷史數據
            return int(base_period * 1.5)

    def _calculate_trials_config(self) -> Dict[str, int]:
        """計算試驗數配置"""
        # 基礎試驗數
        base_l1 = DEFAULT_TRIALS_CONFIG['layer1_base_trials']
        base_l2 = DEFAULT_TRIALS_CONFIG['layer2_base_trials']

        # 根據時框調整
        tf_multiplier = self.timeframe_profile['trials_multiplier']

        # 根據市值等級調整
        tier = self.symbol_profile['market_cap_tier']
        tier_multiplier = 1.0 if tier == 1 else (0.9 if tier == 2 else 0.8)

        total_multiplier = tf_multiplier * tier_multiplier

        layer1_trials = int(base_l1 * total_multiplier)
        layer2_trials = int(base_l2 * total_multiplier)

        # 🎯 分配第一層的多目標與單目標試驗數 (60% 多目標, 40% 單目標)  
        stage1_multi_trials = int(layer1_trials * 0.6)
        stage1_single_trials = int(layer1_trials * 0.4)

        return {
            'layer1_total': layer1_trials,
            'layer2_total': layer2_trials,
            'stage1_multi_trials': stage1_multi_trials,      # 🆕 多目標階段
            'stage1_single_trials': stage1_single_trials,    # 🆕 單目標階段
            'stage1_trials': int(
                layer1_trials * DEFAULT_TRIALS_CONFIG['stage1_ratio']),
            'stage2_trials': int(
                layer1_trials * DEFAULT_TRIALS_CONFIG['stage2_ratio'])
        }

    def _calculate_model_param_ranges(self) -> Dict[str, Tuple]:
        """計算模型參數範圍"""
        base_ranges = LIGHTGBM_PARAM_RANGES.copy()

        # 根據樣本密度調整
        density = self.timeframe_profile['sample_density']
        liquidity = self.symbol_profile['liquidity_level']

        # 調整 min_child_samples（樣本少時需要更大值）
        if density in ['extremely_low', 'very_low']:
            min_val, max_val = base_ranges['min_child_samples']
            base_ranges['min_child_samples'] = (
                int(min_val * 1.5), int(max_val * 1.2))

        # 調整正則化參數（根據流動性）
        liq_adj = (
            LIQUIDITY_ADJUSTMENTS[liquidity]['regularization_boost'])
        if liq_adj != 1.0:
            for param in ['reg_alpha', 'reg_lambda']:
                min_val, max_val = base_ranges[param]
                base_ranges[param] = (
                    min_val, min(1.0, max_val * liq_adj))

        return base_ranges

    def _generate_rationale(self) -> Dict[str, str]:
        """生成參數選擇理由"""
        return {
            'symbol_characteristics': (
                f"波動率: {self.symbol_profile['volatility_level']}, "
                f"流動性: {self.symbol_profile['liquidity_level']}, "
                f"市值等級: {self.symbol_profile['market_cap_tier']}"),
            'timeframe_characteristics': (
                f"樣本密度: {self.timeframe_profile['sample_density']}, "
                f"趨勢持續性: {self.timeframe_profile['trend_persistence']}"),
            'lag_range_rationale': (
                f"基於滯後敏感度 {self.symbol_profile['lag_sensitivity']} 調整"),
            'threshold_rationale': (
                f"基於波動率 {self.symbol_profile['volatility_level']} "
                f"和幣種乘數 "
                f"{self.symbol_profile['base_threshold_multiplier']} 調整"),
            'feature_count_rationale': (
                f"基於特徵偏好 {self.symbol_profile['feature_count_preference']} "
                "和樣本密度調整")
        }

    def print_parameter_summary(self):
        """打印參數摘要"""
        params = self.calculate_all_parameters()

        print("\n" + "=" * 60)
        print(f"參數配置摘要: {self.symbol} {self.timeframe}")
        print("=" * 60)

        print("\n📊 標籤參數:")
        print(f"  滯後期範圍: {params['label_lag_range']}")
        print(f"  閾值範圍: {params['label_threshold_range']}")

        print("\n🎯 特徵參數:")
        print(f"  特徵數範圍: {params['n_features_range']}")

        print("\n🔧 訓練參數:")
        print(f"  交叉驗證: {params['cv_folds']} 折")
        print(f"  訓練期: {params['training_period']} 個月")

        print("\n⚡ 試驗配置:")
        trials = params['trials_config']
        print(f"  第一層: {trials['layer1_total']} "
              f"(階段1: {trials['stage1_trials']}, "
              f"階段2: {trials['stage2_trials']})")
        print(f"  第二層: {trials['layer2_total']}")

        print("\n💡 配置理由:")
        for key, reason in params['rationale'].items():
            print(f"  {key}: {reason}")

        print("=" * 60 + "\n")

    def _calculate_seed_params(self) -> Dict[str, Any]:
        """計算Warm-Start種子參數 (與AdvancedOptunaOptimizer.TIMEFRAME_SEEDS整合)"""
        # 從標籤範圍獲取warm_start值
        lag_min, lag_max, warm_start_lag = self._calculate_label_lag_range()
        threshold_dict = self._calculate_label_threshold_range()
        feature_min, feature_max = self._calculate_feature_count_range()
        
        # 計算warm_start特徵數（偏向較少特徵）
        warm_start_features = int(feature_min + (feature_max - feature_min) * 0.3)
        
        # 根據時框特性決定標籤類型偏好
        trend_persistence = self.timeframe_profile['trend_persistence']
        if trend_persistence in ['very_high', 'high']:
            preferred_label_type = 'multiclass'  # 高持續性適合多分類
        else:
            preferred_label_type = 'binary'      # 低持續性適合二分類
        
        return {
            'label_lag': warm_start_lag,
            'label_threshold': threshold_dict['mid'],  # 使用中間值
            'label_type': preferred_label_type,
            'n_features': warm_start_features
        }

    def _calculate_optimization_directions(self) -> List[str]:
        """計算多目標優化方向"""
        # 根據文檔要求：最大化平均得分 & 最小化得分方差（即最大化一致性）
        return ['maximize', 'maximize']  # [avg_score, consistency_score]