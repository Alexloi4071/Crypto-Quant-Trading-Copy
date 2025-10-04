#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
時間框架參數自動縮放系統
基於文檔"🎯 多時間框架通用系統設計.md"
歸類到config/目錄管理複雜邏輯
"""
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any

import numpy as np
import pandas as pd

from .timeframe_profiles import TIMEFRAME_PROFILES, SAMPLE_DENSITY_ADJUSTMENTS


class TimeFrameScaler:
    """完整時間框架縮放器，整合 base profiles + meta 調整."""

    timeframe_minutes = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '8h': 480,
        '1d': 1440, '1D': 1440, '3d': 4320, '1w': 10080, '1W': 10080
    }

    window_multipliers = {
        '15m': 1,
        '30m': 2,
        '1h': 4,
        '2h': 8,
        '4h': 16,
        '1d': 96,
        '1D': 96,
        '1w': 672,
        '1W': 672
    }

    base_lag_ranges = {
        '5m': (1, 12),
        '15m': (1, 20),
        '30m': (2, 24),
        '1h': (4, 48),
        '2h': (4, 60),
        '4h': (6, 72),
        '1d': (1, 60),
        '1D': (1, 60),
        '1w': (1, 52),
        '1W': (1, 52)
    }

    def __init__(self, logger=None):
        self.logger = logger

    def get_scale_factor(self, target_timeframe: str, base_timeframe: str = '15m') -> float:
        base_minutes = self.timeframe_minutes.get(base_timeframe, 15)
        target_minutes = self.timeframe_minutes.get(target_timeframe, 15)
        return target_minutes / base_minutes

    def get_profile(self, timeframe: str) -> Dict:
        profile = TIMEFRAME_PROFILES.get(timeframe)
        if profile is None:
            profile = TIMEFRAME_PROFILES.get(timeframe.lower())
        if profile is None and timeframe.upper() != timeframe:
            profile = TIMEFRAME_PROFILES.get(timeframe.upper())
        return profile or {}

    def get_base_lag_range(self, timeframe: str) -> Tuple[int, int]:
        profile = self.get_profile(timeframe)
        if profile and 'base_lag_range' in profile:
            base_min, base_max = profile['base_lag_range']
        else:
            base_min, base_max = self.base_lag_ranges.get(timeframe, (1, 20))
        return max(1, base_min), max(base_min + 1, base_max)

    def adjust_lag_range_with_meta(self, timeframe: str, base_range: Tuple[int, int], meta_vol: float) -> Tuple[int, int]:
        base_min, base_max = base_range
        # meta_vol 越高，允許上限微調 +10%，但仍夾在 base_max + 5
        extension = int(meta_vol * 50)
        extended_max = min(base_max + 5, base_max + extension)
        return base_min, max(base_min + 1, extended_max)

    def clip_lag(self, lag: int, timeframe: str, meta_vol: float = 0.02) -> int:
        base_range = self.get_base_lag_range(timeframe)
        _, max_allowed = self.adjust_lag_range_with_meta(timeframe, base_range, meta_vol)
        clipped = max(base_range[0], min(lag, max_allowed))
        if self.logger and clipped != lag:
            self.logger.warning(f"{timeframe} lag {lag} 超出允許範圍，clip→{clipped}")
        return clipped

    def get_feature_window_multiplier(self, timeframe: str) -> int:
        return self.window_multipliers.get(timeframe, 1)

    def scale_window(self, window: int, timeframe: str, cap: int = 200) -> int:
        mult = self.get_feature_window_multiplier(timeframe)
        scaled = max(1, int(window * mult))
        return min(scaled, cap)

    def apply_profile_adjustments(self, params: Dict, timeframe: str) -> Dict:
        profile = self.get_profile(timeframe)
        if not profile:
            return params

        adjusted = params.copy()
        density_key = profile.get('sample_density')
        if density_key:
            density_adj = SAMPLE_DENSITY_ADJUSTMENTS.get(density_key)
            if density_adj:
                adjusted['density_feature_reduction'] = density_adj['feature_reduction']
                adjusted['density_regularization_boost'] = density_adj['regularization_boost']

        adjusted['profile_training_period_months'] = profile.get('training_period_months')
        adjusted['profile_cv_folds'] = profile.get('cv_folds')
        adjusted['profile_trials_multiplier'] = profile.get('trials_multiplier', 1.0)

        return adjusted


class RelativeTimeUnit:
    """相對時間單位系統（文檔設計）"""
    
    @staticmethod
    def define_base_parameters() -> Dict:
        """定義基準參數（基於15m優化結果）"""
        return {
            # 標籤生成（基於15m）
            'label_lag_rtu': 12,  # 12個時間單位 = 15m*12 = 3小時
            'short_term_rtu': 4,  # 4個單位 = 1小時
            'medium_term_rtu': 12,  # 12個單位 = 3小時  
            'long_term_rtu': 24,  # 24個單位 = 6小時
            
            # 技術指標窗口（基於當前BTCUSDT_15m優化）
            'sma_windows_rtu': [4, 8, 16, 32, 64],  # 1h, 2h, 4h, 8h, 16h
            'ema_windows_rtu': [4, 8, 16, 32, 64],
            'rsi_windows_rtu': [14, 21],  # 3.5h, 5.25h
            'bb_windows_rtu': [20],  # 5h
            'volatility_windows_rtu': [10, 20, 96],  # 2.5h, 5h, 24h
            'volume_sma_window_rtu': 20,  # 5h
            
            # 交叉驗證（基於Layer4優化）
            'purge_period_rtu': 192,  # 48小時（文檔建議24-48h）
            'embargo_period_rtu': 64,  # 16小時
            
            # 市場狀態識別
            'regime_detection_window_rtu': 96,  # 24小時
            'trend_window_rtu': 32,  # 8小時
            
            # 標籤閾值（基於Layer1優化）
            'pos_quantile': 0.85,  # 85%分位數（文檔建議）
            'loss_quantile': 0.15,  # 15%分位數
            
            # 特徵選擇（基於Layer2優化）
            'coarse_k_ratio': 0.6,  # 粗選60%特徵
            'fine_k_ratio': 0.4,   # 精選40%特徵（相對於粗選）
            
            # Kelly資金管理（基於Layer5）
            'kelly_lookback_rtu': 168,  # 42小時回望
            'vol_scaling_window_rtu': 96,  # 24小時vol窗口
            
            # 集成權重更新（基於Layer6）
            'weight_update_freq_rtu': 252,  # 1週更新頻率
            
            # 置信度調整（基於Layer8）
            'confidence_decay_window_rtu': 48  # 12小時衰減
        }
    
    @staticmethod
    def convert_to_absolute(rtu_params: Dict, timeframe: str) -> Dict:
        """將RTU參數轉換為絕對參數（文檔邏輯）"""
        scaler = TimeFrameScaler()
        scale_factor = scaler.get_scale_factor(timeframe)
        
        absolute_params = {}
        for key, value in rtu_params.items():
            if key.endswith('_rtu'):
                absolute_key = key.replace('_rtu', '')
                if isinstance(value, list):
                    absolute_params[absolute_key] = [max(1, int(v / scale_factor)) for v in value]
                else:
                    absolute_params[absolute_key] = max(1, int(value / scale_factor))
            else:
                # 非時間參數保持不變（如閾值、比例）
                absolute_params[key] = value
                
        return absolute_params
    
    @staticmethod
    def get_timeframe_adaptive_thresholds(timeframe: str) -> Dict:
        """根據時間框架動態調整閾值（文檔建議）"""
        if timeframe in ['1m', '5m', '15m']:
            # 短時間框架：較小閾值（噪音多）
            return {'profit_percentile': 0.8, 'loss_percentile': 0.2}
        elif timeframe in ['1h', '2h', '4h']:
            # 中時間框架：中等閾值
            return {'profit_percentile': 0.85, 'loss_percentile': 0.15}
        else:
            # 長時間框架：較大閾值（趨勢明顯）
            return {'profit_percentile': 0.9, 'loss_percentile': 0.1}


class MultiTimeframeCoordinator:
    """多時間框架統一協調器（文檔設計）"""
    
    def __init__(self, symbol: str = 'BTCUSDT', data_path: str = 'data'):
        self.symbol = symbol
        self.data_path = Path(data_path)
        self.supported_timeframes = ['15m', '1h', '4h', '1d']
        self.base_config = RelativeTimeUnit.define_base_parameters()
        self.scaler = TimeFrameScaler()
        self.global_vol: Dict[str, float] = {}
        self.meta_vol: float = 0.02
        self._build_meta_study()

    def _build_meta_study(self) -> None:
        """計算多時框波動指標"""
        vols = {}
        for tf in self.supported_timeframes:
            try:
                ohlcv_file = self.data_path / 'raw' / self.symbol / f"{self.symbol}_{tf}_ohlcv.parquet"
                if not ohlcv_file.exists():
                    vols[tf] = 0.02
                    continue
                df = pd.read_parquet(ohlcv_file, engine='pyarrow')
                returns = df['close'].pct_change().dropna()
                rolling_std = returns.rolling(window=100, min_periods=50).std()
                vols[tf] = float(rolling_std.mean()) if len(rolling_std) > 0 else 0.02
            except Exception:
                vols[tf] = 0.02

        self.global_vol = vols
        if vols:
            self.meta_vol = float(np.mean(list(vols.values())))
        else:
            self.meta_vol = 0.02
        
    def get_scaled_config_for_timeframe(self, timeframe: str) -> Dict:
        """為特定時間框架生成縮放配置"""
        # 轉換RTU參數為絕對參數
        absolute_params = RelativeTimeUnit.convert_to_absolute(self.base_config, timeframe)
 
        # 添加時間框架自適應閾值
        adaptive_thresholds = RelativeTimeUnit.get_timeframe_adaptive_thresholds(timeframe)
        absolute_params.update(adaptive_thresholds)
 
        scale_factor = self.scaler.get_scale_factor(timeframe)
        minutes = self.scaler.timeframe_minutes.get(timeframe, 60)

        meta_vol = self.meta_vol
        profile = self.scaler.get_profile(timeframe)
        base_range = self.scaler.get_base_lag_range(timeframe)
        lag_min, lag_max = self.scaler.adjust_lag_range_with_meta(timeframe, base_range, meta_vol)
        absolute_params['label_lag_min'] = lag_min
        absolute_params['label_lag_max'] = lag_max
        absolute_params['label_lag'] = max(lag_min, min(absolute_params.get('label_lag', lag_min), lag_max))

        feature_range = (max(1, lag_min // 2), max(lag_min // 2 + 1, lag_max * 2))
        absolute_params['feature_lag_min'] = feature_range[0]
        absolute_params['feature_lag_max'] = feature_range[1]

        # Layer1/Layer2 分位數預設
        absolute_params['feature_profit_q_min'] = max(0.55, absolute_params.get('pos_quantile', 0.85) - 0.20)
        absolute_params['feature_profit_q_max'] = min(0.95, absolute_params.get('pos_quantile', 0.85) + 0.10)
        absolute_params['feature_loss_q_min'] = max(0.05, absolute_params.get('loss_quantile', 0.15) - 0.10)
        absolute_params['feature_loss_q_max'] = min(0.45, absolute_params.get('loss_quantile', 0.15) + 0.10)

        # lookback 窗口維持等效歷史長度（基準15m: 300-1000 bars）
        base_lookback_min, base_lookback_max = 300, 1000
        lookback_min = max(50, int(round(base_lookback_min / scale_factor)))
        lookback_max = max(lookback_min + 50, int(round(base_lookback_max / scale_factor)))
        absolute_params['lookback_window_min'] = lookback_min
        absolute_params['lookback_window_max'] = lookback_max

        # Layer0 清洗窗口
        cleaning_impute_base = max(3, int(round(10 / scale_factor)))
        absolute_params['cleaning_impute_window'] = cleaning_impute_base
        smooth_min = max(5, int(round(5 / scale_factor)))
        smooth_max = max(smooth_min + 2, int(round(20 / scale_factor)))
        absolute_params['cleaning_smooth_window_min'] = smooth_min
        absolute_params['cleaning_smooth_window_max'] = smooth_max

        # CV gap (維持固定時間距離)
        cv_gap = max(2, int(round(10 / scale_factor)))
        absolute_params['cv_gap'] = cv_gap

        # 特徵選擇比例
        coarse_ratio = min(0.9, max(0.2, absolute_params.get('coarse_k_ratio', 0.6)))
        fine_ratio = min(coarse_ratio, max(0.1, absolute_params.get('fine_k_ratio', 0.4)))
        absolute_params['coarse_k_min_ratio'] = max(0.2, coarse_ratio * 0.7)
        absolute_params['coarse_k_max_ratio'] = min(0.95, coarse_ratio * 1.1)
        absolute_params['fine_k_min_ratio'] = max(0.05, fine_ratio * 0.5)
        absolute_params['fine_k_max_ratio'] = min(absolute_params['coarse_k_max_ratio'], fine_ratio)

        # 多時框波動資訊
        absolute_params['global_vol'] = dict(self.global_vol)
        absolute_params['global_vol_current'] = self.global_vol.get(timeframe, self.meta_vol)
        absolute_params['meta_vol'] = self.meta_vol
        absolute_params['meta_timeframes'] = list(self.supported_timeframes)
        absolute_params['scale_factor'] = scale_factor
        absolute_params['timeframe_minutes'] = minutes
        absolute_params['timeframe'] = timeframe
        absolute_params = self.scaler.apply_profile_adjustments(absolute_params, timeframe)

        return absolute_params
    
    def optimize_single_timeframe(self, timeframe: str, n_trials: int = 100) -> Dict:
        """優化單個時間框架（使用縮放參數）"""
        from optuna_system.coordinator import OptunaCoordinator
        
        # 獲取縮放配置
        scaled_config = self.get_scaled_config_for_timeframe(timeframe)
        
        print(f"🚀 開始{self.symbol}_{timeframe}優化（基於15m縮放）")
        print(f"縮放參數: lag={scaled_config.get('label_lag', 12)}, "
              f"purge={scaled_config.get('purge_period', 192)}, "
              f"pos_quantile={scaled_config.get('pos_quantile', 0.85)}")
        
        # 使用縮放配置創建協調器
        coordinator = OptunaCoordinator(
            symbol=self.symbol,
            timeframe=timeframe,
            data_path='data',
            scaled_config=scaled_config
        )
        
        # 執行優化
        result = coordinator.quick_complete_optimization()
        
        # 添加時間框架信息
        result['timeframe'] = timeframe
        result['scale_factor'] = self.scaler.get_scale_factor(timeframe)
        result['scaled_config'] = scaled_config
        
        print(f"✅ {timeframe}優化完成，F1={result.get('best_score', 'N/A'):.4f}")
        return result
    
    def optimize_all_timeframes(self, n_trials: int = 100) -> Dict[str, Dict]:
        """一鍵優化所有時間框架（文檔目標）"""
        results = {}
        
        for timeframe in self.supported_timeframes:
            try:
                result = self.optimize_single_timeframe(timeframe, n_trials)
                results[timeframe] = result
            except Exception as e:
                print(f"❌ {timeframe}優化失敗: {e}")
                results[timeframe] = {'error': str(e)}
        
        # 生成對比報告
        self._generate_comparison_report(results)
        return results
    
    def _generate_comparison_report(self, results: Dict[str, Dict]):
        """生成時間框架對比報告"""
        print(f"\n📊 {self.symbol}多時間框架優化對比報告")
        print("=" * 60)
        
        for timeframe, result in results.items():
            if 'error' not in result:
                score = result.get('best_score', 'N/A')
                scale = result.get('scale_factor', 1)
                print(f"{timeframe:>4}: F1={score:.3f} (縮放係數={scale:.2f})")
            else:
                print(f"{timeframe:>4}: 失敗 - {result['error']}")
        
        print("=" * 60)


# 參數映射表（文檔表格實現）
TIMEFRAME_PARAMETER_MAPPING = {
    # 格式: 參數類型 -> {timeframe: 相對於15m的值}
    'label_lag': {
        '15m': 12,  # 3小時
        '1h': 3,    # 3小時 (12/4)
        '4h': 1,    # 4小時 (12/16, 但最小1)
        '1d': 1     # 1天 (12/96, 但最小1)
    },
    'purge_period': {
        '15m': 192,  # 48小時
        '1h': 48,    # 48小時 (192/4)  
        '4h': 12,    # 48小時 (192/16)
        '1d': 2      # 48小時 (192/96)
    },
    'sma_short': {
        '15m': 16,   # 4小時
        '1h': 4,     # 4小時
        '4h': 1,     # 4小時
        '1d': 1      # 1天
    },
    'volatility_window': {
        '15m': 96,   # 24小時
        '1h': 24,    # 24小時
        '4h': 6,     # 24小時
        '1d': 1      # 1天
    }
}
