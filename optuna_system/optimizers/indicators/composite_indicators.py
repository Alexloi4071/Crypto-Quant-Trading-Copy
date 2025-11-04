# -*- coding: utf-8 -*-
"""
复合指标
包含：SMI, Elder Ray, CMO, KST, TSI, AO, 统计指标等
"""

import warnings
from typing import Dict
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class CompositeIndicators:
    """
    复合指标实现
    
    包含15+个指标
    """
    
    def __init__(self):
        pass
    
    def calculate_all(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """计算所有复合指标"""
        if ohlcv.empty:
            return pd.DataFrame(index=ohlcv.index)
        
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        
        features = pd.DataFrame(index=ohlcv.index)
        
        # 1. Stochastic Momentum Index
        features['smi'] = self.calculate_smi(high, low, close)
        features['smi_signal'] = features['smi'].ewm(span=3).mean()
        
        # 2. Elder Ray
        bull_power, bear_power = self.calculate_elder_ray(high, low, close)
        features['elder_bull_power'] = bull_power
        features['elder_bear_power'] = bear_power
        
        # 3. Chande Momentum Oscillator
        features['cmo'] = self.calculate_cmo(close)
        
        # 4. Know Sure Thing
        features['kst'] = self.calculate_kst(close)
        features['kst_signal'] = features['kst'].rolling(9).mean()
        
        # 5. True Strength Index
        features['tsi'] = self.calculate_tsi(close)
        features['tsi_signal'] = features['tsi'].ewm(span=7).mean()
        
        # 6. Awesome Oscillator
        features['ao'] = self.calculate_ao(high, low)
        
        # 7. 统计指标
        stats = self.calculate_statistical_features(close)
        for key, value in stats.items():
            features[f'stat_{key}'] = value
        
        # 清理
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def calculate_smi(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     k_period: int = 10, d_period: int = 3) -> pd.Series:
        """Stochastic Momentum Index"""
        ll = low.rolling(k_period).min()
        hh = high.rolling(k_period).max()
        diff = hh - ll
        rdiff = close - (hh + ll) / 2
        
        avgrdiff = rdiff.ewm(span=d_period).mean().ewm(span=d_period).mean()
        avgdiff = diff.ewm(span=d_period).mean().ewm(span=d_period).mean()
        
        smi = 100 * avgrdiff / (avgdiff / 2 + 1e-9)
        return smi
    
    def calculate_elder_ray(self, high: pd.Series, low: pd.Series, close: pd.Series,
                           period: int = 13) -> tuple:
        """Elder Ray Index"""
        ema = close.ewm(span=period).mean()
        bull_power = high - ema
        bear_power = low - ema
        return bull_power, bear_power
    
    def calculate_cmo(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Chande Momentum Oscillator"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        sum_gain = gain.rolling(period).sum()
        sum_loss = loss.rolling(period).sum()
        
        cmo = 100 * (sum_gain - sum_loss) / (sum_gain + sum_loss + 1e-9)
        return cmo
    
    def calculate_kst(self, close: pd.Series) -> pd.Series:
        """Know Sure Thing"""
        roc1 = close.pct_change(10).rolling(10).mean()
        roc2 = close.pct_change(15).rolling(10).mean()
        roc3 = close.pct_change(20).rolling(10).mean()
        roc4 = close.pct_change(30).rolling(15).mean()
        
        kst = (roc1 * 1) + (roc2 * 2) + (roc3 * 3) + (roc4 * 4)
        return kst * 100
    
    def calculate_tsi(self, close: pd.Series, long_period: int = 25, 
                     short_period: int = 13) -> pd.Series:
        """True Strength Index"""
        momentum = close.diff()
        
        double_smoothed_pc = momentum.ewm(span=long_period).mean().ewm(span=short_period).mean()
        double_smoothed_abs_pc = momentum.abs().ewm(span=long_period).mean().ewm(span=short_period).mean()
        
        tsi = 100 * double_smoothed_pc / (double_smoothed_abs_pc + 1e-9)
        return tsi
    
    def calculate_ao(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """Awesome Oscillator"""
        median_price = (high + low) / 2
        ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
        return ao
    
    def calculate_statistical_features(self, close: pd.Series, window: int = 20) -> Dict:
        """统计特征"""
        stats = {}
        
        # Z-Score
        rolling_mean = close.rolling(window).mean()
        rolling_std = close.rolling(window).std()
        stats['zscore'] = (close - rolling_mean) / (rolling_std + 1e-9)
        
        # Percentile Rank
        stats['percentile'] = close.rolling(window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        
        # Skewness
        stats['skewness'] = close.rolling(window).skew()
        
        # Kurtosis
        stats['kurtosis'] = close.rolling(window).kurt()
        
        # Entropy (简化版)
        returns = close.pct_change()
        stats['return_std'] = returns.rolling(window).std()
        stats['return_skew'] = returns.rolling(window).skew()
        
        # Autocorrelation
        stats['autocorr_1'] = close.rolling(window).apply(
            lambda x: pd.Series(x).autocorr(1) if len(x) > 1 else 0, raw=False
        )
        stats['autocorr_5'] = close.rolling(window).apply(
            lambda x: pd.Series(x).autocorr(5) if len(x) > 5 else 0, raw=False
        )
        
        # Hurst Exponent (简化版)
        stats['hurst'] = close.rolling(window).apply(self._calculate_hurst, raw=False)
        
        return stats
    
    def _calculate_hurst(self, ts):
        """Calculate Hurst Exponent (simplified)"""
        try:
            lags = range(2, min(10, len(ts)//2))
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            if len(tau) > 0 and all(t > 0 for t in tau):
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
        except:
            pass
        return 0.5


def main():
    """测试"""
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='1D')
    np.random.seed(42)
    close_prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
    ohlcv = pd.DataFrame({
        'high': close_prices * 1.01,
        'low': close_prices * 0.99,
        'close': close_prices,
    }, index=dates)
    
    ci = CompositeIndicators()
    features = ci.calculate_all(ohlcv)
    print(f"Generated {features.shape[1]} composite indicator features")
    print(features.columns.tolist())


if __name__ == '__main__':
    main()

