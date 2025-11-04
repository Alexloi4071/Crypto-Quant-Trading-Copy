# -*- coding: utf-8 -*-
"""高级趋势指标"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')


class AdvancedTrend:
    """高级趋势指标：Hull MA, TEMA, DEMA, ZLEMA, KAMA, ALMA, Ichimoku, SuperTrend, Aroon, Vortex"""
    
    def calculate_all(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if ohlcv.empty:
            return pd.DataFrame(index=ohlcv.index)
        
        high, low, close = ohlcv['high'], ohlcv['low'], ohlcv['close']
        features = pd.DataFrame(index=ohlcv.index)
        
        for period in [9, 21, 50]:
            features[f'hull_ma_{period}'] = self.calculate_hull_ma(close, period)
            features[f'tema_{period}'] = self.calculate_tema(close, period)
            features[f'dema_{period}'] = self.calculate_dema(close, period)
            features[f'kama_{period}'] = self.calculate_kama(close, period)
        
        # Ichimoku
        ichimoku = self.calculate_ichimoku_full(high, low, close)
        for key, value in ichimoku.items():
            features[f'ichimoku_{key}'] = value
        
        # SuperTrend
        atr = self.calculate_atr(high, low, close, 10)
        features['supertrend'] = self.calculate_supertrend(high, low, close, atr)
        
        # Aroon
        aroon_up, aroon_down, aroon_osc = self.calculate_aroon_full(high, low)
        features['aroon_up'] = aroon_up
        features['aroon_down'] = aroon_down
        features['aroon_osc'] = aroon_osc
        
        # Vortex
        vi_plus, vi_minus = self.calculate_vortex(high, low, close)
        features['vortex_plus'] = vi_plus
        features['vortex_minus'] = vi_minus
        
        return features.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
    
    def calculate_hull_ma(self, close: pd.Series, period: int) -> pd.Series:
        wma_half = close.rolling(period//2).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
        wma_full = close.rolling(period).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
        raw_hma = 2 * wma_half - wma_full
        hma = raw_hma.rolling(int(np.sqrt(period))).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
        return hma
    
    def calculate_tema(self, close: pd.Series, period: int) -> pd.Series:
        ema1 = close.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()
        return 3 * ema1 - 3 * ema2 + ema3
    
    def calculate_dema(self, close: pd.Series, period: int) -> pd.Series:
        ema1 = close.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        return 2 * ema1 - ema2
    
    def calculate_kama(self, close: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
        change = abs(close - close.shift(period))
        volatility = close.diff().abs().rolling(period).sum()
        er = change / (volatility + 1e-9)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = pd.Series(index=close.index, dtype=float)
        kama.iloc[0] = close.iloc[0]
        for i in range(1, len(close)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])
        return kama
    
    def calculate_ichimoku_full(self, high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
        tenkan_period, kijun_period, senkou_span_b_period = 9, 26, 52
        tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2
        kijun = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
        senkou_b = ((high.rolling(senkou_span_b_period).max() + low.rolling(senkou_span_b_period).min()) / 2).shift(kijun_period)
        chikou = close.shift(-kijun_period)
        
        return {
            'tenkan': tenkan, 'kijun': kijun, 'senkou_a': senkou_a,
            'senkou_b': senkou_b, 'chikou': chikou,
            'cloud_top': senkou_a.where(senkou_a > senkou_b, senkou_b),
            'cloud_bottom': senkou_a.where(senkou_a < senkou_b, senkou_b),
            'tk_cross': ((tenkan > kijun).astype(int) - (tenkan < kijun).astype(int)),
            'price_cloud': ((close > senkou_a) & (close > senkou_b)).astype(int)
        }
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def calculate_supertrend(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                            atr: pd.Series, multiplier: float = 3.0) -> pd.Series:
        hl_avg = (high + low) / 2
        upper = hl_avg + multiplier * atr
        lower = hl_avg - multiplier * atr
        
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(1, index=close.index)
        
        for i in range(1, len(close)):
            if close.iloc[i] > upper.iloc[i-1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
            
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lower.iloc[i]
            else:
                supertrend.iloc[i] = upper.iloc[i]
        
        return (close - supertrend) / (close + 1e-9)
    
    def calculate_aroon_full(self, high: pd.Series, low: pd.Series, period: int = 25) -> tuple:
        aroon_up = high.rolling(period+1).apply(lambda x: (period - x.argmax()) / period * 100, raw=False)
        aroon_down = low.rolling(period+1).apply(lambda x: (period - x.argmin()) / period * 100, raw=False)
        aroon_osc = aroon_up - aroon_down
        return aroon_up, aroon_down, aroon_osc
    
    def calculate_vortex(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple:
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        vm_plus = (high - low.shift(1)).abs()
        vm_minus = (low - high.shift(1)).abs()
        
        vi_plus = vm_plus.rolling(period).sum() / (tr.rolling(period).sum() + 1e-9)
        vi_minus = vm_minus.rolling(period).sum() / (tr.rolling(period).sum() + 1e-9)
        return vi_plus, vi_minus


def main():
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='1D')
    np.random.seed(42)
    close_prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
    ohlcv = pd.DataFrame({'high': close_prices * 1.01, 'low': close_prices * 0.99, 'close': close_prices}, index=dates)
    at = AdvancedTrend()
    features = at.calculate_all(ohlcv)
    print(f"Generated {features.shape[1]} advanced trend features")


if __name__ == '__main__':
    main()

