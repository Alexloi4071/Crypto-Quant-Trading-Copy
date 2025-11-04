# -*- coding: utf-8 -*-
"""
市场结构指标
包含：枢轴点、分形、摆动点、趋势结构识别
"""

import warnings
from typing import Dict, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class MarketStructure:
    """
    市场结构指标实现
    
    包含20+个特征：
    - 多种枢轴点（Standard, Fibonacci, Camarilla, Woodie's, DeMark）
    - Williams分形
    - 摆动高低点
    - 趋势结构（HH, HL, LH, LL, MSB, CHoCH）
    """
    
    def __init__(self):
        pass
    
    def calculate_all(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有市场结构特征
        
        Args:
            ohlcv: OHLCV数据
            
        Returns:
            包含所有市场结构特征的DataFrame
        """
        if ohlcv.empty:
            return pd.DataFrame(index=ohlcv.index)
        
        high = ohlcv['high'].copy()
        low = ohlcv['low'].copy()
        close = ohlcv['close'].copy()
        
        features = pd.DataFrame(index=ohlcv.index)
        
        # 1. 枢轴点（多种类型）
        pivot_features = self.calculate_pivot_points(ohlcv)
        for key, value in pivot_features.items():
            features[f'pivot_{key}'] = value
        
        # 2. 分形
        fractal_features = self.detect_fractals(high, low)
        for key, value in fractal_features.items():
            features[f'fractal_{key}'] = value
        
        # 3. 摆动点
        swing_features = self.identify_swing_points(high, low, close)
        for key, value in swing_features.items():
            features[f'swing_{key}'] = value
        
        # 4. 趋势结构
        structure_features = self.analyze_trend_structure(high, low, close)
        for key, value in structure_features.items():
            features[f'trend_structure_{key}'] = value
        
        # 清理NaN和Inf
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def calculate_pivot_points(self, ohlcv: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算多种类型的枢轴点
        
        Args:
            ohlcv: OHLCV数据
            
        Returns:
            枢轴点特征字典
        """
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        open_price = ohlcv['open'] if 'open' in ohlcv.columns else close
        
        pivot_features = {}
        
        # 1. Standard Pivot Points
        pp_standard = (high + low + close) / 3
        pivot_features['standard_pp'] = pp_standard
        pivot_features['standard_r1'] = 2 * pp_standard - low
        pivot_features['standard_s1'] = 2 * pp_standard - high
        pivot_features['standard_r2'] = pp_standard + (high - low)
        pivot_features['standard_s2'] = pp_standard - (high - low)
        pivot_features['standard_r3'] = high + 2 * (pp_standard - low)
        pivot_features['standard_s3'] = low - 2 * (high - pp_standard)
        
        # 归一化距离（相对于收盘价）
        pivot_features['standard_pp_distance'] = (close - pp_standard) / (close + 1e-9)
        pivot_features['standard_r1_distance'] = (pivot_features['standard_r1'] - close) / (close + 1e-9)
        pivot_features['standard_s1_distance'] = (close - pivot_features['standard_s1']) / (close + 1e-9)
        
        # 2. Fibonacci Pivot Points
        pp_fib = pp_standard
        pivot_range = high - low
        pivot_features['fib_pp'] = pp_fib
        pivot_features['fib_r1'] = pp_fib + 0.382 * pivot_range
        pivot_features['fib_r2'] = pp_fib + 0.618 * pivot_range
        pivot_features['fib_r3'] = pp_fib + pivot_range
        pivot_features['fib_s1'] = pp_fib - 0.382 * pivot_range
        pivot_features['fib_s2'] = pp_fib - 0.618 * pivot_range
        pivot_features['fib_s3'] = pp_fib - pivot_range
        
        # 3. Camarilla Pivot Points
        pp_cam = close
        pivot_features['cam_pp'] = pp_cam
        pivot_features['cam_r1'] = close + (high - low) * 1.1 / 12
        pivot_features['cam_r2'] = close + (high - low) * 1.1 / 6
        pivot_features['cam_r3'] = close + (high - low) * 1.1 / 4
        pivot_features['cam_r4'] = close + (high - low) * 1.1 / 2
        pivot_features['cam_s1'] = close - (high - low) * 1.1 / 12
        pivot_features['cam_s2'] = close - (high - low) * 1.1 / 6
        pivot_features['cam_s3'] = close - (high - low) * 1.1 / 4
        pivot_features['cam_s4'] = close - (high - low) * 1.1 / 2
        
        # 4. Woodie's Pivot Points
        pp_woodie = (high + low + 2 * close) / 4
        pivot_features['woodie_pp'] = pp_woodie
        pivot_features['woodie_r1'] = 2 * pp_woodie - low
        pivot_features['woodie_r2'] = pp_woodie + (high - low)
        pivot_features['woodie_s1'] = 2 * pp_woodie - high
        pivot_features['woodie_s2'] = pp_woodie - (high - low)
        
        # 5. DeMark Pivot Points
        # 如果收盘 < 开盘：X = high + 2*low + close
        # 如果收盘 > 开盘：X = 2*high + low + close
        # 如果收盘 = 开盘：X = high + low + 2*close
        X = pd.Series(index=close.index, dtype=float)
        X = X.where(close >= open_price, high + 2*low + close)
        X = X.where(close <= open_price, 2*high + low + close)
        X = X.fillna(high + low + 2*close)
        
        pp_demark = X / 4
        pivot_features['demark_pp'] = pp_demark
        pivot_features['demark_r1'] = X / 2 - low
        pivot_features['demark_s1'] = X / 2 - high
        
        return pivot_features
    
    def detect_fractals(self, high: pd.Series, low: pd.Series) -> Dict[str, pd.Series]:
        """
        检测Williams分形
        
        Args:
            high: 最高价
            low: 最低价
            
        Returns:
            分形特征字典
        """
        fractal_features = {}
        
        # Williams Fractal Up: 中间K线高点高于左右各2根K线
        # Williams Fractal Down: 中间K线低点低于左右各2根K线
        
        fractal_up = pd.Series(0, index=high.index)
        fractal_down = pd.Series(0, index=low.index)
        
        for i in range(2, len(high) - 2):
            # Fractal Up
            if (high.iloc[i] > high.iloc[i-1] and 
                high.iloc[i] > high.iloc[i-2] and
                high.iloc[i] > high.iloc[i+1] and 
                high.iloc[i] > high.iloc[i+2]):
                fractal_up.iloc[i] = 1
            
            # Fractal Down
            if (low.iloc[i] < low.iloc[i-1] and 
                low.iloc[i] < low.iloc[i-2] and
                low.iloc[i] < low.iloc[i+1] and 
                low.iloc[i] < low.iloc[i+2]):
                fractal_down.iloc[i] = 1
        
        fractal_features['up_signal'] = fractal_up
        fractal_features['down_signal'] = fractal_down
        
        # Fractal高低点价位
        fractal_up_price = high.where(fractal_up == 1, np.nan).fillna(method='ffill')
        fractal_down_price = low.where(fractal_down == 1, np.nan).fillna(method='ffill')
        
        fractal_features['up_level'] = fractal_up_price
        fractal_features['down_level'] = fractal_down_price
        
        return fractal_features
    
    def identify_swing_points(self, high: pd.Series, low: pd.Series, 
                             close: pd.Series, window: int = 5) -> Dict[str, pd.Series]:
        """
        识别摆动高低点
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            window: 摆动点识别窗口
            
        Returns:
            摆动点特征字典
        """
        swing_features = {}
        
        # Swing High: 窗口内的最高点
        # Swing Low: 窗口内的最低点
        
        swing_high = high.rolling(window, center=True).max()
        swing_low = low.rolling(window, center=True).min()
        
        # 标记摆动点
        is_swing_high = (high == swing_high).astype(int)
        is_swing_low = (low == swing_low).astype(int)
        
        swing_features['high_signal'] = is_swing_high
        swing_features['low_signal'] = is_swing_low
        
        # 摆动点价位（最新的摆动高低点）
        swing_high_price = high.where(is_swing_high == 1, np.nan).fillna(method='ffill')
        swing_low_price = low.where(is_swing_low == 1, np.nan).fillna(method='ffill')
        
        swing_features['high_level'] = swing_high_price
        swing_features['low_level'] = swing_low_price
        
        # 距离摆动点的距离（归一化）
        swing_features['high_distance'] = (swing_high_price - close) / (close + 1e-9)
        swing_features['low_distance'] = (close - swing_low_price) / (close + 1e-9)
        
        return swing_features
    
    def analyze_trend_structure(self, high: pd.Series, low: pd.Series, 
                               close: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
        """
        分析趋势结构
        
        识别：
        - HH (Higher High): 更高的高点
        - HL (Higher Low): 更高的低点
        - LH (Lower High): 更低的高点
        - LL (Lower Low): 更低的低点
        - MSB (Market Structure Break): 市场结构突破
        - CHoCH (Change of Character): 趋势性质改变
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            window: 分析窗口
            
        Returns:
            趋势结构特征字典
        """
        structure_features = {}
        
        # 计算滚动高低点
        rolling_high = high.rolling(window).max()
        rolling_low = low.rolling(window).min()
        
        # HH: 当前高点 > 前一个滚动高点
        prev_rolling_high = rolling_high.shift(window)
        hh = (high > prev_rolling_high).astype(int)
        
        # HL: 当前低点 > 前一个滚动低点
        prev_rolling_low = rolling_low.shift(window)
        hl = (low > prev_rolling_low).astype(int)
        
        # LH: 当前高点 < 前一个滚动高点
        lh = (high < prev_rolling_high).astype(int)
        
        # LL: 当前低点 < 前一个滚动低点
        ll = (low < prev_rolling_low).astype(int)
        
        structure_features['hh'] = hh
        structure_features['hl'] = hl
        structure_features['lh'] = lh
        structure_features['ll'] = ll
        
        # 趋势方向（基于HH/HL vs LH/LL）
        # 1: 上升趋势（HH and HL）
        # -1: 下降趋势（LH and LL）
        # 0: 盘整
        uptrend_signal = ((hh == 1) & (hl == 1)).astype(int)
        downtrend_signal = ((lh == 1) & (ll == 1)).astype(int)
        
        trend_direction = pd.Series(0, index=close.index)
        trend_direction = trend_direction.where(uptrend_signal == 0, 1)
        trend_direction = trend_direction.where(downtrend_signal == 0, -1)
        
        structure_features['trend_direction'] = trend_direction
        
        # MSB (Market Structure Break): 突破关键支撑/阻力
        # 上升趋势中突破支撑 = MSB看跌
        # 下降趋势中突破阻力 = MSB看涨
        
        support_level = rolling_low.shift(1)
        resistance_level = rolling_high.shift(1)
        
        msb_bearish = (
            (trend_direction.shift(1) == 1) &  # 之前上升趋势
            (close < support_level * 0.98)  # 突破支撑
        ).astype(int)
        
        msb_bullish = (
            (trend_direction.shift(1) == -1) &  # 之前下降趋势
            (close > resistance_level * 1.02)  # 突破阻力
        ).astype(int)
        
        structure_features['msb_bearish'] = msb_bearish
        structure_features['msb_bullish'] = msb_bullish
        
        # CHoCH (Change of Character): 趋势性质改变
        # 趋势方向突然反转
        choch = (trend_direction.diff().abs() == 2).astype(int)
        
        structure_features['choch'] = choch
        
        return structure_features


def main():
    """测试Market Structure实现"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    
    # 生成测试数据
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='1D')
    np.random.seed(42)
    
    close_prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
    ohlcv = pd.DataFrame({
        'open': close_prices * (1 + np.random.randn(len(dates)) * 0.005),
        'high': close_prices * (1 + np.abs(np.random.randn(len(dates))) * 0.01),
        'low': close_prices * (1 - np.abs(np.random.randn(len(dates))) * 0.01),
        'close': close_prices,
        'volume': np.abs(np.random.randn(len(dates))) * 1000000
    }, index=dates)
    
    # 测试Market Structure
    ms = MarketStructure()
    features = ms.calculate_all(ohlcv)
    
    print("Market Structure Features Generated:")
    print(f"Total features: {features.shape[1]}")
    print(f"\nFeature columns:")
    for col in features.columns:
        print(f"  - {col}")
    
    # 显示一些信号统计
    print(f"\nSignal Statistics:")
    signal_cols = [c for c in features.columns if 'signal' in c or 'hh' in c or 'll' in c or 'msb' in c or 'choch' in c]
    for col in signal_cols:
        count = features[col].sum()
        print(f"  {col}: {count}")
    
    print(f"\nSample data (last 10 rows):")
    print(features[['pivot_standard_pp_distance', 'trend_structure_trend_direction', 
                   'swing_high_distance']].tail(10))


if __name__ == '__main__':
    main()

