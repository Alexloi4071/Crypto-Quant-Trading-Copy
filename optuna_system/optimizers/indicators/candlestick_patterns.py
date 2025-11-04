# -*- coding: utf-8 -*-
"""
K线形态识别
包含：单根、双根、三根和复杂K线形态
"""

import warnings
from typing import Dict
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class CandlestickPatterns:
    """
    K线形态识别
    
    包含50+个形态：
    - 单根K线形态（15个）
    - 双根K线形态（10个）
    - 三根K线形态（12个）
    - 复杂形态（8个）
    """
    
    def __init__(self):
        pass
    
    def calculate_all(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有K线形态特征
        
        Args:
            ohlcv: OHLCV数据
            
        Returns:
            包含所有K线形态特征的DataFrame
        """
        if ohlcv.empty:
            return pd.DataFrame(index=ohlcv.index)
        
        features = pd.DataFrame(index=ohlcv.index)
        
        # 1. 单根K线形态
        single_patterns = self.detect_single_candle_patterns(ohlcv)
        for key, value in single_patterns.items():
            features[f'candle_single_{key}'] = value
        
        # 2. 双根K线形态
        double_patterns = self.detect_double_candle_patterns(ohlcv)
        for key, value in double_patterns.items():
            features[f'candle_double_{key}'] = value
        
        # 3. 三根K线形态
        triple_patterns = self.detect_triple_candle_patterns(ohlcv)
        for key, value in triple_patterns.items():
            features[f'candle_triple_{key}'] = value
        
        # 4. 复杂形态
        complex_patterns = self.detect_complex_patterns(ohlcv)
        for key, value in complex_patterns.items():
            features[f'candle_complex_{key}'] = value
        
        # Shift信号1期
        features = features.shift(1).fillna(0)
        
        # 清理NaN和Inf
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        return features
    
    def detect_single_candle_patterns(self, ohlcv: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        检测单根K线形态
        
        Args:
            ohlcv: OHLCV数据
            
        Returns:
            单根K线形态字典
        """
        open_price = ohlcv['open']
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        
        patterns = {}
        
        # 计算K线属性
        body = close - open_price
        body_abs = abs(body)
        upper_shadow = high - close.where(close > open_price, open_price)
        lower_shadow = open_price.where(close > open_price, close) - low
        total_range = high - low
        
        # 1. Doji（十字星）
        patterns['doji'] = (body_abs <= total_range * 0.1).astype(int)
        
        # 2. Long Legged Doji（长脚十字星）
        patterns['long_legged_doji'] = (
            (body_abs <= total_range * 0.1) &
            (upper_shadow > body_abs * 2) &
            (lower_shadow > body_abs * 2)
        ).astype(int)
        
        # 3. Dragonfly Doji（蜻蜓十字星）
        patterns['dragonfly_doji'] = (
            (body_abs <= total_range * 0.1) &
            (lower_shadow > total_range * 0.6) &
            (upper_shadow < total_range * 0.1)
        ).astype(int)
        
        # 4. Gravestone Doji（墓碑十字星）
        patterns['gravestone_doji'] = (
            (body_abs <= total_range * 0.1) &
            (upper_shadow > total_range * 0.6) &
            (lower_shadow < total_range * 0.1)
        ).astype(int)
        
        # 5. Hammer（锤子线）
        patterns['hammer'] = (
            (body_abs > 0) &
            (lower_shadow >= body_abs * 2) &
            (upper_shadow <= body_abs * 0.3) &
            (body > 0)  # 阳线
        ).astype(int)
        
        # 6. Inverted Hammer（倒锤线）
        patterns['inverted_hammer'] = (
            (body_abs > 0) &
            (upper_shadow >= body_abs * 2) &
            (lower_shadow <= body_abs * 0.3) &
            (body > 0)  # 阳线
        ).astype(int)
        
        # 7. Hanging Man（上吊线）
        patterns['hanging_man'] = (
            (body_abs > 0) &
            (lower_shadow >= body_abs * 2) &
            (upper_shadow <= body_abs * 0.3) &
            (body < 0)  # 阴线
        ).astype(int)
        
        # 8. Shooting Star（流星线）
        patterns['shooting_star'] = (
            (body_abs > 0) &
            (upper_shadow >= body_abs * 2) &
            (lower_shadow <= body_abs * 0.3) &
            (body < 0)  # 阴线
        ).astype(int)
        
        # 9. Marubozu（光头光脚）
        patterns['marubozu'] = (
            (upper_shadow <= body_abs * 0.1) &
            (lower_shadow <= body_abs * 0.1)
        ).astype(int)
        
        # 10. Spinning Top（陀螺线）
        patterns['spinning_top'] = (
            (body_abs <= total_range * 0.3) &
            (upper_shadow >= body_abs) &
            (lower_shadow >= body_abs)
        ).astype(int)
        
        # 11. High Wave（高浪线）
        patterns['high_wave'] = (
            (body_abs <= total_range * 0.2) &
            (total_range > total_range.rolling(10).mean() * 1.5)
        ).astype(int)
        
        # 12. Belt Hold（捉腰带线）
        # 看涨：开盘=最低价，长阳线
        patterns['belt_hold_bullish'] = (
            (open_price <= low * 1.001) &
            (body > total_range * 0.6) &
            (body > 0)
        ).astype(int)
        
        # 看跌：开盘=最高价，长阴线
        patterns['belt_hold_bearish'] = (
            (open_price >= high * 0.999) &
            (body < -total_range * 0.6)
        ).astype(int)
        
        # 13. Long Line（长线）
        avg_range = total_range.rolling(20).mean()
        patterns['long_line'] = (
            (body_abs > avg_range * 1.5)
        ).astype(int)
        
        # 14. Short Line（短线）
        patterns['short_line'] = (
            (body_abs < avg_range * 0.5)
        ).astype(int)
        
        # 15. Four Price Doji（四价同值）
        patterns['four_price_doji'] = (
            (open_price == high) &
            (open_price == low) &
            (open_price == close)
        ).astype(int)
        
        return patterns
    
    def detect_double_candle_patterns(self, ohlcv: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        检测双根K线形态
        
        Args:
            ohlcv: OHLCV数据
            
        Returns:
            双根K线形态字典
        """
        open_price = ohlcv['open']
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        
        patterns = {}
        
        # 当前和前一根K线
        body_1 = close.shift(1) - open_price.shift(1)
        body_2 = close - open_price
        body_abs_1 = abs(body_1)
        body_abs_2 = abs(body_2)
        
        # 1. Engulfing（吞没）
        # 看涨吞没：前阴后阳，阳线完全吞没阴线
        patterns['engulfing_bullish'] = (
            (body_1 < 0) &  # 前一根阴线
            (body_2 > 0) &  # 当前阳线
            (open_price <= close.shift(1)) &  # 开盘<=前收盘
            (close >= open_price.shift(1))  # 收盘>=前开盘
        ).astype(int)
        
        # 看跌吞没：前阳后阴，阴线完全吞没阳线
        patterns['engulfing_bearish'] = (
            (body_1 > 0) &  # 前一根阳线
            (body_2 < 0) &  # 当前阴线
            (open_price >= close.shift(1)) &  # 开盘>=前收盘
            (close <= open_price.shift(1))  # 收盘<=前开盘
        ).astype(int)
        
        # 2. Harami（孕线）
        # 看涨孕线：前长阴后短阳，阳线在阴线内部
        patterns['harami_bullish'] = (
            (body_1 < 0) &
            (body_abs_1 > body_abs_2) &
            (body_2 > 0) &
            (open_price >= close.shift(1)) &
            (close <= open_price.shift(1))
        ).astype(int)
        
        # 看跌孕线：前长阳后短阴，阴线在阳线内部
        patterns['harami_bearish'] = (
            (body_1 > 0) &
            (body_abs_1 > body_abs_2) &
            (body_2 < 0) &
            (open_price <= close.shift(1)) &
            (close >= open_price.shift(1))
        ).astype(int)
        
        # 3. Piercing Line（刺透线）
        patterns['piercing_line'] = (
            (body_1 < 0) &  # 前一根阴线
            (body_2 > 0) &  # 当前阳线
            (open_price < low.shift(1)) &  # 开盘低于前低点
            (close > (open_price.shift(1) + close.shift(1)) / 2) &  # 收盘超过前K线中点
            (close < open_price.shift(1))  # 但未完全吞没
        ).astype(int)
        
        # 4. Dark Cloud Cover（乌云盖顶）
        patterns['dark_cloud_cover'] = (
            (body_1 > 0) &  # 前一根阳线
            (body_2 < 0) &  # 当前阴线
            (open_price > high.shift(1)) &  # 开盘高于前高点
            (close < (open_price.shift(1) + close.shift(1)) / 2) &  # 收盘低于前K线中点
            (close > open_price.shift(1))  # 但未完全吞没
        ).astype(int)
        
        # 5. Tweezers（镊子）
        # 镊子顶：两根K线高点相近
        patterns['tweezers_top'] = (
            (abs(high - high.shift(1)) <= (high + high.shift(1)) / 2 * 0.002) &
            (body_1 > 0) &
            (body_2 < 0)
        ).astype(int)
        
        # 镊子底：两根K线低点相近
        patterns['tweezers_bottom'] = (
            (abs(low - low.shift(1)) <= (low + low.shift(1)) / 2 * 0.002) &
            (body_1 < 0) &
            (body_2 > 0)
        ).astype(int)
        
        # 6. Counterattack（反击线）
        # 看涨反击：前阴后阳，收盘价相同
        patterns['counterattack_bullish'] = (
            (body_1 < 0) &
            (body_2 > 0) &
            (abs(close - close.shift(1)) <= close * 0.002)
        ).astype(int)
        
        # 看跌反击：前阳后阴，收盘价相同
        patterns['counterattack_bearish'] = (
            (body_1 > 0) &
            (body_2 < 0) &
            (abs(close - close.shift(1)) <= close * 0.002)
        ).astype(int)
        
        # 7. Matching Low（匹配低点）
        patterns['matching_low'] = (
            (body_1 < 0) &
            (body_2 < 0) &
            (abs(close - close.shift(1)) <= close * 0.002)
        ).astype(int)
        
        return patterns
    
    def detect_triple_candle_patterns(self, ohlcv: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        检测三根K线形态
        
        Args:
            ohlcv: OHLCV数据
            
        Returns:
            三根K线形态字典
        """
        open_price = ohlcv['open']
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        
        patterns = {}
        
        # 三根K线的实体
        body_1 = close.shift(2) - open_price.shift(2)
        body_2 = close.shift(1) - open_price.shift(1)
        body_3 = close - open_price
        
        # 1. Morning Star（启明星）
        patterns['morning_star'] = (
            (body_1 < 0) &  # 第一根阴线
            (abs(body_2) < abs(body_1) * 0.3) &  # 第二根小K线
            (body_3 > 0) &  # 第三根阳线
            (close > (open_price.shift(2) + close.shift(2)) / 2)  # 第三根收盘超过第一根中点
        ).astype(int)
        
        # 2. Evening Star（黄昏星）
        patterns['evening_star'] = (
            (body_1 > 0) &  # 第一根阳线
            (abs(body_2) < abs(body_1) * 0.3) &  # 第二根小K线
            (body_3 < 0) &  # 第三根阴线
            (close < (open_price.shift(2) + close.shift(2)) / 2)  # 第三根收盘低于第一根中点
        ).astype(int)
        
        # 3. Three White Soldiers（三个白兵）
        patterns['three_white_soldiers'] = (
            (body_1 > 0) &
            (body_2 > 0) &
            (body_3 > 0) &
            (close.shift(1) > close.shift(2)) &
            (close > close.shift(1))
        ).astype(int)
        
        # 4. Three Black Crows（三只乌鸦）
        patterns['three_black_crows'] = (
            (body_1 < 0) &
            (body_2 < 0) &
            (body_3 < 0) &
            (close.shift(1) < close.shift(2)) &
            (close < close.shift(1))
        ).astype(int)
        
        # 5. Three Inside Up（三内部上升）
        patterns['three_inside_up'] = (
            (body_1 < 0) &  # 第一根阴线
            (body_2 > 0) &  # 第二根阳线（孕线）
            (abs(body_2) < abs(body_1)) &
            (body_3 > 0) &  # 第三根阳线
            (close > close.shift(2))  # 突破第一根高点
        ).astype(int)
        
        # 6. Three Inside Down（三内部下降）
        patterns['three_inside_down'] = (
            (body_1 > 0) &  # 第一根阳线
            (body_2 < 0) &  # 第二根阴线（孕线）
            (abs(body_2) < abs(body_1)) &
            (body_3 < 0) &  # 第三根阴线
            (close < close.shift(2))  # 跌破第一根低点
        ).astype(int)
        
        # 7. Three Outside Up（三外部上升）
        patterns['three_outside_up'] = (
            (body_1 < 0) &  # 第一根阴线
            (body_2 > 0) &  # 第二根阳线（吞没）
            (open_price.shift(1) <= close.shift(2)) &
            (close.shift(1) >= open_price.shift(2)) &
            (body_3 > 0) &  # 第三根阳线
            (close > close.shift(1))
        ).astype(int)
        
        # 8. Three Outside Down（三外部下降）
        patterns['three_outside_down'] = (
            (body_1 > 0) &  # 第一根阳线
            (body_2 < 0) &  # 第二根阴线（吞没）
            (open_price.shift(1) >= close.shift(2)) &
            (close.shift(1) <= open_price.shift(2)) &
            (body_3 < 0) &  # 第三根阴线
            (close < close.shift(1))
        ).astype(int)
        
        # 9. Upside Gap Two Crows（向上跳空两只乌鸦）
        patterns['upside_gap_two_crows'] = (
            (body_1 > 0) &  # 第一根阳线
            (body_2 < 0) &  # 第二根阴线
            (low.shift(1) > high.shift(2)) &  # 跳空
            (body_3 < 0) &  # 第三根阴线
            (abs(body_3) > abs(body_2))
        ).astype(int)
        
        # 10. Three Line Strike（三线打击）
        # 看涨版本
        patterns['three_line_strike_bullish'] = (
            (body_1 < 0) &
            (body_2 < 0) &
            (body_3 < 0) &
            (close.shift(1) < close.shift(2)) &
            (close.shift(0) < close.shift(1)) &
            # 第四根K线（当前）
            (close - open_price > 0) &  # 阳线
            (close > open_price.shift(2))  # 完全吞没前三根
        ).astype(int)
        
        # 11. Abandoned Baby（弃婴）
        # 看涨弃婴
        patterns['abandoned_baby_bullish'] = (
            (body_1 < 0) &  # 第一根阴线
            (high.shift(1) < low.shift(2)) &  # 第二根与第一根跳空
            (low > high.shift(1)) &  # 第三根与第二根跳空
            (body_3 > 0)  # 第三根阳线
        ).astype(int)
        
        # 看跌弃婴
        patterns['abandoned_baby_bearish'] = (
            (body_1 > 0) &  # 第一根阳线
            (low.shift(1) > high.shift(2)) &  # 第二根与第一根跳空
            (high < low.shift(1)) &  # 第三根与第二根跳空
            (body_3 < 0)  # 第三根阴线
        ).astype(int)
        
        return patterns
    
    def detect_complex_patterns(self, ohlcv: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        检测复杂K线形态
        
        Args:
            ohlcv: OHLCV数据
            
        Returns:
            复杂形态字典
        """
        open_price = ohlcv['open']
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        
        patterns = {}
        
        # 1. Island Reversal（岛形反转）
        # 向上岛形：下跳空 -> 横盘 -> 上跳空
        gap_down = (high < low.shift(1)).astype(int)
        gap_up = (low > high.shift(1)).astype(int)
        
        patterns['island_reversal_bullish'] = (
            gap_down.shift(5).rolling(3).sum() > 0
        ) & (gap_up.rolling(2).sum() > 0).astype(int)
        
        # 向下岛形：上跳空 -> 横盘 -> 下跳空
        patterns['island_reversal_bearish'] = (
            gap_up.shift(5).rolling(3).sum() > 0
        ) & (gap_down.rolling(2).sum() > 0).astype(int)
        
        # 2. Rising Three Methods（上升三法）
        body = close - open_price
        patterns['rising_three_methods'] = (
            (body.shift(4) > 0) &  # 第一根阳线
            (body.shift(3) < 0) &  # 三根小阴线
            (body.shift(2) < 0) &
            (body.shift(1) < 0) &
            (close.shift(1) > close.shift(4) * 0.9) &  # 小阴线在第一根范围内
            (body > 0) &  # 第五根阳线
            (close > close.shift(4))  # 突破第一根高点
        ).astype(int)
        
        # 3. Falling Three Methods（下降三法）
        patterns['falling_three_methods'] = (
            (body.shift(4) < 0) &  # 第一根阴线
            (body.shift(3) > 0) &  # 三根小阳线
            (body.shift(2) > 0) &
            (body.shift(1) > 0) &
            (close.shift(1) < close.shift(4) * 1.1) &  # 小阳线在第一根范围内
            (body < 0) &  # 第五根阴线
            (close < close.shift(4))  # 跌破第一根低点
        ).astype(int)
        
        # 4. Mat Hold（铺垫Hold）
        patterns['mat_hold'] = (
            (body.shift(4) > 0) &  # 第一根大阳线
            (abs(body.shift(4)) > abs(body).rolling(20).mean() * 1.5) &
            (body.shift(3) < 0) &  # 后三根小K线
            (abs(body.shift(3)) < abs(body.shift(4)) * 0.3) &
            (abs(body.shift(2)) < abs(body.shift(4)) * 0.3) &
            (abs(body.shift(1)) < abs(body.shift(4)) * 0.3) &
            (body > 0) &  # 第五根阳线
            (close > close.shift(4))
        ).astype(int)
        
        # 5. Stick Sandwich（夹心线）
        patterns['stick_sandwich'] = (
            (body.shift(2) < 0) &  # 第一根阴线
            (body.shift(1) > 0) &  # 第二根阳线
            (body < 0) &  # 第三根阴线
            (abs(close - close.shift(2)) <= close * 0.002)  # 第一和第三根收盘价相同
        ).astype(int)
        
        # 6. Breakaway（突围）
        # 看涨突围
        patterns['breakaway_bullish'] = (
            (body.shift(4) < 0) &  # 第一根阴线
            (high.shift(3) < low.shift(4)) &  # 跳空
            (body.shift(3) < 0) &  # 连续下跌
            (body.shift(2) < 0) &
            (body.shift(1) < 0) &
            (body > 0) &  # 第五根阳线
            (close > open_price.shift(4))  # 填补跳空
        ).astype(int)
        
        # 7. Ladder Top（梯顶）
        patterns['ladder_top'] = (
            (body.shift(4) > 0) &  # 连续三根阳线
            (body.shift(3) > 0) &
            (body.shift(2) > 0) &
            (close.shift(2) > close.shift(3)) &
            (close.shift(3) > close.shift(4)) &
            (body.shift(1) > 0) &  # 第四根小阳线
            (abs(body.shift(1)) < abs(body.shift(2)) * 0.5) &
            (body < 0)  # 第五根阴线
        ).astype(int)
        
        # 8. Two Gapping（双跳空）
        patterns['two_gapping'] = (
            (low.shift(1) > high.shift(2)) &  # 第一个跳空
            (low > high.shift(1))  # 第二个跳空
        ).astype(int)
        
        return patterns


def main():
    """测试Candlestick Patterns实现"""
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
    
    # 测试Candlestick Patterns
    cp = CandlestickPatterns()
    features = cp.calculate_all(ohlcv)
    
    print("Candlestick Pattern Features Generated:")
    print(f"Total features: {features.shape[1]}")
    print(f"\nFeature columns (first 20):")
    for col in features.columns[:20]:
        print(f"  - {col}")
    
    # 显示一些形态统计
    print(f"\nPattern Statistics (patterns with signals > 0):")
    for col in features.columns:
        count = features[col].sum()
        if count > 0:
            print(f"  {col}: {count}")


if __name__ == '__main__':
    main()

